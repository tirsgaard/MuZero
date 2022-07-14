import numpy as np
from collections import deque, defaultdict
from torch.multiprocessing import Queue


class experience_replay_server:
    def __init__(self, ex_Q, experience_settings, MCTS_settings):
        self.hist_size = experience_settings["history_size"]  # The number of sequences of frames to store in memory
        self.seq_size = experience_settings["sequence_length"]  # The number of frames in each sequence
        self.K = experience_settings["K"]  # Number of steps to unroll during training. Needed here to determine delay of sending
        self.ex_Q = ex_Q  # For receiving jobs
        self.action_size = MCTS_settings["action_size"]
        self.n_actions = np.prod(self.action_size)
        self.obs_size = MCTS_settings["observation_size"]
        self.past_obs = experience_settings["past_obs"]  # Number of past observations to stack
        self.bayes = MCTS_settings["bayesian"]

        # This will be handled as a circular array,
        #     so the array start can be moved when a new sequence having to reformat the whole array
        self.N = self.hist_size*self.seq_size
        self.storage = np.empty((self.hist_size,), dtype="object")  # For priority sampling
        self.game_id = np.empty((self.N,), dtype="int32")
        self.store_id = np.empty((self.hist_size,), dtype="object")  # For retreiving frames across sequences
        for i in range(self.hist_size):
            self.store_id[i] = deque()  # Initiallize to []
        self.next_id = 0  # ID to use for next game
        self.P = np.zeros((self.N), dtype="float")  # For priority sampling
        self.total_store = 0
        self.P_replace_idx = 0
        self.agent_to_game = defaultdict(lambda: None)  # Dictionary to get what the next agent should be working on
        self.game_to_seq = defaultdict(lambda: [])  # Dictionary to game_id to sequence list

    def get_Q(self):
        return self.ex_Q

    def recv_store(self):
        S_array, a_array, r_array, done_array, v_array, pi_array, z_array, agent_id = self.ex_Q.get(True, None)
        a_array = a_array.astype(np.int64)  # Easier to convert here
        #  Convert agent id into game id
        new_game = False
        game_id = self.agent_to_game[agent_id]
        if (game_id == None) or done_array[-1]:  # Assign new id if agent has none or game is done
            game_id = self.next_id
            self.agent_to_game[agent_id] = game_id
            self.next_id += 1
            new_game = True

        # Check if removed element was the last sequence from a game
        seq_list = self.store_id[self.P_replace_idx]  # Get reference to list
        if len(seq_list) > 1:
            # Case where list cannot be overwritten
            seq_list.popleft()
        elif len(seq_list) == 1:
            # Unreference dictionary to fix memory leak
            old_game_id = self.game_id[self.P_replace_idx]
            self.game_to_seq.pop(old_game_id, None)
            seq_list.popleft()

        # It can be guaranteed this will not belong to another game,
        #     as there are more ids than sequences possible
        # Id of new sequence in game index list
        if new_game:
            # Get sequence list and add new obs
            self.game_to_seq[game_id] = self.store_id[self.P_replace_idx]
            self.store_id[self.P_replace_idx].append(self.P_replace_idx)  # Add global index to list of same games
        else:
            # Get the list corresponding to the existing game
            self.game_to_seq[game_id].append(self.P_replace_idx)

        #  Calculate importance weighting
        p = np.zeros(self.seq_size, dtype="float")
        # Pad p with 0 to avoid sampling empty states when termination has happened
        end_index = (1-self.K) if (v_array.shape[0]+1-self.past_obs)>self.seq_size else v_array.shape[0]
        val_len = v_array.shape[0] + (self.past_obs>0) - self.past_obs
        p[0:val_len] = np.abs(v_array - z_array)[(self.past_obs-1):end_index]
        start_idx = self.P_replace_idx * self.seq_size
        end_idx = (self.P_replace_idx + 1) * self.seq_size
        self.P[start_idx:end_idx] = p
        self.game_id[start_idx:end_idx] = game_id

        # Add the new elements
        self.storage[self.P_replace_idx] = [S_array, a_array, r_array, done_array, v_array, pi_array, z_array]
        # Move array start
        self.P_replace_idx = (self.P_replace_idx + 1) % self.hist_size  # sequence to next replace
        self.total_store += val_len # Update number of stored total

    def return_batches(self, batch_size, alpha, K, uniform_sampling=False):
        # Return batch_size samples sampled via prioritized sampling with parameter alpha.
        # K is the number of unrolling steps. Priorizied sampling can be converted to uniform sampling
        # But NOTE samples with error = 0 will still not be sampled
        if uniform_sampling:
            non_zero = self.P != 0
            N_count = np.sum(non_zero)  # This is also needed
            P_temp2 = self.P[non_zero] > 0  # Binarize
            P_sum = np.sum(P_temp2)
            P_temp2 = P_temp2 / P_sum
            P_temp = self.P.copy()
            P_temp[non_zero] = P_temp2
        else:
            # Normalize priority dist
            if alpha != 1:
                non_zero = self.P != 0
                N_count = np.sum(non_zero)  # This is also needed
                P_temp2 = self.P[non_zero] ** -alpha
                P_sum = np.sum(P_temp2)
                P_temp2 = P_temp2 / P_sum
                P_temp = self.P.copy()
                P_temp[non_zero] = P_temp2
            else:
                N_count = np.sum(self.P != 0)  # This is also needed
                P_sum = np.sum(self.P)
                P_temp = self.P / P_sum

        # Select index of batches
        batch_idx = np.random.choice(self.hist_size * self.seq_size, size=batch_size, p=P_temp, replace=True)  # Very slow
        S_batch, a_batch, r_batch, done_batch, pi_batch, z_batch = self.batch_sample(batch_idx, K)

        """
        # Get values from batches
        S_batch = []
        a_batch = []
        r_batch = []
        done_batch = []
        pi_batch = []
        z_batch = []
        # Loop over all batch entries
        for i in range(batch_size):
            S, a, r, done, pi, z = self.get_sample(batch_idx[i], K)
            assert(np.all(pi.sum(axis=1) == 1.))
            pad_length = self.K - len(z)
            # Pad samples if a value after termination extends into K length
            if pad_length != 0:
                a = np.pad(a, (0, pad_length), mode='constant', constant_values=np.random.randint(0, self.n_actions))
                r = np.pad(r, (0, pad_length), mode='constant', constant_values=0)
                done = np.pad(done, (0, pad_length), mode='constant', constant_values=1)
                pi = np.pad(pi, ((0, pad_length), (0, 0)), mode='constant', constant_values=1/pi.shape[-1])  # Assume equal taken action
                assert(np.all(pi.sum(axis=1) == 1.))
                z = np.pad(z, (0, pad_length), mode='constant', constant_values=0)

            S_batch.append(S)
            a_batch.append(a)
            r_batch.append(r)
            done_batch.append(done)
            pi_batch.append(pi)
            z_batch.append(z)
        # Stack batches and send. The format is (B, K, object_shape)
        S_batch = np.stack(S_batch)
        a_batch = np.stack(a_batch)
        r_batch = np.stack(r_batch)
        done_batch = np.stack(done_batch)
        pi_batch = np.stack(pi_batch)
        z_batch = np.stack(z_batch)
        """
        return S_batch, a_batch, r_batch, done_batch, pi_batch, z_batch, batch_idx, self.P[batch_idx]/P_sum, int(N_count)

    def update_weightings(self, new_weightings, indexes):
        self.P[indexes] = new_weightings

    def batch_sample(self, batch_idx, K):
        batch_size = len(batch_idx)

        hist_idx = batch_idx // self.seq_size
        seq_idx = self.past_obs - 1 + (batch_idx % self.seq_size)
        seq_idx_end = seq_idx + K
        undershoot = seq_idx + 1 - self.past_obs  # Case where past frames goes back to previous block (not happening currently)
        sequences = self.storage[hist_idx]

        S_batch = []

        # # Pre-fill array with boundry conditions
        a_batch = np.random.randint(0, self.n_actions, size=(batch_size, K))
        r_batch = np.zeros((batch_size, K), dtype=np.float32)
        done_batch = np.ones((batch_size, K))
        z_batch = np.zeros((batch_size, K), dtype=np.float32)
        if self.bayes:
            pi_batch = np.zeros((batch_size, K, self.n_actions, 2), dtype=np.float32)
            pi_batch[:, :, :, 1] = 10**-5  # Set variance to very low
        else:
        # Action density
            a_dens = 1 / self.n_actions  # Predefine for skip having to calc for all values filled in array
            pi_batch = np.full((batch_size, K, self.n_actions), a_dens, dtype=np.float32)


        S_arrays, a_arrays, r_arrays, done_arrays, v_arrays, pi_arrays, z_arrays = map(list, zip(*sequences))
        for i in range(len(sequences)):
            #S_array, a_array, r_array, done_array, v_array, pi_array, z_array = sequences[i]
            # Construct batch
            seq_start = seq_idx[i]
            seq_end = seq_idx_end[i]
            length = a_arrays[i][seq_start:seq_end].shape[0] # How much of K is before termination
            a_batch[i, 0:length] = a_arrays[i][seq_start:seq_end]
            r_batch[i, 0:length] = r_arrays[i][seq_start:seq_end]
            done_batch[i, 0:length] = done_arrays[i][seq_start:seq_end]
            pi_batch[i, 0:length] = pi_arrays[i][seq_start:seq_end]
            z_batch[i, 0:length] = z_arrays[i][seq_start:seq_end]
            S = S_arrays[i][undershoot[i]:(seq_start + 1)]
            S_batch.append(S.reshape((-1,) + self.obs_size))

        # Stack batches and send. The format is (B, K, object_shape)
        S_batch = np.stack(S_batch)
        return S_batch, a_batch, r_batch, done_batch, pi_batch, z_batch

    def get_sample(self, batch_idx, K):
        # Get values from storage
        hist_idx = batch_idx // self.seq_size
        seq_idx = self.past_obs - 1 + (batch_idx % self.seq_size)
        # Seq start
        S_array, a_array, r_array, done_array, v_array, pi_array, z_array = self.storage[hist_idx]
        # Construct batch
        a = a_array[seq_idx:(seq_idx + K)]
        r = r_array[seq_idx:(seq_idx + K)]
        done = done_array[seq_idx:(seq_idx + K)]
        pi = pi_array[seq_idx:(seq_idx + K)]
        z = z_array[seq_idx:(seq_idx + K)]
        undershoot = seq_idx + 1 - self.past_obs  # Case where past frames goes back to previous block
        S = S_array[undershoot:(seq_idx+1)]


        """ This cannot currently happen
        if undershoot < 0:
            # Case where previous observations happens across sequence break, so we need to add elements from previous sequence
            # Seq end
            # Get index of next sequence
            game_id = self.game_id[batch_idx]
            seq_list = self.game_to_seq[game_id]
            for i in range(len(seq_list)):
                if seq_list[i]==hist_idx:
                    index = i-1
                    break
            
            if index < 0:
                # Case where the previuos samples cross game start
                # Pad observation array
                S_array2 = np.repeat(S[0][None], -undershoot, axis=0)
            else:
                # Case where previous sample exists
                next_seq = seq_list[index]
                S_array2, a_array2, r_array2, done_array2, v_array2, pi_array2, z_array2 = self.storage[next_seq // self.hist_size]
                S_array2 = S_array2[undershoot:]
            S = np.concatenate([S_array2, S], axis=0)  # Concat elements to produce array of size (,)
        """

        element = [S, a, r, done, pi, z]
        return element


class bootstrap_returner:
    # Class for handling the delay in values, when n-step bootstrap return is used
    def __init__(self, n, gamma, K):
        self.n = n  # Number of steps to bootstrap from
        self.gamma = gamma
        self.K = K
        self.r_list = deque()
        self.gamma_mask = gamma ** np.arange(n)
        self.step = 0  # Number of steps run so far

    def iterate(self, r, v):
        self.r_list.append(r)
        z = sum(self.gamma_mask[0:len(self.r_list)]*self.r_list) + v*self.gamma**len(self.r_list)  # r_list might not be n-long if terminated before n-steps
        if len(self.r_list) != 0:
            self.r_list.popleft()
        return np.float32(z)

    def add_r(self, r):
        self.r_list.append(r)

    def update(self, r, v, done):
        # Function takes new r and v value, and returns n-steps bootstrap value z from n-steps ago
        if not done:
            z = self.iterate(r, v)
            return [z]
        else:
            n_return = self.n  # Number of values to return
            # Case where environment is done and all values needs to be calculated
            if len(self.r_list) < (self.n-1):  # -1 to account for value of r has not been added yet
                # Case where early termination of environment happened, and list was not filled up
                self.add_r(r)
                n_return = len(self.r_list)  # A value for each case in r_list needs to be added
                for i in range(self.n-len(self.r_list)-1):  # -1 to account for value of r has not been added yet
                    self.add_r(0)

                z_list = []  # First value adds r in case of terminal reward
                for i in range(0, n_return):
                    z_list.append(self.iterate(0, 0))
            else:
                z_list = [self.iterate(r, v)]  # First value adds r in case of terminal reward
                for i in range(1, n_return):
                    z_list.append(self.iterate(0, 0))
            return z_list


class experience_replay_sender:
    # Class for sending observations to the experience replay server to store, used in each running agent
    def __init__(self, ex_Q, agent_id, gamma, experience_settings):
        self.ex_Q = ex_Q
        self.agent_id = agent_id
        self.n = experience_settings["n_bootstrap"]
        self.seq_len = experience_settings["sequence_length"]
        self.past_obs = experience_settings["past_obs"]
        self.K = experience_settings["K"]
        self.needed_delay = self.n + self.K
        self.gamma = gamma
        self.bootstrap_returner = bootstrap_returner(self.n, self.gamma, self.K)
        # A list for storing lists of inputs to send. Usually S_new, S, r, done
        self.S_storage = []
        self.a_storage = []
        self.r_storage = []
        self.done_storage = []
        self.pi_storage = []
        self.v_storage = []
        self.z_storage = []
        # Store all arrays for readability in other functions
        self.list_storage = [self.S_storage,
                             self.a_storage,
                             self.r_storage,
                             self.done_storage,
                             self.pi_storage,
                             self.v_storage,
                             self.z_storage]
        self.seq_delay_cache = deque()  # The sending of sequences needs to be delayed so n-step return can be calculated
        self.step = 0

    def reset(self):
        self.bootstrap_returner = bootstrap_returner(self.n, self.gamma, self.K)
        self.S_storage = []
        self.a_storage = []
        self.r_storage = []
        self.done_storage = []
        self.pi_storage = []
        self.v_storage = []
        self.z_storage = []
        # Store all arrays for readability in other functions
        self.list_storage = [self.S_storage,
                             self.a_storage,
                             self.r_storage,
                             self.done_storage,
                             self.v_storage,
                             self.pi_storage,
                             self.z_storage]
        self.seq_delay_cache = deque()
        self.step = 0

    def send(self, done):
        if (len(self.list_storage[0]) == (self.seq_len+self.past_obs+self.K - (self.past_obs>0) - (self.K>0))) or done:
            # Convert all sequences (lists) to array
            message = []
            new_list = []
            n_get = done * (self.seq_len - len(self.list_storage[0]))  # This to not send too many frames when done
            n_get = n_get if n_get<0 else len(self.list_storage[0])
            n_ele = -(self.K+self.past_obs - 1 - (self.past_obs > 1))
            for obs_type in self.list_storage:
                message.append(np.stack(obs_type[:n_get]))
                new_list.append(obs_type[n_ele:])  # Add last K points and past obs from last batch
            message.append(self.agent_id)
            self.ex_Q.put(message, True, 0.01)  # Something is wrong if this times out
            # Reset sequence
            self.list_storage = new_list
            if done:
                if n_get < 0:
                    # Case where not all data could be send in block, so send remaining
                    self.send(done)
                else:
                    self.reset()

    def store(self, S, a, r, done, v, pi):
        # Function for storing experiences
        # Input:
        #    obs_list: A list containing this to store. Example: [S, a, r, done, v, pi]
        self.step += 1
        obs_list = [S, a, r, done, v, pi]
        self.seq_delay_cache.append(obs_list)
        if (self.step >= self.n) or done:
            # Case where bootstrap n-step can be calculated
            zs = self.bootstrap_returner.update(r, v, done)

            if self.step == self.n or (done and (self.step < self.n)):
                # Add values to the beginning
                self.pad_beginning(zs[0])

            # Loop is for the case when env is done and all remaining bootstrap can be calculated
            n_send = 0
            for z in zs:
                # Send remaining values
                n_send += 1
                delayed_sample = self.seq_delay_cache.popleft()
                delayed_sample.append(z)
                # Store list
                for j in range(len(delayed_sample)):
                    self.list_storage[j].append(delayed_sample[j])

                self.send(done and n_send==len(zs))  # This sends the information if there is enough
        elif self.step < self.n:
            # Add r before n-bootstrap can be calculated
            self.bootstrap_returner.add_r(r)

    def pad_beginning(self, z):
        start_sample = self.seq_delay_cache[0]
        for i in range(len(start_sample)):
            for j in range(self.past_obs-1):
                self.list_storage[i].append(start_sample[i])

        # The z value is added seperatly to avoid copying start_sample
        for j in range(self.past_obs-1):
            self.list_storage[-1].append(z)

class frame_stacker:
    def __init__(self, n_stack, boundry_type="copy"):
        self.frames = deque(maxlen=n_stack)
        self.n_stack = n_stack
        self.boundry_type = boundry_type

    def get_stack(self, F):
        if len(self.frames) == 0:
            # Case of initial observation
            if self.boundry_type == "copy":
                self.frames.extend([F]*self.n_stack)
        # Add observation
        self.frames.append(F)
        # Stack frames to numpy array and send back
        S = np.concatenate(self.frames, axis=0)
        return S
