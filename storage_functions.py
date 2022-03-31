import numpy as np
from collections import deque, defaultdict
from multiprocessing import Queue
#from torch.multiprocessing import Queue

class experience_replay_server:
    def __init__(self, experience_settings, MCTS_settings):
        self.hist_size = experience_settings["history_size"]  # The number of sequences of frames to store in memory
        self.seq_size = experience_settings["sequence_length"]  # The number of frames in each sequence
        self.K = experience_settings["K"]  # Number of steps to unroll during training. Needed here to determine delay of sending
        self.ex_Q = Queue()  # For receiving jobs
        self.action_size = MCTS_settings["action_size"]
        self.obs_size = MCTS_settings["observation_size"]
        self.past_obs = experience_settings["past_obs"]  # Number of past observations to stack

        # This will be handled as a circular array,
        #     so the array start can be moved when a new sequence having to reformat the whole array
        self.storage = np.empty((self.hist_size,), dtype="object")  # For priority sampling
        self.game_id = np.empty((self.hist_size * self.seq_size,), dtype="int32")
        self.store_id = np.empty((self.hist_size,), dtype="object")  # For retreiving frames across sequences
        for i in range(self.hist_size):
            self.store_id[i] = deque()  # Initiallize to []
        self.next_id = 0  # ID to use for next game
        self.P = np.zeros((self.hist_size * self.seq_size), dtype="float")  # For priority sampling
        self.total_store = 0
        self.P_replace_idx = 0
        self.agent_to_game = defaultdict(lambda: None)  # Dictionary to get what the next agent should be working on
        self.game_to_seq = defaultdict(lambda: [])  # Dictionary to game_id to sequence list

    def get_Q(self):
        return self.ex_Q

    def recv_store(self):
        S_array, a_array, r_array, done_array, v_array, pi_array, z_array, agent_id = self.ex_Q.get(True, None)
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
            self.game_to_seq.pop(old_game_id)
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
        val_len = v_array.shape[0] + 1 - self.past_obs
        p[0:val_len] = np.abs(v_array - z_array)[(self.past_obs-1):end_index]
        start_idx = self.P_replace_idx * self.seq_size
        end_idx = (self.P_replace_idx + 1) * self.seq_size
        self.P[start_idx:end_idx] = p
        self.game_id[start_idx:end_idx] = game_id

        # Add the new elements
        self.storage[self.P_replace_idx] = [S_array, a_array, r_array, done_array, v_array, pi_array, z_array]
        # Move array start
        self.P_replace_idx = (self.P_replace_idx + 1) % self.hist_size  # sequence to next replace
        self.total_store += 1

    def return_batches(self, batch_size, alpha, K):
        # Normalize priority dist
        if alpha != 1:
            non_zero = self.P != 0
            P_temp2 = self.P[non_zero] ** -alpha
            P_temp2 = P_temp2 / np.sum(P_temp2)
            P_temp = self.P.copy()
            P_temp[non_zero] = P_temp2
        else:
            P_temp = self.P / np.sum(self.P)

        # Select index of batches
        batch_idx = np.random.choice(self.hist_size * self.seq_size, size=batch_size, p=P_temp, replace=False)  # Very slow
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
            pad_length = self.K - len(z)
            S_batch.append(S)
            a = np.pad(a, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
            a_batch.append(a)
            r = np.pad(r, (0, pad_length), mode='constant', constant_values=0)
            r_batch.append(r)
            done = np.pad(done, (0, pad_length), mode='constant', constant_values=1)
            done_batch.append(done)
            pi = np.pad(pi, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
            pi_batch.append(pi)
            z = np.pad(z, (0, pad_length), mode='constant', constant_values=0)
            z_batch.append(z)
        # Stack batches and send. The format is (B, K, object_shape)
        S_batch = np.stack(S_batch)
        a_batch = np.stack(a_batch)
        r_batch = np.stack(r_batch)
        done_batch = np.stack(done_batch)
        pi_batch = np.stack(pi_batch)
        z_batch = np.stack(z_batch)
        return S_batch, a_batch, r_batch, done_batch, pi_batch, z_batch, batch_idx, self.P[batch_idx]

    def get_sample(self, batch_idx, K):
        # Get values from storage
        hist_idx = batch_idx // self.seq_size
        seq_idx = self.past_obs - 1 + batch_idx % self.seq_size


        # Seq start
        S_array1, a_array1, r_array1, done_array1, v_array1, pi_array1, z_array1 = self.storage[hist_idx]
        # Construct batch
        a = a_array1[seq_idx:(seq_idx + K)]
        r = r_array1[seq_idx:(seq_idx + K)]
        done = done_array1[seq_idx:(seq_idx + K)]
        pi = pi_array1[seq_idx:(seq_idx + K)]
        z = z_array1[seq_idx:(seq_idx + K)]

        undershoot = seq_idx + 1 - self.past_obs  # Case where past frames goes back to previous block
        S = S_array1[undershoot:(seq_idx+1)]
        test = 2

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
    def __init__(self, n, gamma):
        self.n = n  # Number of steps to bootstrap from
        self.gamma = gamma
        self.r_list = deque()
        self.gamma_mask = gamma ** np.arange(n)
        self.step = 0  # Number of steps run so far

    def iterate(self, r, v):
        self.r_list.append(r)
        z = sum(self.gamma_mask[0:len(self.r_list)] * self.r_list) + v*self.gamma**len(self.r_list)  # r_list might not be n-long if terminated before n-steps
        if len(self.r_list) != 0:
            self.r_list.popleft()
        return z

    def add_r(self, r):
        self.r_list.append(r)

    def update(self, r, v, done):
        # Function takes new r and v value, and returns n-steps bootstrap value z from n-steps ago
        if not done:
            z = self.iterate(r, v)
            return [z]
        else:
            # Case where environment is done and all values needs to be calculated
            z_list = [self.iterate(r, 0)]  # First value adds r in case of terminal reward
            for i in range(1, self.n):
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
        self.bootstrap_returner = bootstrap_returner(self.n, self.gamma)
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
        self.bootstrap_returner = bootstrap_returner(self.n, self.gamma)
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
        self.seq_delay_cache = deque()
        self.step = 0

    def send(self, done):
        if (len(self.list_storage[0]) == (self.seq_len+self.past_obs+self.K-2)) or done:
            # Convert all sequences (lists) to array
            message = []
            new_list = []

            n_ele = -(self.K+self.past_obs - 1 - (self.past_obs>1))
            for obs_type in self.list_storage:
                message.append(np.stack(obs_type))
                new_list.append(obs_type[n_ele:])  # Add last K points and past obs from last batch
            message.append(self.agent_id)
            self.ex_Q.put(message, True, 0.01)  # Something is wrong if this times out
            # Reset sequence
            self.list_storage = new_list
            if done:
                self.reset()

    def store(self, S, a, r, done, v, pi):
        # Function for storing experiences
        # Input:
        #    obs_list: A list containing this to store. Example: [S, a, r, done, v, pi]
        self.step += 1
        obs_list = [S, a, r, done, v, pi]
        self.seq_delay_cache.append(obs_list)

        if self.step < self.n:
            # Add r before n-bootstrap can be calculated
            self.bootstrap_returner.add_r(r)
        if self.step >= self.n or done:
            # Case where bootstrap n-step can be calculated
            zs = self.bootstrap_returner.update(r, v, done)

            if self.step == self.n or (done and self.step<self.n):
                # Add values to the beginning
                self.pad_beginning(zs[0])

            # Loop is for the case when env is done and all remaining bootstrap can be calculated
            n_send = 0
            for z in zs[0:len(self.seq_delay_cache)]:
                # Send remaining values
                n_send += 1
                delayed_sample = self.seq_delay_cache.popleft()
                delayed_sample.append(z)
                # Store list
                for j in range(len(delayed_sample)):
                    self.list_storage[j].append(delayed_sample[j])
                terminal_obs = (n_send == self.n) and done  # The delay needs to be included in the terminal state
                self.send(terminal_obs)  # This sends the information if there is enough

    def pad_beginning(self, z):
        start_sample = self.seq_delay_cache[0]
        for i in range(len(start_sample)):
            for j in range(self.past_obs-1):
                self.list_storage[i].append(start_sample[i])

        # The z value is added seperatly to avoid copying start_sample
        for j in range(self.past_obs-1):
            self.list_storage[-1].append(z)
