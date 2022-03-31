import numpy as np
from multiprocessing import Queue
from storage_functions import experience_replay_sender, experience_replay_server
import time

MCTS_settings = {"n_parallel_explorations": 4,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                 "action_size": (4,),  # size of action space
                 "observation_size": (3,3),  # shape of observation space
                 "gamma" : 0.9}  # parameter for pUCT selection

# Settings for experience replay and storing of values in general
experience_settings = {"history_size": 50,  # The number of sequences of frames to store in memory
                    "sequence_length": 10,  # The number of frames in each sequence
                    "n_bootstrap": 4,  # Number of steps forward to bootstrap from
                    "past_obs": 6,  # Number of past observations to stack. Original Atari was 32
                    "K": 5  # Number of steps to unroll during training. Needed here to determine delay of sending
                   }
n_bootstrap = experience_settings["n_bootstrap"]
gamma = MCTS_settings["gamma"]
past_obs = experience_settings["past_obs"]

EX_server = experience_replay_server(experience_settings, MCTS_settings)
get_ex_Q = EX_server.get_Q()
EX_sender1 = experience_replay_sender(get_ex_Q, 1, gamma, experience_settings)
EX_sender2 = experience_replay_sender(get_ex_Q, 2, gamma, experience_settings)
np.random.seed(19)
N_rounds = 1000
S_stack1 = np.random.rand(N_rounds, 3,3)
S_stack1[:,0,0] = np.arange(N_rounds)
a_stack1 = np.random.rand(N_rounds,4)
r_stack1 = np.arange(N_rounds)
v_stack1 = 2*np.arange(N_rounds)
done_stack1 = np.zeros((N_rounds))
pi_stack1 = np.random.rand(N_rounds,4)
def calc_n_bootstrap(index, n_bootstrap):
    gammas = sum(gamma ** np.arange(n_bootstrap) * r_stack1[index:(n_bootstrap+index)])
    z = sum(gamma ** np.arange(n_bootstrap) * r_stack1[index:(n_bootstrap+index)]) + v_stack1[index+n_bootstrap - 1] * gamma ** (
        n_bootstrap)
    return z
z_stack1 = []
for i in range(N_rounds-4):
    z_stack1.append(calc_n_bootstrap(i, n_bootstrap))
z_stack1 = np.stack(z_stack1)
# Convert to unrolled array to account for past observations
S_unrolled = np.concatenate([np.repeat(S_stack1[0][None], past_obs-1, axis=0), S_stack1])
a_unrolled = np.concatenate([np.repeat(a_stack1[0][None], past_obs-1, axis=0), a_stack1])
r_unrolled = np.concatenate([np.repeat(r_stack1[0][None], past_obs-1, axis=0), r_stack1])
v_unrolled = np.concatenate([np.repeat(v_stack1[0][None], past_obs-1, axis=0), v_stack1])
done_unrolled = np.concatenate([np.repeat(done_stack1[0][None], past_obs-1, axis=0), done_stack1])
pi_unrolled = np.concatenate([np.repeat(pi_stack1[0][None], past_obs-1, axis=0), pi_stack1])
z_unrolled = np.concatenate([np.repeat(z_stack1[0][None], past_obs-1, axis=0), z_stack1])

# First check for errors upon sending more data than can be in buffer
for i in range(N_rounds):
    EX_sender1.store(S_stack1[i], a_stack1[i], r_stack1[i], done_stack1[i], v_stack1[i], pi_stack1[i])
EX_server.recv_store()
gamma = MCTS_settings["gamma"]


# Do checks
seq_len = experience_settings["sequence_length"]
K = experience_settings["K"]
hist_len = experience_settings["history_size"]
# Check for correct behavior upon stack being full
for i_stacks in range(N_rounds // (hist_len*seq_len)):
    for i in range(hist_len-1): # Exclude last observation
        EX_server.recv_store()
        # Check P-values are updated
        start_index = i_stacks*hist_len*seq_len+i*seq_len
        end_index = i_stacks*hist_len*seq_len+(i+1)*seq_len + K + past_obs - 2
        assert(np.all(EX_server.P[start_index:(end_index-K+1)]!=0))
        # Check all values
        assert(np.all(EX_server.storage[i][0] == S_unrolled[start_index:end_index]))
        assert(np.all(EX_server.storage[i][1] == a_unrolled[start_index:end_index]))
        assert(np.all(EX_server.storage[i][2] == r_unrolled[start_index:end_index]))
        assert(np.all(EX_server.storage[i][3] == done_unrolled[start_index:end_index]))
        assert(np.all(EX_server.storage[i][4] == v_unrolled[start_index:end_index]))
        assert(np.all(EX_server.storage[i][5] == pi_unrolled[start_index:end_index]))
        assert (np.all(EX_server.storage[i][6] == z_unrolled[start_index:end_index]))
        # Check if sampling works
        EX_server.return_batches(i+1, 1, K)

# Check for correct behavior for batches done
EX_server = experience_replay_server(experience_settings, MCTS_settings)
get_ex_Q = EX_server.get_Q()
EX_sender = experience_replay_sender(get_ex_Q, 1, gamma, experience_settings)
np.random.seed(1)
cut_len = seq_len-3
S_stack1 = np.random.rand(cut_len, 3,3)
S_stack1[:,0,0] = np.arange(cut_len)
a_stack1 = np.random.rand(cut_len,4)
r_stack1 = np.arange(cut_len)
v_stack1 = 2*np.arange(cut_len)
done_stack1 = np.zeros((cut_len))
done_stack1[-1] = 1
pi_stack1 = np.random.rand(cut_len,4)
for i in range(cut_len):
    EX_sender.store(S_stack1[i], a_stack1[i], r_stack1[i], done_stack1[i], v_stack1[i], pi_stack1[i])
EX_server.recv_store()
result = EX_server.return_batches(4, 1, K)

cut_len = seq_len-2
S_stack1 = np.random.rand(cut_len, 3,3)
S_stack1[:,0,0] = np.arange(cut_len)
a_stack1 = np.random.rand(cut_len,4)
r_stack1 = np.arange(cut_len)
v_stack1 = 2*np.arange(cut_len)
done_stack1 = np.zeros((cut_len))
done_stack1[-1] = 1
pi_stack1 = np.random.rand(cut_len,4)
for i in range(cut_len):
    EX_sender.store(S_stack1[i], a_stack1[i], r_stack1[i], done_stack1[i], v_stack1[i], pi_stack1[i])
EX_server.recv_store()
for i in range(N_rounds):
    # Just sample some games
    batch_size = np.random.randint(1,15)
    result = EX_server.return_batches(batch_size, 1, K)
    alpha = np.random.rand()
    result = EX_server.return_batches(batch_size, alpha, K)

N_episodes = 100
max_episode_len = 1000
total_samples = 0

EX_server = experience_replay_server(experience_settings, MCTS_settings)
get_ex_Q = EX_server.get_Q()
EX_sender = experience_replay_sender(get_ex_Q, 1, gamma, experience_settings)
EX_sender2 = experience_replay_sender(get_ex_Q, 2, gamma, experience_settings)

def check_batch(result, batch_size):
    # Check if S and r agree
    assert(result[0].shape[0]==batch_size)
    assert (result[1].shape[0] == batch_size)
    assert (result[2].shape[0] == batch_size)
    assert (result[3].shape[0] == batch_size)
    assert (result[4].shape[0] == batch_size)
    assert (result[5].shape[0] == batch_size)
    assert(np.any(result[0][:,-1, 0, 0] == result[2][:,0]))

time_start = time.time()
for i in range(N_episodes):
    # Sample epiosde length
    episode_len = np.random.randint(1,max_episode_len)

    S_stack = np.random.rand(episode_len, 3, 3)
    S_stack[:, 0, 0] = np.arange(episode_len)
    a_stack = np.random.rand(episode_len, 4)
    r_stack = np.arange(episode_len)
    v_stack = 2 * np.arange(episode_len)
    done_stack = np.zeros((episode_len))
    done_stack[episode_len-1] = 1
    pi_stack = np.random.rand(episode_len, 4)

    for j in range(episode_len):
        EX_sender.store(S_stack[j], a_stack[j], r_stack[j], done_stack[j], v_stack[j], pi_stack[j])
        batch_size = np.random.randint(1, 15)
        if total_samples >= batch_size:
            result = EX_server.return_batches(batch_size, 1, K)
            check_batch(result, batch_size)
        EX_sender2.store(S_stack[j], a_stack[j], r_stack[j], done_stack[j], v_stack[j], pi_stack[j])
        if not get_ex_Q.empty():
            EX_server.recv_store()


    total_samples += episode_len
time_end = time.time()
duration = time_end - time_start
print("Game time:" + str(round(N_episodes/duration)))



# Now randomize episode lengths and run many tests
print("All tests passed!")
#EX_server.ex_Q.close()