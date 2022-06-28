

MuZero_settings = {"N_training_games": 200000,  # Total number of games to run pr. training loop
                    "temp_switch": 16,  # Number of turns before other temperature measure is used
                    "eta_par": 0.03,  # Distributional value for action selection
                    "epsilon": 0.25,  # Distributional value for action selection
                   "save_image": False,  # Save image of environment when MCT is saved
                   "low_support": -300,  # Lowest value of supported values for reward and value head
                   "high_support": 300,  # Highest value of supported values for reward and value head
                   "n_support": 601,  # Number of support values. To include a value for 0 keep the number of heads odd
                   }

# Settings for experience replay and storing of values in general
experience_settings = {"history_size": 10**4,  # The number of sequences of frames to store in memory
                        "sequence_length": 200,  # The number of frames in each sequence
                        "epsilon": 0.25,  # Distributional value for action selection
                        "n_bootstrap": 10,  # Number of steps forward to bootstrap from
                        "past_obs": 4,  # Number of past observations to stack. Original Atari was 32
                        "K": 10  # Number of steps to unroll during training. Needed here to determine delay of sending
                   }

# These are the settings for the Monte Carlo Tree Search (MCTS),
MCTS_settings = {"n_parallel_explorations": 1,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                 "action_size": (4,),  # size of action space
                 "observation_size": (32, 32),  # size of observation space
                 "observation_channels": 1*4,  # number of channels of observation space (i.e. 3*4 for RGB and 4x frame stack)
                 "hidden_S_size": (4, 4),  # Size of the hidden state
                 "hidden_S_channel": 4,  # Size of the hidden state
                 "virtual_loss": 3,  # Magnitude of loss during parallel explorations
                 "number_of_threads": 16,  # Number of games / threads to run on CPU
                 "N_MCTS_sim": 50,  # Number of MCTS simulations for each action
                 "c1": 1.25,  # parameter for pUCT selection
                 "c2": 19652,
                 "gamma": 0.997,
                 "min_val": None,
                 "max_val": None,
                }  # parameter reinforcement learning

training_settings = {"train_batch_size": 2,  # Batch size on GPU during training
                     "num_epochs": 100,  # Maximum length of training epoch before break
                     "lr_init": 10**-3,  # Original Atari rate was 0.05
                     "lr_decay_rate": 0.1,
                     "lr_decay_steps": 400e3,  # Original Atari was 350e3
                     "alpha": 1,
                     "beta": 1,
                     "momentum": 0.9,  # Original was 0.9
                     "weight_decay": 1e-4,  # Original was 1-e4
                     "uniform_sampling": False,  # If prioritized sampling should be disabled, Original False
                     "scale_values": True,  # Scale values with transforms. Original True
                     "use_different_gpu": True,  # Use different gpu's for training and self-play if possible
                     }