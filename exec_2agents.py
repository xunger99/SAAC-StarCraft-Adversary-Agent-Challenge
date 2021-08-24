# Simlified DQN for pysc2 mini-game, by Xun, Oct 2020.
# 1 vs 1. Both are controlled by agents. 
#

import importlib

import numpy
import traceback
import os
import json
import random
from absl import app
from absl import flags
import pdb

# own classes
from env import  SC2_Env_xun2 #Sc2Env2Outputs  #Sc2Env1Output,  SC2_Env_xun2
from sc2Processor import Sc2Processor
from sc2Policy import Sc2Policy #, Sc2PolicyD
from sc2DqnAgent import Sc2DqnAgent_v5 

from prioReplayBuffer import PrioritizedReplayBuffer, ReplayBuffer  #remove by xun

# framework classes
from pysc2.env import sc2_env
# By xun, to use Plasdml and AMD GPU. 
#import plaidml
#       plaidml.keras.install_backend()
#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Conv2DTranspose
from keras.layers.merge import concatenate, add

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# for the debug of all process costs.
from pysc2.lib import stopwatch


# To display the relations between functions, xun 2021. 
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput

# MiniGame 1: MoveToBeacon
# MiniGame 2: CollectMineralShards
_ENV_NAME = "FindAndDefeatDronesAA" #"PursuitEvasion1" #"FindAndDefeatZerglings"
_SCREEN = 32
_MINIMAP = 16

_VISUALIZE = False# Simlified this code by Xun, Oct 2020.
_TEST = False

_profile=True


FLAGS = flags.FLAGS
flags.DEFINE_string("agent", "scripted_agent1.FindAndDefeatZergling_4",  
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent2", "pysc2.agents.random_agent.RandomAgent", 
                    "Which agent to run, as a python path to an Agent class.")


def __main__(unused_argv):
    agent_name = "Xun_test"
    run_number = 1
    
    
#    graphviz = GraphvizOutput()
#    graphviz.output_file = 'basic.png'
    
    
    results_dir = "weights/{}/{}/{}".format(_ENV_NAME, agent_name, run_number)
   
    agent_classes = []
    
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)
    agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)
    agents = [agent_cls() for agent_cls in agent_classes]
   
    #pdb.set_trace()
    
    
#    with PyCallGraph(output=graphviz):
    fully_conf_v_10(results_dir, agents)
    

 

# Prepare the network
def fully_conf_v_10(a_dir, agents):        
    try:
        seed = random.randint(1, 324234)
    
        env = SC2_Env_xun2(screen=_SCREEN, visualize=_VISUALIZE, env_name=_ENV_NAME, 
            training=not _TEST, agents = agents)        
        
        env.seed(seed)
        numpy.random.seed(seed)

        nb_actions = 3

        prio_replay = True  #False #True # modified by xun to avoid using new lib
        multi_step_size = 3

        # HyperParameter
        action_repetition = 1
        gamma = .99
        memory_size = 200000
        learning_rate = .0001
        warm_up_steps = 4000
        train_interval = 4

        bad_prio_replay = False #True # modified by xun to avoid using new lib
        prio_replay_alpha = 0.6
        prio_replay_beta = (0.5, 1.0, 200000)   # (beta_start, beta_end, 

        eps_start = 1     # modified by xun from 1 to the current value
        eps_end = 0
        eps_steps = 4000

        #Prepare the directory    
        directory = a_dir
        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_filename = directory + '/dqn_weights.h5f'
        checkpoint_weights_filename = directory + '/dqn_weights_{step}.h5f'
        log_filename = directory + '/dqn_log.json'
        log_filename_gpu = directory + '/dqn_log_gpu.json'
        log_interval = 8000 
        
        #Prepare the network                  
        kernel_size=7   
        n_filters=16
        main_input = Input(shape=(3, env.screen, env.screen), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)
        
        # Normal deep network
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        branch = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        coord_out = Conv2D(1, (1, 1), padding='same', activation='linear')(branch)
        act_out = Flatten()(branch)
        #act_out = Dense(256, activation='relu')(act_out)
        act_out = Dense(256, activation='relu')(act_out)
        act_out = Dense(nb_actions, activation='linear')(act_out)     
                
          
        full_conv_sc2 = Model(main_input, [act_out, coord_out])
        
        memory = PrioritizedReplayBuffer(memory_size, prio_replay_alpha)
        policy = LinearAnnealedPolicy(Sc2Policy(env=env,nb_actions=nb_actions), attr='eps', value_max=eps_start, value_min=eps_end,
                                      value_test=eps_end, nb_steps=eps_steps)
                                                                            
        test_policy = Sc2Policy(env=env, eps=eps_end)
        processor = Sc2Processor(screen=env._SCREEN)        
        #pdb.set_trace()        
        dqn = Sc2DqnAgent_v5(model=full_conv_sc2, nb_actions=nb_actions, screen_size=env._SCREEN,
                             memory=memory, processor=processor,gamma=gamma,
                             nb_steps_warmup=warm_up_steps,multi_step_size=multi_step_size,
                             policy=policy, test_policy=test_policy, target_model_update=10000,
                             train_interval=train_interval, delta_clip=1.)                         
        dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

#        if _profile:
#            stopwatch.sw.enable()   
        dqn.fit(env, nb_steps=300000, nb_max_start_steps=0, 
            action_repetition=action_repetition)          
 
        dqn.save_weights(weights_filename, overwrite=True)

    except KeyboardInterrupt:
        exit(0)
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass



def conv2d_block(input_tensor, n_filters, kernel_size=7):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    x = Activation("relu")(x)          

    x = Conv2D(filters=n_filters*2, kernel_size=(kernel_size-2, kernel_size-2), kernel_initializer="he_normal",
               padding="same")(x)
    x = Activation("relu")(x)         
    return x



if __name__ == '__main__':
    app.run(__main__)



