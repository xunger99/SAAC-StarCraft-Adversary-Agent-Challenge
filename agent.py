#---------------------------- 
# The DQN agent, simplified and modified for 1 vs 1 case.  
#
# This code is modified from keras-rl and dqn-pysc2. 
# Most, if not all, modifications were explicitly pointed out in the code.    
#     
#  Xun Huang, Jul 29, 2021
#---------------------------- 

import warnings
from copy import deepcopy
import numpy as np
import pdb

from keras.callbacks import History
from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)


class Agent3(object):
    """Modified Version Keras-rl core/Agent
    """
    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.
        """
        self.training = True

        callbacks = [] if not callbacks else callbacks[:]
        
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)

        params = {
            'nb_steps': nb_steps,
        }

        
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                #pdb.set_trace()
                if observation is None:  # start of a new episode
                    # print('if observation is None ...')  # debuging code by xunger
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)
                    
                    # Obtain the initial observation by resetting the environment.
                    #self.reset_states()
                    #pdb.set_trace()
                    try:
                        observation = deepcopy(env.reset())
                    except protocol.ConnectionError:
                    #    pdb.set_trace()   
                        env.close()
                        observation = deepcopy(env.start())   

                        
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

    
                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None
      
                #pdb.set_trace()    
                # Run a single step.
                #callbacks.on_step_begin(episode_step)
                # ********************************************************************************
                # !!!!!
                # !!!!!
                # !!!!!                            
                # This is where all of the work happens. We first perceive and compute the action
                # (first step) and then use the reward to improve (backward step).
                # !!!!!
                # !!!!!
                # !!!!!            
                # ********************************************************************************  
#                if self.step%5000==0:
#                    pdb.set_trace()                          
                action = self.forward(observation)        
                
                     
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
#                    print('agent step 2')        
#                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)                    
                    observation = deepcopy(observation)
                    
                    #pdb.set_trace()  #rewards_history.append(reward)
                    reward += r
                    if done:
                        break
                        
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True

                metrics = self.backward(reward, terminal=done, observation_1=observation)
                episode_reward += reward

                episode_step += 1
                self.step += 1
                
                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more q-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    # BUT: I disagree, and don't wanna damage my backward call, cause memory is different
                    # anyways now....
                    
                    # Note: this part is different from the keras-rl.core.  Xun
                    
                    
                    self.forward(observation)
                    
                    self.backward(0., terminal=True, observation_1=observation)
                    #pdb.set_trace()
                    
                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    
                    for _ in range(self.recent.maxlen):
                        self.recent.append(None)
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely abortedself.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history




    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    # observation_1 was included here, different from the keras-rl.core code!!! xun
    def backward(self, reward, terminal, observation_1):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.

        # Returns
            A list of the model's layers
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).

        # Returns
            A list of metric's names (string)
        """
        return []


    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass



