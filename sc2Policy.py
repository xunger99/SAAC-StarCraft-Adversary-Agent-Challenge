# map the dqn output to actions understandable by pysc2

from rl.policy import Policy
import numpy as np
from sc2DqnAgent import Sc2Action
import pdb

class Sc2Policy(Policy):

    def __init__(self, env, nb_actions=3, eps=0.1, testing=False):
        super(Sc2Policy, self).__init__()
        self.eps = eps
        self.nb_pixels = env._SCREEN
        #pdb.set_trace()
        self.nb_actions = nb_actions
        self.testing = testing

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (numpy array of shape (2, ?)):
            one List of q-estimates for action-selection
            one array of shape (screensize, screensize) for position selection

        # Returns
            Selection action 
        """
        
        action = Sc2Action()

        # Epsilon-Greedy
        # pdb.set_trace()
        egran=np.random.uniform()
        if egran < self.eps and not self.testing:
            action.action = np.random.random_integers(0, self.nb_actions-1)
            action.coords = (np.random.random_integers(0, self.nb_pixels-1),  np.random.random_integers(0, self.nb_pixels-1))
            if self.eps <0.05:
                print('eps:',self.eps)

        else:
            # greedy.
            action.action = np.argmax(q_values[0])      
            #pdb.set_trace()
            action.coords = np.unravel_index(q_values[1].argmax(), q_values[1].shape)[1:3]
            
            # action.coords = np.unravel_index(np.reshape(q_values[1][0][:][:], (16, 16)).argmax(), np.reshape(

        assert len(action.coords) == 2

        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(Sc2Policy, self).get_config()
        config['eps'] = self.eps
        config['testing'] = self.testing
        return config


