#---------------------------- 
# The environment for 1 vs 1 with 2 agents, one is scripted and the other is from DQN. 
#
# This code is only for research purpose.   
# This code is modified from keras-rl and dqn-pysc2. The class action_to_sc2 requires further modifications
# to improve the DQN performance. 
#     
#  Xun Huang, Jul 29, 2021
#---------------------------- 

from rl.core import Env
#from pysc2.env import sc2_env
import sc2_env_xun
from pysc2.lib import features
from pysc2.lib import actions
import numpy as np
import pdb

from pysc2.lib import protocol      # Xun, for protocol.ConnectionError 

FUNCTIONS = actions.FUNCTIONS


class SC2_Env_xun2(Env):
    last_obs = None     # observation, used for deep learning
    last_obs1= None     # observation, used for deep learning
    env = None          # used by close subroutine
    agents=[]
    _SCREEN = None
    _MINIMAP = None
    _ENV_NAME = None
    _TRAINING = None

    def __init__(self, screen=16, visualize=False, env_name="MoveToBeacon", training=False, agents=[]):
        print("init SC2")

        self._SCREEN = screen
        self._MINIMAP = screen
        self._VISUALIZE = visualize
        self._ENV_NAME = env_name
        self._TRAINING = training
        self.env = sc2_env_xun.SC2Env_xun(
            map_name=self._ENV_NAME,
            players=[sc2_env_xun.Agent(sc2_env_xun.Race.protoss),sc2_env_xun.Agent(sc2_env_xun.Race.zerg)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self._SCREEN,
                    minimap=self._MINIMAP
                ),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=2880,
            visualize=self._VISUALIZE
        )
        
        #pdb.set_trace()    
        self.agents=agents


    # Actions assigned here.  Xun, 2021    
    def action_to_sc2(self, act):
        #pdb.set_trace()
        real_action = FUNCTIONS.no_op()
        real_action1= FUNCTIONS.no_op()

        if act.action == 1:    #Move camera & move or attack
            # Must move camera first, then attack. Otherwise, the attacking coordinates would be incorrect. 
            if FUNCTIONS.move_camera.id in self.last_obs.observation.available_actions:     # By Xun
                real_action1 = FUNCTIONS.move_camera((act.coords[1], act.coords[0]))

#            if FUNCTIONS.Attack_screen.id in self.last_obs.observation.available_actions:   # By Xun
#                real_action = FUNCTIONS.Attack_screen("now", (act.coords[1], act.coords[0]))    
            if FUNCTIONS.Move_screen.id in self.last_obs.observation.available_actions:   # By Xun
                real_action = FUNCTIONS.Move_screen("now", (act.coords[1], act.coords[0]))    
#            else:
#                print('cannot attack')
            real_action=[real_action1,real_action]        

 
        elif act.action == 2:         #Select army or worker  
#            if FUNCTIONS.select_army.id in self.last_obs.observation.available_actions:
#                real_action =  FUNCTIONS.select_army("select")
            #pdb.set_trace()
            if FUNCTIONS.select_idle_worker.id in self.last_obs.observation.available_actions:
                real_action =  FUNCTIONS.select_idle_worker("select_all")

            player_y, player_x = (self.last_obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero() 
            
            xmean=ymean=0
            if(len(player_y)==0):
                pass
#                pdb.set_trace()
            else:
                xmean=player_x.mean()
                ymean=player_y.mean()
            
            if FUNCTIONS.move_camera.id in self.last_obs.observation.available_actions:
                real_action1 = FUNCTIONS.move_camera((xmean, ymean))
            
            real_action=[real_action,real_action1]   
            
                   
        elif act.action == 0:         # 
            pass # do nothing to continue the former action
               #real_action = FUNCTIONS.select_point("toggle", (act.coords[1], act.coords[0]))                   
        else:
#            pass
            assert False
        return real_action

    
    
    # User should edit this subroutine to obtain the requred observation and rewards. xun, Aug 2021    
    def step(self, action):                
        
        observation_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        for agent, obs_spec, act_spec in zip(self.agents, observation_spec, action_spec):
            agent.setup(obs_spec, act_spec)

        real_action = self.action_to_sc2(action)  # Action from the DQN
        obs=self.last_obs1
                
        #actions = [agent.step(obs0) for agent, obs0 in zip(self.agents,obs)]  # enable this line will restore two two agents: 1 scripted vs 1 random.
        agent=self.agents[0]
        obs0=obs[0]
        action0 = agent.step(obs0)              # Action from the agent[0] for the voidray 
        actions = [action0, real_action]
        
        #pdb.set_trace()
        
        # fix from websocket timeout issue... Xun, Aug 2021
        try:
            observation = self.env.step(actions)
        except protocol.ConnectionError:
            #pdb.set_trace()
            self.close()
            #self.start()
            observation = self.start() 
            
        
        # Observation[0:1], 0 for agent 0, 1 for agent 1, the latter is for evasion part here.  xun   
        self.last_obs = observation[1]                  
        self.last_obs1 = observation                    
        # small_observation = observation[0].observation.feature_screen.unit_density
        act_obs=np.zeros([32,32])
        #pdb.set_trace()
        act_obs[(action.coords[1],action.coords[0])]=action.action
        small_observation = [observation[0].observation.feature_screen.player_relative,
                             observation[0].observation.feature_screen.selected,
                             act_obs]
		#observation[0].observation.feature_screen.visibility_map]                                           
        #pdb.set_trace()
        # Modified by Xun                                 
        #return small_observation, observation[0].reward, observation[0].last(), {}
        reward=observation[1][3].score_by_category[0][2]    # the unit number of the evasion part, xun 
        return small_observation, reward, observation[0].last(), {}            
    
    # fix from websocket timeout issue... Xun, Aug 2021
    def start(self):
        self.env = sc2_env_xun.SC2Env_xun(
            map_name=self._ENV_NAME,
            players=[sc2_env_xun.Agent(sc2_env_xun.Race.protoss),sc2_env_xun.Agent(sc2_env_xun.Race.zerg)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self._SCREEN,
                    minimap=self._MINIMAP
                ),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=2880,
            visualize=self._VISUALIZE
        )
        return self.reset()
            

    def reset(self):
        #pdb.set_trace()
        observation = self.env.reset()

        if self._TRAINING and np.random.random_integers(0, 1) == 4:
            ys, xs = np.where(observation[0].observation.feature_screen.player_relative == 1)
            observation = self.env.step(actions=(FUNCTIONS.select_point("toggle", (xs[0], ys[0])),))

 #       observation = self.env.step(actions=(FUNCTIONS.select_army(0),))   
        # Select all for both sides, xun 
        #pdb.set_trace()
        observation = self.env.step(actions=(FUNCTIONS.select_army(0),))   
        #observation = self.env.step(actions=(FUNCTIONS.select_army(0),FUNCTIONS.select_army(0)))    # modified by xun

        self.last_obs = observation[1]
        self.last_obs1= observation
#        small_observation = [observation[0].observation.feature_screen.player_relative,
#                             observation[0].observation.feature_screen.selected]
        small_observation = [observation[0].observation.feature_screen.player_relative,
                             observation[0].observation.feature_screen.selected,
                             observation[0].observation.feature_screen.visibility_map]
                             

        return small_observation

    def render(self, mode: str = 'human', close: bool = False):
        pass


    def close(self):
        if self.env:
            self.env.close()

    def seed(self, seed=None):
        if seed:
            self.env._random_seed = seed

    def set_env_name(self, name: str):
        self._ENV_NAME = name

    def set_screen(self, screen: int):
        self._SCREEN = screen

    def set_visualize(self, visualize: bool):
        self._VISUALIZE = visualize

    def set_minimap(self, minimap: int):
        self._MINIMAP = minimap

    @property
    def screen(self):
        return self._SCREEN


"""
    def configure(self, *args, **kwargs):

        switcher = {
            '_ENV_NAME': self.set_env_name,
            '_SCREEN': self.set_screen,
            '_MINIMAP': self.set_minimap,
            '_VISUALIZE': self.set_visualize,
        }

        if kwargs is not None:
            for key, value in kwargs:
                func = switcher.get(key, lambda: print)
                func(value)
"""
        
        

