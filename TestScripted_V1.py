#
# Xun's fist script code, updated from pysc2/tests code. June 16, 2021. 
#
# (1) Specifically for the map entitled FindAndDefeatZergling (single player case). 
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import random_agent
#from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.tests import utils
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features, units

from absl import flags
from absl.testing import absltest
import sys

import pdb
import time

# Imported by Xun
import run_loop1
import random_agent1
import scripted_agent1



_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


FLAGS = flags.FLAGS
FLAGS(sys.argv)

class TestScripted(utils.TestCase):
  steps = 2880
  step_mul = 8
  episodes = 10


  def test_defeat_zerglings(self):
    agent_format = sc2_env.AgentInterfaceFormat(
      feature_dimensions=sc2_env.Dimensions(
        screen=(32,32),
        minimap=(32,32),
      ),
      use_raw_units=True, 
      use_feature_units=True 
    )
    #pdb.set_trace()  #"FindAndDefeatZerglings", #"Empty_xun1" ,#"FindAndDefeatZerglings2",
    #,sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_hard)
    #sc2_env.Bot(sc2_env.Race.zerg)], #
    with sc2_env.SC2Env(
        map_name="FindAndDefeatZerglings", 
        players=[sc2_env.Agent(sc2_env.Race.terran)], 
        step_mul=self.step_mul,
        disable_fog=True, #False, #True,
        visualize=False, #True,
        agent_interface_format=[agent_format],
        game_steps_per_episode=2880)as env:       # self.steps * self.step_mul) as env:
      
      
      obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
      player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

      # Break Point!!
      #pdb.set_trace()
      print(player_relative)

      #agent = random_agent1.RandomAgent1()     Enable random agent
      
      # Instead, enable scripted agent
      # agent=scripted_agent1.FindAndDefeatZergling()

      agent=scripted_agent1.FindAndDefeatZergling_4()
      #agent =random_agent.RandomAgent
      agent2=random_agent.RandomAgent
            
      #pdb.set_trace()
      run_loop1.run_loop1([agent], env, self.steps, self.episodes)  #agent,agent2]
      
      #pdb.set_trace()
      #self.tearDown()
#    self.assertEqual(agent.steps, self.steps)




    
        

if __name__ == "__main__":
  absltest.main()
  
  
  

  