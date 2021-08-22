#---------------------------- 
# The environment for 1 vs 1 with 2 agents, one is scripted and the other is from DQN. 
#
# This code is only for research purpose.   
# The code is developed based on the sc2_env from pysc2.
# Modications: include FindAndDefeatZergling_1 to _3 classes, and _3 is the script used to visit the whole map.   
#     
#  Xun Huang, Jul 29, 2021
#---------------------------- 

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is a working agent code that can achieve max = 40. xun, June 29, 2021. 
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

import pdb
     

# V1, actually adopt a random agent, for testing purpose only. Xun, Jun 18, 2021
class FindAndDefeatZergling_1(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(FindAndDefeatZergling_1, self).step(obs)
    #pdb.set_trace()
    function_id = numpy.random.choice(obs.observation.available_actions)
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
    
    
    
    
    
# V2, scripted attacking. Xun, Jun 18, 2021
# Only goes to (0,0) when no enemy. 
class FindAndDefeatZergling_2(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(FindAndDefeatZergling_2, self).step(obs)
    #pdb.set_trace()
    coords=(0, 0)
    
    
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
      pdb.set_trace()
      
      player_relative = obs.observation.feature_screen.player_relative
      zergling = _xy_locs(player_relative == _PLAYER_ENEMY)
      # no visible enemy, explore by attacking (0, 0)    
      if not zergling:
        return FUNCTIONS.Attack_screen("now", (0,0)) #FUNCTIONS.no_op()
      # Visible enemy, find the zergling with max y coord.
      target = zergling[numpy.argmax(numpy.array(zergling)[:, 1])]
      #coords= (target[1], target[0])   # (x, y)
      #target=coords      
      return FUNCTIONS.Attack_screen("now", target)

    else:
        if FUNCTIONS.select_army.id in obs.observation.available_actions:
            return FUNCTIONS.select_army("select")
        else:
            pass
            
    return FUNCTIONS.no_op()        


# V3, scripted attacking with improved performance. Xun, Jun 18, 2021
# Here the marines do a large Z route. 
class FindAndDefeatZergling_3(base_agent.BaseAgent):
  """An agent is developed from CollectMineralShardsFeatureUnits class. 
  """
  id=0    
  def setup(self, obs_spec, action_spec):
    super(FindAndDefeatZergling_3, self).setup(obs_spec, action_spec)
    #pdb.set_trace()
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

    #self.coords=[(26,25),(32,0),(0,32),(32,32)]  
    #self.coords=[(5,10),(26,10),(5,25),(26,25)]      #self.coords=[(4.9,9.9),(15,9.9),(5,17),(15,17),(5,25),(15,25),(15,10),(26,10),(15,17),(26,17),(15,25),(26,25)] 
    # Obtain the following coordinates by testing, xun
    #self.coords=[(4.9,10.5),(26,14.2),(4.9,18.5),(26.1,19.5),(4.9,22.5),(25,25), (26.1,9.9)] 
    self.coords=[(5,10),(26,10),(26,19),(5,19),(5,25),(26,25), (5,10)] 
    #self.coords=[(14,16),(50,16),(50,28),(14,28),(14,40),(50,40), (14,16)] 
    self.xmean_old = 0  
    self.ymean_old = 0     
    
  def reset(self):
    super(FindAndDefeatZergling_3, self).reset()
    self._marine_selected = False
    self._previous_mineral_xy = [-1, -1]
    self.id = 0   # trace the settled points
    self.xmean_old = 0  
    self.ymean_old = 0   

  def step(self, obs):
    super(FindAndDefeatZergling_3, self).step(obs)
    
    
    #obs.observation['raw_units'] 
    real_action  = FUNCTIONS.no_op()
    real_action1 = FUNCTIONS.no_op()
    
    player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero() 
    xmean=player_x.mean()
    ymean=player_y.mean()
    #pdb.set_trace()
    #real_action1 = FUNCTIONS.move_camera((xmean, ymean))
    
    #pdb.set_trace()
    real_action1 = FUNCTIONS.move_camera(self.coords[self.id])
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:  
        real_action = FUNCTIONS.Attack_screen("now", self.coords[self.id])
    else:
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            real_action = FUNCTIONS.Move_screen("now", self.coords[self.id])
                  
        
    # Find the distance to the taeget position    
    distances = numpy.linalg.norm(numpy.array([xmean,ymean]) - numpy.array(self.coords[self.id]))    
    #if self.id == 1: 
    #    print('distances: ', distances, ', xmean:', xmean, ', ymean:', ymean, ', coords:', self.coords[self.id])
        #pdb.set_trace()

    if abs(distances) < 3: # 2 is a prescribed value, less than which we can say the target position is arrived
        self.id += 1
        self.id = self.id%7    # here I only define 4 corners. xun
        #pdb.set_trace()
    
    
    real_action = [real_action, real_action1]
    return real_action 







def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  #pdb.set_trace()
  return list(zip(x, y))
    

class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not beacon:
        return FUNCTIONS.no_op()
      beacon_center = numpy.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", beacon_center)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShardsFeatureUnits(base_agent.BaseAgent):
  """An agent for solving the CollectMineralShards map with feature units.

  Controls the two marines independently:
  - select marine
  - move to nearest mineral shard that wasn't the previous target
  - swap marine and repeat
  """

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsFeatureUnits, self).setup(obs_spec, action_spec)
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(CollectMineralShardsFeatureUnits, self).reset()
    self._marine_selected = False
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CollectMineralShardsFeatureUnits, self).step(obs)
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    if not marines:
      return FUNCTIONS.no_op()
    marine_unit = next((m for m in marines
                        if m.is_selected == self._marine_selected), marines[0])
    marine_xy = [marine_unit.x, marine_unit.y]

    if not marine_unit.is_selected:
      # Nothing selected or the wrong marine is selected.
      self._marine_selected = True
      return FUNCTIONS.select_point("select", marine_xy)

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      # Find and move to the nearest mineral.
      minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == _PLAYER_NEUTRAL]

      if self._previous_mineral_xy in minerals:
        # Don't go for the same mineral shard as other marine.
        minerals.remove(self._previous_mineral_xy)

      if minerals:
        # Find the closest.
        distances = numpy.linalg.norm(
            numpy.array(minerals) - numpy.array(marine_xy), axis=1)
        closest_mineral_xy = minerals[numpy.argmin(distances)]

        # Swap to the other marine.
        self._marine_selected = False
        self._previous_mineral_xy = closest_mineral_xy
        return FUNCTIONS.Move_screen("now", closest_mineral_xy)

    return FUNCTIONS.no_op()


class CollectMineralShardsRaw(base_agent.BaseAgent):
  """An agent for solving CollectMineralShards with raw units and actions.

  Controls the two marines independently:
  - move to nearest mineral shard that wasn't the previous target
  - swap marine and repeat
  """

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsRaw, self).setup(obs_spec, action_spec)
    if "raw_units" not in obs_spec:
      raise Exception("This agent requires the raw_units observation.")

  def reset(self):
    super(CollectMineralShardsRaw, self).reset()
    self._last_marine = None
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CollectMineralShardsRaw, self).step(obs)
    marines = [unit for unit in obs.observation.raw_units
               if unit.alliance == _PLAYER_SELF]
    if not marines:
      return RAW_FUNCTIONS.no_op()
    marine_unit = next((m for m in marines if m.tag != self._last_marine))
    marine_xy = [marine_unit.x, marine_unit.y]

    minerals = [[unit.x, unit.y] for unit in obs.observation.raw_units
                if unit.alliance == _PLAYER_NEUTRAL]

    if self._previous_mineral_xy in minerals:
      # Don't go for the same mineral shard as other marine.
      minerals.remove(self._previous_mineral_xy)

    if minerals:
      # Find the closest.
      distances = numpy.linalg.norm(
          numpy.array(minerals) - numpy.array(marine_xy), axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]

      self._last_marine = marine_unit.tag
      self._previous_mineral_xy = closest_mineral_xy
      return RAW_FUNCTIONS.Move_pt("now", marine_unit.tag, closest_mineral_xy)

    return RAW_FUNCTIONS.no_op()


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
      if not roaches:
        return FUNCTIONS.no_op()

      # Find the roach with max y coord.
      target = roaches[numpy.argmax(numpy.array(roaches)[:, 1])]
      return FUNCTIONS.Attack_screen("now", target)

    if FUNCTIONS.select_army.id in obs.observation.available_actions:
      return FUNCTIONS.select_army("select")

    return FUNCTIONS.no_op()


class DefeatRoachesRaw(base_agent.BaseAgent):
  """An agent specifically for solving DefeatRoaches using raw actions."""

  def setup(self, obs_spec, action_spec):
    super(DefeatRoachesRaw, self).setup(obs_spec, action_spec)
    if "raw_units" not in obs_spec:
      raise Exception("This agent requires the raw_units observation.")

  def step(self, obs):
    super(DefeatRoachesRaw, self).step(obs)
    marines = [unit.tag for unit in obs.observation.raw_units
               if unit.alliance == _PLAYER_SELF]
    roaches = [unit for unit in obs.observation.raw_units
               if unit.alliance == _PLAYER_ENEMY]

    if marines and roaches:
      # Find the roach with max y coord.
      target = sorted(roaches, key=lambda r: r.y)[0].tag
      return RAW_FUNCTIONS.Attack_unit("now", marines, target)

    return FUNCTIONS.no_op()
