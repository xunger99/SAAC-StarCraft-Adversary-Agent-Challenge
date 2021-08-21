# The code is developed based on run_loop from pysc2. 
#   Xun, June 17, 2021.

"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pdb


def run_loop1(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
      
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)
    
  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        # Disabled by Xun, cosz the code seems to contain a bug.            
        #if max_frames and total_frames >= max_frames:
        #pdb.set_trace()       
        #  return
        #if 360 - total_frames <= 1:   # by xun, try this code, here 360 = 2880/step_mul, the latter is 8
          #pdb.set_trace()
          #total_frames = 0   
          #break
        
        if timesteps[0].last():
          #pdb.set_trace()       
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))