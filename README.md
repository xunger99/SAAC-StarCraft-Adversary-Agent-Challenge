# SAAC-StarCraft-Adversary-Agent-Challenge

It is well known in the StarCraft programming community that the current PySC2 interface could produce websocket errors during the low-level message passing between multiple agent interfaces. To bypass this issue, a thorough programming debug has been conducted in this work to identify the corresponding code. Then, a temporary fix has been adopted to rectify the issue before any official fix is available from DeepMind in the near future. 

Details are: 
1. The episode steps are 2880, but will end incorrectly when multi-interface is chosen. Hence, set 2880 to 2872 in the code to bypass the issue. 
2. In agent.py, include the following fix, 
                    try:
                        observation = deepcopy(env.reset())
                    except protocol.ConnectionError:
                    #    pdb.set_trace()   
                        env.close()
                        observation = deepcopy(env.start())   
3. In env.py, include the following fix: 
      try:
            observation = self.env.step(actions)
        except protocol.ConnectionError:
            #pdb.set_trace()
            self.close()
            #self.start()
            observation = self.start()


