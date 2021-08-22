# SAAC-StarCraft-Adversary-Agent-Challenge

Introduction
A reinforcement learning environment with adversary agents is proposed in this work for pursuit--evasion game in the presence of fog of war, which is of both scientific significance and practical importance in aerospace applications. One of the most popular learning environments, StarCraft, is adopted here and the associated mini-games are analyzed to identify the current limitation for training adversary agents. The key contribution includes the analysis of the potential performance of an agent by incorporating control and differential game theory into the specific reinforcement learning environment, and the development of an adversary-agents challenge (SAAC) environment by extending the current StarCraft mini-games. The subsequent study showcases the use of this learning environment and the effectiveness of an adversary agent for evasion units. Overall, the proposed SAAC environment should benefit pursuit--evasion studies with rapidly-emerging reinforcement learning technologies. Last but not least, the corresponding tutorial code can be found at GitHub. 



Code issues: 
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


