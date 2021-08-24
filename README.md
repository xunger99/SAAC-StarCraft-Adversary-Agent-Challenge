# SAAC-StarCraft-Adversary-Agent-Challenge

# Paper 
A draft that explains the detailed design of the code is arXiv:submit/3894432 (https://arxiv.org/submit/3894432/view), "Adversary-Agent Reinforcement Learning for Pursuit–Evasion" by Prof. Xun Huang, Aug 2021. If you have used this code in your research work, please cite the paper. 



# Introduction
A reinforcement learning environment with adversary agents is proposed in this work for pursuit--evasion game in the presence of fog of war, which is of both scientific significance and practical importance in aerospace applications. One of the most popular learning environments, StarCraft, is adopted here and the associated mini-games are analyzed to identify the current limitation for training adversary agents. The key contribution includes the analysis of the potential performance of an agent by incorporating control and differential game theory into the specific reinforcement learning environment, and the development of an adversary-agents challenge (SAAC) environment by extending the current StarCraft mini-games. The subsequent study showcases the use of this learning environment and the effectiveness of an adversary agent for evasion units. Overall, the proposed SAAC environment should benefit pursuit--evasion studies with rapidly-emerging reinforcement learning technologies. Last but not least, the corresponding tutorial code can be found at GitHub. 



# Installation steps: 
1. conda, (python 2.8, 2.7, 2.6 all work)
2. pip tensorflow-gpu (2.5.0, without GPU is OK too)
3. keras (2.4.3)
4. PySC2 (3.0.0)
5. baselines (0.1.6) 
6. battle.net + download maps 
7. Download files from this folder
* The above steps have been tested on Mac OSX/Windows 10/Ubuntu platforms. If you met installation problems, please google solutions. 

## Tips
1. Currrntly, I provide three tests and two mini-game maps. 
Map 1: FindAndDefeatDrones.SC2Map, single-player game. Please copy this file to the mini_games folder of the StarCtaft installation folder on your computer.  
Map 2: FindAndDefeatDronesAA.SC2Map, double-player game. Please copy this file to the melee folder of the StarCtaft installation folder on your computer. 
2. Then, open the files mini_games.py and melee.py on your pysc2/maps folder, include the names of these two games therein. Then, the SC2 env will be adble to load the new games. 
3. You can edit these two games by map editor, and follow the above steps to run your own games. 

# Running cases:
1. python TestScripted_V1.py
2. python TestScripted_V2.py
3. python exec_2agents.py

# Code issues: 
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


