# guessing-game
Multi-agent guessing game playground. Based on [Multi-Agent Cooperation and the 
Emergence of (Natural) Language (Lazaridou et al., 2017)](https://arxiv.org/abs/1612.07182). 
Part of my master thesis project.


## NOTES:

- my implementation of the Reinforce algorithm (according to the paper) doesn't work at all
  - [reinforce_agent.py](agent/reinforce_agent.py)
  - loss function issue??
  - gibbs sampler is too random
- using **Q-learning** instead of Reinforce does work smooth
  - [q_agent.py](agent/q_agent.py)
  - however the results are not as good as in the original paper (Lazaridou &al, 2017) with Reinforce
