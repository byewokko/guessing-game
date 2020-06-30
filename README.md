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
  - however the results are not as good they were supposed to be with Reinforce, according the original paper (Lazaridou &al, 2017)

## TODO:

- check [this pytorch implementation](https://github.com/thirdratecyberpunk/lazaridou-game)
- cluster concepts/symbols using tSNE
- reproducibility: set seeds
    - [explained here](https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
- dataset splitting
    - separate training ans validation at least
- exploding weights -> NaN
    - probably caused by extreme values in the softmax layer
