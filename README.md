# guessing-game
Multi-agent guessing game playground. Re-implementation of [Multi-Agent Cooperation and the 
Emergence of (Natural) Language (Lazaridou et al., 2017)](https://arxiv.org/abs/1612.07182). 
Part of my master thesis project.


## TODO:

- check [this pytorch implementation](https://github.com/thirdratecyberpunk/lazaridou-game)
- cluster concepts/symbols using tSNE
- reproducibility: set seeds
    - [explained here](https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
- dataset splitting
    - separate training ans validation at least
- 2020/04/25 ... made the agents learn with **Q-learning** instead of Reinforce
- the agents are not learning at all
    - take time to understand [this explanation of policy 
gradient](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
    - revisit the loss according to the paper
    - contact the authors of the paper for clarifications
        - single or separate embedding layers?
        - which optimizer? custom weight updates?
        - use biases?
        - UPDATE: according to Alex P., hyperparameters don't matter
    - reuse this agent to play cartpole and debug
- exploding weights -> NaN
    - some hyperparameter setting lead to exploding weights in the bottom (embedding) layers
- implement batches ... DONE
    - the paper uses batch_size = 32
