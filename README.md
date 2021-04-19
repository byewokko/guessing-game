# guessing-game
Multi-agent guessing game playground. Based on [Multi-Agent Cooperation and the 
Emergence of (Natural) Language (Lazaridou et al., 2017)](https://arxiv.org/abs/1612.07182). 
Part of my master thesis project.

## Training:

Run the experiment using the following command:

```python run.py [settings.yaml]```

Experiment settings and model hyperparameters can be specified via the 
settings file only.

It is possible to queue up several experiments using a csv file:

```python run_many.py [queue.csv] [results-output.csv]```

Final training stats of each model are written to the specified results file.

## Testing

To run a test, change the `mode` from `train` to `test` in the settings file. 
You might also want to change the number of episodes and the image dataset 
to be used.

## Result analysis

Some basic pivot tables and tSNE clustering is available in 
`Model output analysis.ipynb` notebook.

## TODOs

- [x] refactor train procedure
- [x] reimplement reinforce agent
- [ ] reimplement Q-agent
  - [ ] fix training/predicting input
- [ ] reimplement Compound Agent
- analysis
  - [ ] TSNE / the other thing
  - [ ] symbol purity
    