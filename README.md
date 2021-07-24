# guessing-game
Multi-agent guessing game playground. Based on [Multi-Agent Cooperation and the 
Emergence of (Natural) Language (Lazaridou et al., 2017)](https://arxiv.org/abs/1612.07182). 
The experimental part of my master thesis.

## Training

Train the models using the following command:

```sh
python train.py [settings-train.yml]
```

Experiment settings and model hyperparameters can be specified via the 
settings file only.

It is possible to queue up several experiments using a csv file:

```sh
python train.py [settings-train.yml] [batch-settings.csv]
```

Final training stats of each model are written to the specified results file.

## Create a testset

To generate a test set, run:

```sh
python make_test.py [settings-make-test.yml]
```


## Testing

To test your models, run the following command:
```sh
python test.py [settings-test.yml]
```

The models and the test file need to be specified in the settings file.


## Result analysis

Result analyses can be carried out via the jupyter notebooks in the `analysis` folder.



## TODOs

- [x] plot the 6switch models
- [x] pick the best one from each setting, move to sep folder
- [x] test the best on the regular testset
- [x] test the best on the same-synset testset
- [ ] cluster analysis of test output
- [ ] symbol purity of test output
- [ ] cluster analysis of embedding layer
- [ ] qualitative analysis of clusters

    