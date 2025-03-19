# NBA Lineup Prediction

This is Group #8 assignment on designing a model that is able to predict the best player for a lineup if a player is missing.

Our Program uses Word2Vec for the embedding of players, teams.

Pytorch for the training of the model that is used for the preidcition. 

Pickle to transport the embeddings from the training file to the testing file. 

Can be used with a CPU or a CUDA compatible device.

Setup: 

1. Install pycharm or any IDE
3. new project - custom - virtualenv - use the virtual environtment setup in this github - Python 3.11
4. Install cuda toolkit if running cuda, if no cuda compatible devices:
      import os
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces PyTorch to use CPU

5. installing imports: 
      import numpy as np
      import pandas as pd
      import torch
      import torch.nn as nn
      from gensim.models import Word2Vec
      import pickle


Files:

matchups-2007.csv ----> matchups-2015.csv: Data files used for the training of the model
NBA_test.csv: The test file with the missing player in each row/game
NBA_test_labels.csv: contains the removed_values which represent the true values for the missing players for NBA_test.csv
training2-2.py: The main training file used to train a model to predict the missing player
testing2.py: The main testing file used to test the trained model against test data and to see how it performs
nba_lineup_model.pth: The trained model file
playerEmbeddings.pkl: player/team embeddings, used to transfer the embeddings from the training file to the testing files

