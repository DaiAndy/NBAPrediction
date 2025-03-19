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
