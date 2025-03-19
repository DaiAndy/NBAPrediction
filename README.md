# NBA Lineup Prediction

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
