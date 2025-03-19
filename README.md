Setup:

nvidia smi - checks for cuda compatibility

install anaconda3 for windows - add path to system variables

install pycharm

new project - custom - virtualenv

install cuda toolkit if running cuda, if no cuda compatible devices: import os os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forces PyTorch to use CPU

installing imports: install torch install pandas
