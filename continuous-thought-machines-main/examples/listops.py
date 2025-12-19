import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import false
from tqdm import tqdm
import numpy as np
import random
from scipy.special import softmax
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import seaborn as sns
import imageio
import mediapy
from kan import KAN

import torch._dynamo as dynamo
dynamo.config.suppress_errors = False



