import argparse
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import psutil  # 需要pip install psutil
import torch
from model import *
# 训练前环境初始化（防止OMP冲突）
import os
from transformers import PreTrainedTokenizerFast