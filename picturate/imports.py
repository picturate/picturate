# # future imports
# from __future__ import division
# from __future__ import print_function

# Utility libraries
import os
import os.path as osp
import errno
import yaml
from easydict import EasyDict as edict


# Machine learning library imports
import numpy as np

# Image library imports
import skimage.transform
from PIL import Image, ImageDraw, ImageFont

# Torch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert import BertModel