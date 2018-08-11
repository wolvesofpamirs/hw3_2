import pickle
import sys,os
import numpy as np
from common import load


PATH = os.path.dirname(os.path.realpath(__file__))

folder = os.path.join(PATH,'DATA')

LX,LY, XU = load(folder)

#load_label(folder)

print("Hello World!")


