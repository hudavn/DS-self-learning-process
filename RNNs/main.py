# import requests

# data = requests.get("https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt")
# with open("wonderland.txt", "w") as f:
#     f.write(data.text)

import numpy 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os

filename = './wonderland.txt'
raw_text = open(filename).read().lower()
print(raw_text)
    
