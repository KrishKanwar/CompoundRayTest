# %%
# plotting setup
# matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
from cmath import pi
# import pandas as pd

import pickle

with open('MotionDetector/final_result_3D_left.pkl', 'rb') as handle:
    final_result_3D = np.array(pickle.load(handle))

adjacent_ommatid_locations = np.genfromtxt('MotionDetector/ind_nb.csv', delimiter=',')

left_adjacent_ommatid_locations = adjacent_ommatid_locations[1:787, :]
right_adjacent_ommatid_locations = adjacent_ommatid_locations[787:1552, :]

ommatidia350 = final_result_3D[349:350, :, :]
ommatidia351 = final_result_3D[350:351, :, :]


