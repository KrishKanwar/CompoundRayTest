import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import pickle

data = np.genfromtxt("LPTC_rf_1.csv", delimiter=",")

direction = "left"

# Save output
with open("Pkls/xy_return_" + direction + ".pkl", "rb") as handle:
    xy_return = np.array(pickle.load(handle))

with open("Pkls/q_coord_diff_x_no_mult_" + direction + ".pkl", "rb") as handle:
    q_coord_diff_x_no_mult = np.array(pickle.load(handle))
with open("Pkls/q_coord_diff_y_no_mult_" + direction + ".pkl", "rb") as handle:
    q_coord_diff_y_no_mult = np.array(pickle.load(handle))

with open("Pkls/q_coord_diff_x_mult_" + direction + ".pkl", "rb") as handle:
    q_coord_diff_x_mult = np.array(pickle.load(handle))
with open("Pkls/q_coord_diff_y_mult_" + direction + ".pkl", "rb") as handle:
    q_coord_diff_y_mult = np.array(pickle.load(handle))

print(xy_return.shape)
print(q_coord_diff_x_no_mult.shape)
print(q_coord_diff_y_no_mult.shape)
print(q_coord_diff_x_mult.shape)
print(q_coord_diff_y_mult.shape)
