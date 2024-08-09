# Produces Reichardt Function values used in motion_detector.py, Tau value for low pass filter hardcoded
import numpy as np
from cmath import pi
from lowpass_method import low_pass_filter
import pickle
import configparser

config = configparser.ConfigParser()

# MetaTxt to retrieve csv files and scene txt (later replace with a function)
config.read("MetaTxt.txt")
readPath = config.get("data", "path")
csvData = config.get("data", "csvData")
csvNeighborsl = config.get("data", "csvNeighborsLeft")
csvNeighborsr = config.get("data", "csvNeighborsRight")

# Read in scene txt
config.read(readPath)
videoName = config.get("variables", "videoName")  # name of scene folder

# Load number of ommatidia in each eye (split)
data = np.genfromtxt(csvData, delimiter=",")
num_omm_l = data[np.flatnonzero(data[:, 7] == 0), :].shape[0]
num_omm_r = data[np.flatnonzero(data[:, 7] == 1), :].shape[0]

# Load in and split data extraction output
with open("OutputData/" + videoName + "/i_de.pkl", "rb") as handle:
    omm_data = pickle.load(handle)
l_omm_vals = omm_data[:, num_omm_r:]
r_omm_vals = omm_data[:, 0:num_omm_r]

# Ommatidia structure data
omm_data = np.genfromtxt(csvData, delimiter=",")
l_omm_data = omm_data[num_omm_r + 1 :, :]
r_omm_data = omm_data[1 : num_omm_r + 1, :]

# Indicies of neighbors
l_adj_omm_loc = np.genfromtxt(csvNeighborsl, delimiter=",")
l_adj_omm_loc = l_adj_omm_loc[1:, :]
r_adj_omm_loc = np.genfromtxt(csvNeighborsr, delimiter=",")
r_adj_omm_loc = r_adj_omm_loc[1:, :]

# Create copy of ommatid_vals with low pass filter
l_omm_vals_lpf = []
r_omm_vals_lpf = []
t = np.linspace(0, 1.5, 300)  # time points in [s]

for i in range(l_omm_vals.shape[1]):
    u = l_omm_vals[:, i]
    yf = low_pass_filter(t, u, 0.003)
    l_omm_vals_lpf.append(yf)

for i in range(r_omm_vals.shape[1]):
    u = r_omm_vals[:, i]
    yf = low_pass_filter(t, u, 0.003)
    r_omm_vals_lpf.append(yf)

# Make sure all set as np.array
l_omm_vals = np.array(l_omm_vals)
l_omm_vals_lpf = np.array(l_omm_vals_lpf)
l_omm_vals_lpf = np.transpose(l_omm_vals_lpf)  # take the transpose
r_omm_vals = np.array(r_omm_vals)
r_omm_vals_lpf = np.array(r_omm_vals_lpf)
r_omm_vals_lpf = np.transpose(r_omm_vals_lpf)  # take the transpose
l_adj_omm_loc = np.array(l_adj_omm_loc)
r_adj_omm_loc = np.array(r_adj_omm_loc)

# Print shapes
# all "omm_vals" have shape [ommatidia, directions, frames]
print(l_omm_vals.shape)
print(l_omm_vals_lpf.shape)
print(r_omm_vals.shape)
print(r_omm_vals_lpf.shape)
print(l_adj_omm_loc.shape)
print(r_adj_omm_loc.shape)


# From the low-pass filter vals and ommatidia neighbors,
# compute the directional motion signal in 6 hex directions
def hr_func(ovl, aol, ov):
    f_result_3D_4dir = []
    omm_vals_lpf = ovl
    adj_omm_loc = aol
    omm_vals = ov
    for i in range(np.array(omm_vals_lpf).shape[1]):  # iterate over all ommatidia
        real_i = np.where(adj_omm_loc[:, 0] == i + 1)[0].tolist()[
            0
        ]  # change to in order of ommatidia in csv

        single_omm_all_frames = []
        omm_pos = (
            adj_omm_loc[real_i, :] - 1
        )  # row numbers are 1-off from python indexing

        for j in range(np.array(omm_vals_lpf).shape[0]):  # iterate over all frames
            single_omm_single_frame = []

            # Iterate over all 4 averaged neighbors
            for k in range(4):
                # Make sure all 6 neighbored exist
                nan_flag = False
                for z in range(7):
                    co = omm_pos[z]
                    if np.isnan(co):
                        nan_flag = True
                    elif np.isnan(omm_vals[j, int(co)]) or np.isnan(
                        omm_vals_lpf[j, int(co)]
                    ):
                        nan_flag = True

                if nan_flag:
                    single_omm_single_frame.append(np.nan)  # append nan if no neighbor

                # Averaging for 4 neighbors, hardcoded depending on direction
                else:
                    if k == 0: # Horizontal direction ({2,4}-{0,2,5}, right - middle)
                        lp0 = (
                            omm_vals_lpf[j, int(omm_pos[0])]
                            + omm_vals_lpf[j, int(omm_pos[2])]
                            + omm_vals_lpf[j, int(omm_pos[5])]
                        ) / 3
                        n1 = (
                            omm_vals[j, int(omm_pos[3])] + omm_vals[j, int(omm_pos[4])]
                        ) / 2
                        lp1 = (
                            omm_vals_lpf[j, int(omm_pos[3])]
                            + omm_vals_lpf[j, int(omm_pos[4])]
                        ) / 2
                        n0 = (
                            omm_vals[j, int(omm_pos[2])]
                            + omm_vals[j, int(omm_pos[0])]
                            + omm_vals[j, int(omm_pos[5])]
                        ) / 3
                    elif k == 1:
                        lp0 = (
                            omm_vals_lpf[j, int(omm_pos[0])]
                            + omm_vals_lpf[j, int(omm_pos[2])]
                            + omm_vals_lpf[j, int(omm_pos[5])]
                        ) / 3
                        n1 = (
                            omm_vals[j, int(omm_pos[1])] + omm_vals[j, int(omm_pos[6])]
                        ) / 2
                        lp1 = (
                            omm_vals_lpf[j, int(omm_pos[1])]
                            + omm_vals_lpf[j, int(omm_pos[6])]
                        ) / 2
                        n0 = (
                            omm_vals[j, int(omm_pos[2])]
                            + omm_vals[j, int(omm_pos[0])]
                            + omm_vals[j, int(omm_pos[5])]
                        ) / 3
                    elif k == 2: # Vertical direction (2-0, top minus middle)
                        lp0 = omm_vals_lpf[j, int(omm_pos[0])]
                        n1 = omm_vals[j, int(omm_pos[2])]
                        lp1 = omm_vals_lpf[j, int(omm_pos[2])]
                        n0 = omm_vals[j, int(omm_pos[0])]
                    elif k == 3:
                        lp0 = omm_vals_lpf[j, int(omm_pos[0])]
                        n1 = omm_vals[j, int(omm_pos[5])]
                        lp1 = omm_vals_lpf[j, int(omm_pos[5])]
                        n0 = omm_vals[j, int(omm_pos[0])]

                    comparison_val = (lp0 * n1) - (lp1 * n0)  # HR formula

                    single_omm_single_frame.append(comparison_val)

            single_omm_all_frames.append(single_omm_single_frame)

        # Apply low-pass filter to the directional motion
        single_omm_all_frames_lpf_4dir = []
        for m in range(4):
            t = np.linspace(0, 1.5, 300)
            u = np.array(single_omm_all_frames)[:, m]
            vertical_slice_lpf = low_pass_filter(t, u, 0.003)
            single_omm_all_frames_lpf_4dir.append(vertical_slice_lpf)

        f_result_3D_4dir.append(single_omm_all_frames_lpf_4dir)

    return f_result_3D_4dir


# Run function for left eye
f_result_3D_l_4dir = hr_func(l_omm_vals_lpf, l_adj_omm_loc, l_omm_vals)
print(np.array(f_result_3D_l_4dir).shape)

# Run function for right eye
f_result_3D_r_4dir = hr_func(r_omm_vals_lpf, r_adj_omm_loc, r_omm_vals)
print(np.array(f_result_3D_r_4dir).shape)

# Save output
with open("OutputData/" + videoName + "/f_de_l.pkl", "wb") as handle:
    pickle.dump(f_result_3D_l_4dir, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("OutputData/" + videoName + "/f_de_r.pkl", "wb") as handle:
    pickle.dump(f_result_3D_r_4dir, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Sample data plotting
# from matplotlib import pyplot as plt
# f_result_3D_l_4dir = np.array(f_result_3D_l_4dir)
# i = 367
# j = 395
# fig, ax = plt.subplots(1, figsize=(8, 6))
# ax.plot(l_omm_vals[71:77, i], c="r")
# ax.plot(l_omm_vals[71:77, j])

# ax.plot(l_omm_vals_lpf[71:77, i], c="r")
# ax.plot(l_omm_vals_lpf[71:77, j])

# ax.plot(f_result_3D_l_4dir[i, 0, 71:77, c="r")
# ax.plot(f_result_3D_l_4dir[j, 0, 71:77])
