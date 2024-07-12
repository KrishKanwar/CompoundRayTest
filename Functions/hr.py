import numpy as np
from cmath import pi
from lowpass_method import low_pass_filter
import pickle
import configparser

config = configparser.ConfigParser()

config.read("MetaTxt.txt")
readPath = config.get("data", "path")
csvData = config.get("data", "csvData")
csvNeighborsLeft = config.get("data", "csvNeighborsLeft")
csvNeighborsRight = config.get("data", "csvNeighborsRight")
direction = int(config.get("data", "direction"))

config.read(readPath)
videoName = config.get("variables", "videoName")

data = np.genfromtxt(csvData, delimiter=",")
num_omm_right = data[np.flatnonzero(data[:, 7] == 1), :].shape[0]
num_omm_left = data[np.flatnonzero(data[:, 7] == 0), :].shape[0]

with open("OutputData/" + videoName + "/i_de.pkl", "rb") as handle:
    ommatid_data = pickle.load(handle)

# Data extraction data split
right_ommatid_values = ommatid_data[:, 0:num_omm_right]
left_ommatid_values = ommatid_data[:, num_omm_right:]

# Ommatidia strucure data
total_ommatid_data = np.genfromtxt(csvData, delimiter=",")

# Split eyes
right_ommatid_data = total_ommatid_data[1 : num_omm_right + 1, :]
left_ommatid_data = total_ommatid_data[num_omm_right + 1 :, :]

# Indexs of neighbors
adjacent_ommatid_locations_left = np.genfromtxt(csvNeighborsLeft, delimiter=",")
adjacent_ommatid_locations_right = np.genfromtxt(csvNeighborsRight, delimiter=",")

left_adjacent_ommatid_locations = adjacent_ommatid_locations_left[1:, :]
right_adjacent_ommatid_locations = adjacent_ommatid_locations_right[1:, :]

left_ommatid_values_lpf = []
right_ommatid_values_lpf = []

# create copy of ommatid_values with low pass filter
t = np.linspace(0, 1.5, 300)  # time points in [s]
for i in range(left_ommatid_values.shape[1]):
    u = left_ommatid_values[:, i]
    yf = low_pass_filter(t, u, 0.003)
    left_ommatid_values_lpf.append(yf)

for i in range(right_ommatid_values.shape[1]):
    u = right_ommatid_values[:, i]
    yf = low_pass_filter(t, u, 0.003)
    right_ommatid_values_lpf.append(yf)

left_ommatid_values = np.array(left_ommatid_values)
left_ommatid_values_lpf = np.array(left_ommatid_values_lpf)
left_ommatid_values_lpf = np.transpose(left_ommatid_values_lpf)  # take the transpose
right_ommatid_values = np.array(right_ommatid_values)
right_ommatid_values_lpf = np.array(right_ommatid_values_lpf)
right_ommatid_values_lpf = np.transpose(right_ommatid_values_lpf)  # take the transpose
left_adjacent_ommatid_locations = np.array(left_adjacent_ommatid_locations)
right_adjacent_ommatid_locations = np.array(right_adjacent_ommatid_locations)

print(left_ommatid_values.shape)
print(left_ommatid_values_lpf.shape)

print(right_ommatid_values.shape)
print(right_ommatid_values_lpf.shape)

print(left_adjacent_ommatid_locations.shape)
print(right_adjacent_ommatid_locations.shape)

final_result_3D_left_nolpf = []
final_result_3D_left = []


# from the low-pass filter values and ommatidia neighbors,
# compute the directional motion signal in 6 hex directions
def hr_func(ovl, aol, ov):
    final_result_3D_left_4dir = []
    ommatid_values_lpf = ovl
    adjacent_ommatid_locations = aol
    ommatid_values = ov
    for i in range(np.array(ommatid_values_lpf).shape[1]):  # iterate over all ommatidia
        real_i = np.where(adjacent_ommatid_locations[:, 0] == i + 1)[0].tolist()[
            0
        ]  # Change to in order of ommatidia in csv

        single_ommatidia_all_frames = []
        current_ommatid_positions = (
            adjacent_ommatid_locations[real_i, :] - 1
        )  # row numbers are 1-off from python indexing

        for j in range(
            np.array(ommatid_values_lpf).shape[0]
        ):  # iterate over all frames
            single_ommatidia_single_frame = []

            # iterate over all 6 neighbors
            for k in range(4):
                # make sure there is a nb
                nan_flag = False
                for z in range(7):
                    co = current_ommatid_positions[z]
                    if np.isnan(co):
                        nan_flag = True
                    elif np.isnan(ommatid_values[j, int(co)]) or np.isnan(
                        ommatid_values_lpf[j, int(co)]
                    ):
                        nan_flag = True

                if nan_flag:
                    single_ommatidia_single_frame.append(np.nan)
                else:
                    if k == 0:
                        lp0 = (
                            ommatid_values_lpf[j, int(current_ommatid_positions[0])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[2])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[5])]
                        ) / 3
                        n1 = (
                            ommatid_values[j, int(current_ommatid_positions[3])]
                            + ommatid_values[j, int(current_ommatid_positions[4])]
                        ) / 3

                        lp1 = (
                            ommatid_values_lpf[j, int(current_ommatid_positions[3])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[4])]
                        ) / 3
                        n0 = (
                            ommatid_values[j, int(current_ommatid_positions[2])]
                            + ommatid_values[j, int(current_ommatid_positions[0])]
                            + ommatid_values[j, int(current_ommatid_positions[5])]
                        ) / 3
                    elif k == 1:
                        lp0 = (
                            ommatid_values_lpf[j, int(current_ommatid_positions[0])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[2])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[5])]
                        ) / 3
                        n1 = (
                            ommatid_values[j, int(current_ommatid_positions[1])]
                            + ommatid_values[j, int(current_ommatid_positions[6])]
                        ) / 3

                        lp1 = (
                            ommatid_values_lpf[j, int(current_ommatid_positions[1])]
                            + ommatid_values_lpf[j, int(current_ommatid_positions[6])]
                        ) / 3
                        n0 = (
                            ommatid_values[j, int(current_ommatid_positions[2])]
                            + ommatid_values[j, int(current_ommatid_positions[0])]
                            + ommatid_values[j, int(current_ommatid_positions[5])]
                        ) / 3

                    elif k == 2:
                        lp0 = ommatid_values_lpf[j, int(current_ommatid_positions[0])]
                        n1 = ommatid_values[j, int(current_ommatid_positions[2])]

                        lp1 = ommatid_values_lpf[j, int(current_ommatid_positions[2])]
                        n0 = ommatid_values[j, int(current_ommatid_positions[0])]

                    else:
                        lp0 = ommatid_values_lpf[j, int(current_ommatid_positions[0])]
                        n1 = ommatid_values[j, int(current_ommatid_positions[5])]

                        lp1 = ommatid_values_lpf[j, int(current_ommatid_positions[5])]
                        n0 = ommatid_values[j, int(current_ommatid_positions[0])]

                    comparison_value = (lp0 * n1) - (lp1 * n0)

                    single_ommatidia_single_frame.append(comparison_value)

            single_ommatidia_all_frames.append(single_ommatidia_single_frame)

        # final_result_3D_left.append(single_ommatidia_all_frames)
        # apply low-pass filter to the directional motion
        single_ommatidia_all_frames_lpf_4dir = []
        for m in range(4):
            t = np.linspace(0, 1.5, 300)
            # print(np.array(single_ommatidia_all_frames).shape)
            u = np.array(single_ommatidia_all_frames)[:, m]
            vertical_slice_lpf = low_pass_filter(t, u, 0.003)
            single_ommatidia_all_frames_lpf_4dir.append(vertical_slice_lpf)
        final_result_3D_left_4dir.append(single_ommatidia_all_frames_lpf_4dir)
    return final_result_3D_left_4dir


final_result_3D_left_4dir = hr_func(
    left_ommatid_values_lpf, left_adjacent_ommatid_locations, left_ommatid_values
)

final_result_3D_right_4dir = hr_func(
    right_ommatid_values_lpf, right_adjacent_ommatid_locations, right_ommatid_values
)

print("final", np.array(final_result_3D_left_4dir).shape)
print("final", np.array(final_result_3D_right_4dir).shape)

with open("OutputData/" + videoName + "/f_de_l.pkl", "wb") as handle:
    pickle.dump(final_result_3D_left_4dir, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("OutputData/" + videoName + "/f_de_r.pkl", "wb") as handle:
    pickle.dump(final_result_3D_right_4dir, handle, protocol=pickle.HIGHEST_PROTOCOL)

from matplotlib import pyplot as plt

final_result_3D_left_4dir = np.array(final_result_3D_left_4dir)
# final_result_3D_left_nolpf = np.array(final_result_3D_left_nolpf)

i = 367
j = 395
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.plot(left_ommatid_values[71:77, i], c = 'r')
ax.plot(left_ommatid_values[71:77, j])

ax.plot(left_ommatid_values_lpf[71:77, i], c = 'r')
ax.plot(left_ommatid_values_lpf[71:77, j])

ax.plot(final_result_3D_left_4dir[i, 0, 71:77])