import numpy as np
from cmath import pi

from lowpass_method import low_pass_filter

import pickle

with open('MotionDetector/extraction_test.pkl', 'rb') as handle:
    ommatid_data = pickle.load(handle)

print(ommatid_data)
print(ommatid_data.shape)

left_ommatid_values = ommatid_data[:, 0:787]
right_ommatid_values = ommatid_data[:, 787:1552]

total_ommatid_data = np.genfromtxt('MotionDetector/lens_opticAxis_acceptance.csv', delimiter=',')

left_ommatid_data = total_ommatid_data[1:788, :]
right_ommatid_data = total_ommatid_data[788:1553, :]

print(left_ommatid_data)
print(right_ommatid_data)

adjacent_ommatid_locations = np.genfromtxt('MotionDetector/ind_nb.csv', delimiter=',')

left_adjacent_ommatid_locations = adjacent_ommatid_locations[1:788, :]
right_adjacent_ommatid_locations = adjacent_ommatid_locations[788:1553, :]

print(left_adjacent_ommatid_locations)
print(right_adjacent_ommatid_locations)

left_ommatid_values_lpf = []
right_ommatid_values_lpf = []

#create copy of ommatid_values with low pass filter
for i in range(left_ommatid_values.shape[1]):
    t = np.linspace(0, 1.5, 30)
    u = left_ommatid_values[:, i]
    yf = low_pass_filter(t, u)
    left_ommatid_values_lpf.append(yf)

for i in range(right_ommatid_values.shape[1]):
    t = np.linspace(0, 1.5, 30)
    u = right_ommatid_values[:, i]
    yf = low_pass_filter(t, u)
    right_ommatid_values_lpf.append(yf)

# t = np.linspace(0, 1.5, 30)
# u = left_ommatid_values[:, 0]

# yf = low_pass_filter(t, u)
# print(yf)

# print(yf.shape)

# print(np.array(left_ommatid_values).shape)

left_ommatid_values = np.array(left_ommatid_values)
left_ommatid_values_lpf = np.array(left_ommatid_values_lpf)
right_ommatid_values = np.array(right_ommatid_values)
right_ommatid_values_lpf = np.array(right_ommatid_values_lpf)
left_adjacent_ommatid_locations = np.array(left_adjacent_ommatid_locations)
right_adjacent_ommatid_locations = np.array(right_adjacent_ommatid_locations)

print(left_ommatid_values.shape)
print(left_ommatid_values_lpf.shape)

print(right_ommatid_values.shape)
print(right_ommatid_values_lpf.shape)

print(left_adjacent_ommatid_locations.shape)
print(right_adjacent_ommatid_locations.shape)

final_result_3D_left = []

for i in range(np.array(left_ommatid_values_lpf).shape[0]):
    single_ommatidia_30_frames = []

    for j in range(np.array(left_ommatid_values_lpf).shape[1]):
        single_ommatidia_single_frame = []

        for k in range(np.array(left_adjacent_ommatid_locations).shape[1]-1):
            current_ommatid_position = left_adjacent_ommatid_locations[i, 0] - 1
            adjacent_ommatid_position = left_adjacent_ommatid_locations[i, k+1] - 1

            if(np.isnan(current_ommatid_position) or np.isnan(adjacent_ommatid_position)):
                comparison_value = np.nan
            else:

                lp0 = left_ommatid_values_lpf[int(current_ommatid_position), j]
                n1 = left_ommatid_values[j, int(adjacent_ommatid_position)]

                lp1 = left_ommatid_values_lpf[int(adjacent_ommatid_position), j]
                n0 = left_ommatid_values[j, int(current_ommatid_position)]

                comparison_value = (lp0*n1) - (lp1*n0)

            single_ommatidia_single_frame.append(comparison_value)
        
        single_ommatidia_30_frames.append(single_ommatidia_single_frame)

    single_ommatidia_30_frames_lpf = []

    for i in range (6):
        t = np.linspace(0, 1.5, 30)
        u = np.array(single_ommatidia_30_frames)[:,i]
        vertical_slice_lpf = low_pass_filter(t, u)
        single_ommatidia_30_frames_lpf.append(vertical_slice_lpf)

    final_result_3D_left.append(single_ommatidia_30_frames_lpf)

print(np.array(final_result_3D_left).shape)
print(final_result_3D_left)

with open('MotionDetector/final_result_3D_left.pkl', 'wb') as handle:
    pickle.dump(final_result_3D_left, handle, protocol=pickle.HIGHEST_PROTOCOL)