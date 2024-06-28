import numpy as np
from cmath import pi

from lowpass_method import low_pass_filter

import pickle

# # set working dir
# import os
# os.chdir('../')

with open('MotionDetector/extraction_test.pkl', 'rb') as handle:
#with open('extraction_test.pkl', 'rb') as handle:
    ommatid_data = pickle.load(handle)

print(ommatid_data)
print(ommatid_data.shape)

# Data extraction data split
left_ommatid_values = ommatid_data[:, 0:786]
right_ommatid_values = ommatid_data[:, 786:1552]

# Ommatidia strucure data
total_ommatid_data = np.genfromtxt('MotionDetector/lens_opticAxis_acceptance.csv', delimiter=',')
#total_ommatid_data = np.genfromtxt('lens_opticAxis_acceptance.csv', delimiter=',')

# Split eyes
left_ommatid_data = total_ommatid_data[1:787, :]
right_ommatid_data = total_ommatid_data[787:1553, :]

print(left_ommatid_data)
print(right_ommatid_data)

# Indexs of neighbors
adjacent_ommatid_locations = np.genfromtxt('MotionDetector/ind_nb.csv', delimiter=',')
#adjacent_ommatid_locations = np.genfromtxt('ind_nb.csv', delimiter=',')

left_adjacent_ommatid_locations = adjacent_ommatid_locations[1:787, :]
right_adjacent_ommatid_locations = adjacent_ommatid_locations[787:1553, :]

print(left_adjacent_ommatid_locations)
print(right_adjacent_ommatid_locations)

left_ommatid_values_lpf = []
right_ommatid_values_lpf = []

#create copy of ommatid_values with low pass filter
t = np.linspace(0, 1.5, 300) # time points in [s]
for i in range(left_ommatid_values.shape[1]):
    u = left_ommatid_values[:, i]
    yf = low_pass_filter(t, u, 0.03)
    left_ommatid_values_lpf.append(yf)

for i in range(right_ommatid_values.shape[1]):
    u = right_ommatid_values[:, i]
    yf = low_pass_filter(t, u, 0.03)
    right_ommatid_values_lpf.append(yf)



# t = np.linspace(0, 1.5, 30)
# u = left_ommatid_values[:, 0]

# yf = low_pass_filter(t, u)
# print(yf)

# print(yf.shape)

# print(np.array(left_ommatid_values).shape)

left_ommatid_values = np.array(left_ommatid_values)
left_ommatid_values_lpf = np.array(left_ommatid_values_lpf)
left_ommatid_values_lpf = np.transpose(left_ommatid_values_lpf) # take the transpose
right_ommatid_values = np.array(right_ommatid_values)
right_ommatid_values_lpf = np.array(right_ommatid_values_lpf)
right_ommatid_values_lpf = np.transpose(right_ommatid_values_lpf) # take the transpose
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
for i in range(np.array(left_ommatid_values_lpf).shape[1]): # iterate over all ommatidia
    real_i = np.where(left_adjacent_ommatid_locations[:, 0] == i+1)[0].tolist()[0] # Change to in order of ommatidia in csv

    single_ommatidia_all_frames = []
    current_ommatid_position = left_adjacent_ommatid_locations[real_i, 0] - 1 # row numbers are 1-off from python indexing

    for j in range(np.array(left_ommatid_values_lpf).shape[0]): # iterate over all frames
        single_ommatidia_single_frame = []

        # iterate over all 6 neighbors
        for k in range(np.array(left_adjacent_ommatid_locations).shape[1]-1): 
            pp = left_adjacent_ommatid_locations[real_i, k+1] - 1 

            # make sure there is a nb
            if(np.isnan(current_ommatid_position) or np.isnan(pp)):
                comparison_value = np.nan
            else:
                # compare neighbors
                lp0 = left_ommatid_values_lpf[j, int(current_ommatid_position)]
                n1 = left_ommatid_values[j, int(pp)]

                lp1 = left_ommatid_values_lpf[j, int(pp)]
                n0 = left_ommatid_values[j, int(current_ommatid_position)]

                comparison_value = (lp0*n1) - (lp1*n0)

            single_ommatidia_single_frame.append(comparison_value)
        
        single_ommatidia_all_frames.append(single_ommatidia_single_frame)

        
    # final_result_3D_left.append(single_ommatidia_all_frames)
    # apply low-pass filter to the directional motion
    single_ommatidia_all_frames_lpf = []
    for m in range (6):
        t = np.linspace(0, 1.5, 300)
        u = np.array(single_ommatidia_all_frames)[:,m]
        vertical_slice_lpf = low_pass_filter(t, u, 0.003)
        single_ommatidia_all_frames_lpf.append(vertical_slice_lpf)
    final_result_3D_left.append(single_ommatidia_all_frames_lpf)

    final_result_3D_left_nolpf.append(single_ommatidia_all_frames)

print(np.array(final_result_3D_left).shape)
print(np.array(final_result_3D_left_nolpf).shape)

final_result_3D_left = np.array(final_result_3D_left)

#print(final_result_3D_left)

# with open('MotionDetector/final_result_3D_left.pkl', 'wb') as handle:
with open('MotionDetector/final_result_3D_left.pkl', 'wb') as handle:
    pickle.dump(final_result_3D_left, handle, protocol=pickle.HIGHEST_PROTOCOL)



from matplotlib import pyplot as plt
final_result_3D_left_nolpf = np.array(final_result_3D_left_nolpf)

i = 350
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.plot(left_ommatid_values[:,i])
ax.plot(left_ommatid_values_lpf[:,i])
# ax.plot(np.array(final_result_3D_left_nolpf)[i,0,:], color='red')
# ax.plot(np.array(final_result_3D_left_nolpf)[i,1,:], color='green')
# ax.plot(np.array(final_result_3D_left_nolpf)[i,2,:], color='black')

# ax.plot(np.array(final_result_3D_left_nolpf)[i,:,0]/1e2, color='red')
# ax.plot(np.array(final_result_3D_left_nolpf)[i,:,1], color='green')
# ax.plot(np.array(final_result_3D_left_nolpf)[i,:,2]/1e2, color='black')
np.array(final_result_3D_left_nolpf)[i,:,2].argmax()
# final_result_3D_left_nolpf[i,56,2]

f_max = 56
fig, ax = plt.subplots(1, figsize=(8, 6))

ax.plot(left_ommatid_values[56,:])
ax.plot(left_ommatid_values_lpf[56,:])
# ommatid1 = left_adjacent_ommatid_locations[:, 1]
# ommatid2 = left_adjacent_ommatid_locations[:, 2]
# ommatid3 = left_adjacent_ommatid_locations[:, 3]
# ommatid4 = left_adjacent_ommatid_locations[:, 4]
# ommatid5 = left_adjacent_ommatid_locations[:, 5]
# ommatid6 = left_adjacent_ommatid_locations[:, 6]
ommatid_indicies = [350,376,377,351,324,321,346]
fig, ax = plt.subplots(1, figsize=(8, 6))

# for i in range(left_adjacent_ommatid_locations.shape[1]-1):
#     ommatid_indicies.append([i+1])

print(ommatid_indicies)
# ax.plot(left_ommatid_values[14, :])
# ax.plot(left_ommatid_values_lpf[14, :])
# ax.plot(left_ommatid_values[14, ommatid_indicies])
# ax.plot(left_ommatid_values_lpf[14, ommatid_indicies])
frame_array = []
for i in range(10):
    frame_array.append(i+110)
fig, ax = plt.subplots(1, figsize=(8, 6))
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[0]], color = 'r')
# # ax.plot(left_ommatid_values[frame_array, ommatid_indicies[1]], color = 'b')
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[3]], color = 'y')
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[5]], color = 'g')
ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[0]], color = 'r')
# ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[1]], color = 'b')
ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[3]], color = 'y')
ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[5]], color = 'g')

fig, ax = plt.subplots(1, figsize=(8, 6))

# ax.plot(final_result_3D_left_nolpf[350, 0:120, 0], color = 'b')
# ax.plot(final_result_3D_left_nolpf[350, 0:120, 2], color = 'y')
# ax.plot(final_result_3D_left_nolpf[350, 0:120, 4], color = 'g')

ax.plot(final_result_3D_left[350, 0, 100:120], color = 'b')
ax.plot(final_result_3D_left[350, 2, 100:120], color = 'y')
ax.plot(final_result_3D_left[350, 4, 100:120], color = 'g')



# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[0]], color = 'r')
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[2]], color = 'b')
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[4]], color = 'y')
# ax.plot(left_ommatid_values[frame_array, ommatid_indicies[6]], color = 'g')
# ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[0]], color = 'r')
# ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[2]], color = 'b')
# ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[4]], color = 'y')
# ax.plot(left_ommatid_values_lpf[frame_array, ommatid_indicies[6]], color = 'g')