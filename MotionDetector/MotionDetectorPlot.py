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
#with open('final_result_3D_left.pkl', 'rb') as handle:
    final_result_3D = np.array(pickle.load(handle))


# %%
# transformation between Cartesian <-> spherical coordinates
from geometry_copy import cart2sph, sph2cart

# read in eyemap data, column 3:6 are [vx vy vz] viewing directions
data = np.genfromtxt('Projection2D/data_extraction_test_data.csv', delimiter=',')
#data = np.genfromtxt('../Projection2D/data_extraction_test_data.csv', delimiter=',') 
pts = data[1:787, 3:6]
ommatid_data = np.genfromtxt('Projection2D/ommatid_data.csv', delimiter=',')
#ommatid_data = np.genfromtxt('../Projection2D/ommatid_data.csv', delimiter=',')
plot_colors = ommatid_data[0:786]/255.0
print(plot_colors)

# convert to spherical coordinate in [r=1, theta, phi] in radia n
# rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)

# convert to Cartesian
# xyz = sph2cart(rtp)
xyz = pts
# convert to spherical
rtp_main = cart2sph(xyz)

# %%
# Mollweide projections, from 3d to 2d
from geometry_copy import sph2Mollweide

adjacent_ommatid_locations = np.genfromtxt('MotionDetector/ind_nb.csv', delimiter=',')
#adjacent_ommatid_locations = np.genfromtxt('ind_nb.csv', delimiter=',')

# define guidelines
ww = np.stack((np.linspace(0,180,19), np.repeat(-180,19)), axis=1)
w = np.stack((np.linspace(180,0,19), np.repeat(-90,19)), axis=1)
m = np.stack((np.linspace(0,180,19), np.repeat(0,19)), axis=1)
e = np.stack((np.linspace(180,0,19), np.repeat(90,19)), axis=1)
ee = np.stack((np.linspace(0,180,19), np.repeat(180,19)), axis=1)
pts = np.vstack((ww,w,m,e,ee))
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
meridians_xy = sph2Mollweide(rtp[:,1:3])

pts = np.stack((np.repeat(45,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
n45_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(90,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
eq_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(135,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
s45_xy = sph2Mollweide(rtp[:,1:3])

# use rtp from earlier

# # pick a few points
# plot_pts = data[1:787, 3:6]
# print(plot_pts)

# cart_plot_pts = cart2sph(plot_pts)

# print(plot_pts)

# convert to spherical coordinate in [r=1, theta, phi] in radian
#rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
# xy = sph2Mollweide(cart_plot_pts[:,1:3])

xy = sph2Mollweide(rtp_main[:,1:3])

# plot this
#plt.scatter(xy[:,0], xy[:,1], marker='o', s=10)


for g in range(300):
    for h in range(3):
        # for i in range(np.array(xy).shape[0]):
        #     plt.quiver(xy[i,0], xy[i,1], 0, 1)

        #plt.quiver(xy[:,0], xy[:,1], 0, 0, 'blue')
        #plt.plot(xy[:,0], xy[:,1], [1, 0, 0])
        #plt.scatter(xy[:,0], xy[:,1], c = 'blue')  

        # compute direcitonal arrows
        quiver_coord_diff_x1 = []
        quiver_coord_diff_y1 = []

        quiver_coord_diff_x2 = []
        quiver_coord_diff_y2 = []

        

        left_adjacent_ommatid_locations = adjacent_ommatid_locations[1:787, :]
        right_adjacent_ommatid_locations = adjacent_ommatid_locations[787:1552, :]


        for i in range(xy.shape[0]):

            current_ommatid_position = left_adjacent_ommatid_locations[i, 0] - 1
            adjacent_ommatid_position = left_adjacent_ommatid_locations[i, h+1] - 1

            if(np.isnan(current_ommatid_position) or np.isnan(adjacent_ommatid_position)):
                quiver_coord_diff_x1.append(np.nan)
                quiver_coord_diff_y1.append(np.nan)
            # if not nan, 
            else:
                vx = xy[int(current_ommatid_position), 0] - xy[int(adjacent_ommatid_position), 0]
                vy = xy[int(current_ommatid_position), 1] - xy[int(adjacent_ommatid_position), 1]

                divide_factor = np.sqrt(np.power(vx,2) + np.power(vy,2))

                vx = vx/divide_factor
                vy = vy/divide_factor

                vx = vx*final_result_3D[i, h, g]
                vy = vy*final_result_3D[i, h, g]

                quiver_coord_diff_x1.append(vx)
                quiver_coord_diff_y1.append(vy)


        # now do the same for the opposite direction
            current_ommatid_position = left_adjacent_ommatid_locations[i, 0] - 1
            adjacent_ommatid_position = left_adjacent_ommatid_locations[i, h+4] - 1

            if(np.isnan(current_ommatid_position) or np.isnan(adjacent_ommatid_position)):
                quiver_coord_diff_x2.append(np.nan)
                quiver_coord_diff_y2.append(np.nan)
            
            else:

                vx = xy[int(current_ommatid_position), 0] - xy[int(adjacent_ommatid_position), 0]
                vy = xy[int(current_ommatid_position), 1] - xy[int(adjacent_ommatid_position), 1]

                divide_factor = np.sqrt(np.power(vx,2) + np.power(vy,2))

                vx = vx/divide_factor
                vy = vy/divide_factor

                vx = vx*final_result_3D[i, h+3, g]
                vy = vy*final_result_3D[i, h+3, g]

                quiver_coord_diff_x2.append(vx)
                quiver_coord_diff_y2.append(vy)

        # TODO, 
        # 1. bring up the later section, multiply the normailzed vector with the 3D motion values (Nx6)
        # 2. subtract the opposite 2D directional vectors
        # add up (subtract) opposite directions
        quiver_coord_diff_x = np.subtract(quiver_coord_diff_x1, quiver_coord_diff_x2)
        quiver_coord_diff_y = np.subtract(quiver_coord_diff_y1, quiver_coord_diff_y2)

        #print(final_result_3D.shape)

        # final_result_3D_slice = []
        # final_result_3D_slice_no_nan = []

        # for i in range(final_result_3D.shape[0]):
        #     final_result_3D_slice.append(float(final_result_3D[i, 0, g:g+1]))

        #     if(not np.isnan(final_result_3D[i, 0, 0:1])):
        #         final_result_3D_slice_no_nan.append(float(final_result_3D[i, 0, g:g+1]))

        # final_result_3D_slice = np.array(final_result_3D_slice)
        # final_result_3D_slice_no_nan = np.array(final_result_3D_slice_no_nan)

        #print(final_result_3D_slice)

        #print(final_result_3D_slice_no_nan.shape)

        #print(np.max(final_result_3D_slice_no_nan))
        #print(np.min(final_result_3D_slice_no_nan))
        #print(final_result_3D_slice.shape)

        #print(np.max(final_result_3D_slice_no_nan))
        #print(np.min(final_result_3D_slice_no_nan))

        # PLOT
        plt.plot(meridians_xy[:,0], meridians_xy[:,1], '-k', linewidth=1.0)
        plt.plot(n45_xy[:,0], n45_xy[:,1], '-k', linewidth=1)
        plt.plot(eq_xy[:,0], eq_xy[:,1], '-k', linewidth=1)
        plt.plot(s45_xy[:,0], s45_xy[:,1], '-k', linewidth=1)

        # change scales to 1
        plt.quiver(xy[:,0], xy[:,1], quiver_coord_diff_x, quiver_coord_diff_y)

        plt.xlabel("azimuth")
        plt.ylabel("elevation")

        #plt.show()
        plt.savefig('MotionDetector/HR_Figures/' + str(g) + '-frame,' + str(h) + '-direction.png')
        plt.clf()

# # 3d interactive plot
# fig = plt.figure()
# plt.scatter(xy[:,0], xy[:,1], c='b', marker='o', s=10)

# ind_xy_temp = np.genfromtxt('MotionDetector/ind_xy.csv', delimiter=',')

# ind_xy = ind_xy_temp[1:788, 0:1]

# xy_cart = ind_xy_temp[1:788, 0:3]

# ind = xy_cart[xy_cart[:, 1]%3 == 0, 0]
# ind = ind.astype(int)

# # plot points whose ind_xy[:,1] is divisible by 3 with a bigger marker
# plt.scatter(xy[ind,0], xy[ind,1], c='r', marker='o', s=20)

# # change the view angle
# plt.show()
