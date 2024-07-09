# %%
# plotting setup
# matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
from cmath import pi

# import pandas as pd

import pickle

# # set working dir
# import os
# os.chdir('../')

with open("MotionDetector/final_result_3D_left.pkl", "rb") as handle:
    # with open('final_result_3D_left.pkl', 'rb') as handle:
    final_result_3D = np.array(pickle.load(handle))

# fig, ax = plt.subplots(1, figsize=(8, 6))

# ax.plot(final_result_3D[350, 0, 100:120], color = 'b')
# ax.plot(final_result_3D[350, 2, 100:120], color = 'y')
# ax.plot(final_result_3D[350, 4, 100:120], color = 'g')

# %%
# transformation between Cartesian <-> spherical coordinates
from geometry_copy import cart2sph, sph2cart

# read in eyemap data, column 3:6 are [vx vy vz] viewing directions
data = np.genfromtxt("default_eye_data.csv", delimiter=",")
# data = np.genfromtxt('../Projection2D/data_extraction_test_data.csv', delimiter=',')
pts = data[1:787, 3:6]
# ommatid_data = np.genfromtxt("Projection2D/ommatid_data.csv", delimiter=",")
# # ommatid_data = np.genfromtxt('../Projection2D/ommatid_data.csv', delimiter=',')
# plot_colors = ommatid_data[0:786] / 255.0
# print(plot_colors)

# convert to spherical coordinate in [r=1, theta, phi] in radian
xyz = pts
rtp_main = cart2sph(xyz)

# %%
# Mollweide projections, from 3d to 2d
from geometry_copy import sph2Mollweide

adjacent_ommatid_locations = np.genfromtxt("MotionDetector/ind_nb.csv", delimiter=",")
# adjacent_ommatid_locations = np.genfromtxt('ind_nb.csv', delimiter=',')

# define guidelines
ww = np.stack((np.linspace(0, 180, 19), np.repeat(-180, 19)), axis=1)
w = np.stack((np.linspace(180, 0, 19), np.repeat(-90, 19)), axis=1)
m = np.stack((np.linspace(0, 180, 19), np.repeat(0, 19)), axis=1)
e = np.stack((np.linspace(180, 0, 19), np.repeat(90, 19)), axis=1)
ee = np.stack((np.linspace(0, 180, 19), np.repeat(180, 19)), axis=1)
pts = np.vstack((ww, w, m, e, ee))
rtp = np.insert(pts / 180 * pi, 0, np.repeat(1, pts.shape[0]), axis=1)
meridians_xy = sph2Mollweide(rtp[:, 1:3])

pts = np.stack((np.repeat(45, 37), np.linspace(-180, 180, 37)), axis=1)
rtp = np.insert(pts / 180 * pi, 0, np.repeat(1, pts.shape[0]), axis=1)
n45_xy = sph2Mollweide(rtp[:, 1:3])
pts = np.stack((np.repeat(90, 37), np.linspace(-180, 180, 37)), axis=1)
rtp = np.insert(pts / 180 * pi, 0, np.repeat(1, pts.shape[0]), axis=1)
eq_xy = sph2Mollweide(rtp[:, 1:3])
pts = np.stack((np.repeat(135, 37), np.linspace(-180, 180, 37)), axis=1)
rtp = np.insert(pts / 180 * pi, 0, np.repeat(1, pts.shape[0]), axis=1)
s45_xy = sph2Mollweide(rtp[:, 1:3])

xy = sph2Mollweide(rtp_main[:, 1:3])

# fig, ax = plt.subplots(1, figsize=(8, 6))

# ax.scatter(xy[:, 0], xy[:,1], color = 'b')
# ax.scatter(xy[351, 0], xy[351, 1], color = 'y')
# ax.scatter(xy[352, 0], xy[352, 1], color = 'cyan')

for g in range(300):
    for h in range(4):
        # compute directional arrows
        quiver_coord_diff_x1 = []
        quiver_coord_diff_y1 = []

        quiver_coord_diff_x2 = []
        quiver_coord_diff_y2 = []

        left_adjacent_ommatid_locations = adjacent_ommatid_locations[1:787, :]
        right_adjacent_ommatid_locations = adjacent_ommatid_locations[787:1552, :]

        for i in range(xy.shape[0]):
            real_i = np.where(left_adjacent_ommatid_locations[:, 0] == i + 1)[
                0
            ].tolist()[
                0
            ]  # Change to in order of ommatidia in csv
            current_ommatid_position = left_adjacent_ommatid_locations[real_i, 0] - 1
            current_ommatid_positions = left_adjacent_ommatid_locations[real_i, :] - 1

            nan_flag = False

            for z in range(7):
                co = current_ommatid_positions[z]
                if np.isnan(co):
                    nan_flag = True
                elif np.isnan(xy[int(co), 0]) or np.isnan(xy[int(co), 1]):
                    nan_flag = True

            if nan_flag:
                quiver_coord_diff_x1.append(np.nan)
                quiver_coord_diff_y1.append(np.nan)
            else:
                if h == 0:
                    vx0 = xy[int(current_ommatid_position), 0]
                    vx1 = (
                        xy[int(current_ommatid_positions[3]), 0]
                        + xy[int(current_ommatid_positions[4]), 0]
                    ) / 2
                    vx = vx0 - vx1

                    vy0 = xy[int(current_ommatid_position), 1]
                    vy1 = (
                        xy[int(current_ommatid_positions[3]), 1]
                        + xy[int(current_ommatid_positions[4]), 1]
                    ) / 2
                    vy = vy0 - vy1

                elif h == 1:
                    vx0 = xy[int(current_ommatid_position), 0]
                    vx1 = (
                        xy[int(current_ommatid_positions[1]), 0]
                        + xy[int(current_ommatid_positions[6]), 0]
                    ) / 2
                    vx = vx0 - vx1

                    vy0 = xy[int(current_ommatid_position), 1]
                    vy1 = (
                        xy[int(current_ommatid_positions[1]), 1]
                        + xy[int(current_ommatid_positions[6]), 1]
                    ) / 2
                    vy = vy0 - vy1

                elif h == 2:
                    vx = (
                        xy[int(current_ommatid_position), 0]
                        - xy[int(current_ommatid_positions[2]), 0]
                    )
                    vy = (
                        xy[int(current_ommatid_position), 1]
                        - xy[int(current_ommatid_positions[2]), 1]
                    )

                else:
                    vx = (
                        xy[int(current_ommatid_position), 0]
                        - xy[int(current_ommatid_positions[5]), 0]
                    )
                    vy = (
                        xy[int(current_ommatid_position), 1]
                        - xy[int(current_ommatid_positions[5]), 1]
                    )

                divide_factor = np.sqrt(np.power(vx, 2) + np.power(vy, 2))

                vx = vx / divide_factor
                vy = vy / divide_factor

                vx = vx * final_result_3D[i, h, g]
                vy = vy * final_result_3D[i, h, g]

                # vx = (vx/100)+1
                # vy = (vx/100)+1

                quiver_coord_diff_x1.append(vx)
                quiver_coord_diff_y1.append(vy)

        # Subtract the opposite 2D directional vectors
        quiver_coord_diff_x = quiver_coord_diff_x1
        quiver_coord_diff_y = quiver_coord_diff_y1

        # Plot
        plt.plot(meridians_xy[:, 0], meridians_xy[:, 1], "-k", linewidth=1.0)
        plt.plot(n45_xy[:, 0], n45_xy[:, 1], "-k", linewidth=1)
        plt.plot(eq_xy[:, 0], eq_xy[:, 1], "-k", linewidth=1)
        plt.plot(s45_xy[:, 0], s45_xy[:, 1], "-k", linewidth=1)

        # change scales to 1
        plt.quiver(xy[:, 0], xy[:, 1], quiver_coord_diff_x, quiver_coord_diff_y)

        plt.xlabel("azimuth")
        plt.ylabel("elevation")

        # plt.show()
        plt.savefig(
            "HR_Figures/"
            + str(g)
            + "-frame,"
            + str(h)
            + "-direction.png"
        )
        plt.clf()
