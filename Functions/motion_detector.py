import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import os
import pickle
import configparser

config = configparser.ConfigParser()

config.read("MetaTxt.txt")
readPath = config.get("data", "path")
csvData = config.get("data", "csvData")
csvNeighbors = config.get("data", "csvNeighbors")

config.read(readPath)

videoFrames = int(config.get("variables", "videoFrames"))
blenderFile = config.get("variables", "blenderFile")
videoName = config.get("variables", "videoName")

movement_data = config.items("movement")

with open("OutputData/" + videoName + "/f_de.pkl", "rb") as handle:
    # with open('final_result_3D_left.pkl', 'rb') as handle:
    final_result_3D = np.array(pickle.load(handle))

if not os.path.exists("OutputData/" + videoName + "/HR_Frames"):
    os.makedirs("OutputData/" + videoName + "/HR_Frames")

# transformation between Cartesian <-> spherical coordinates
from geometry_copy import cart2sph, sph2cart

# read in eyemap data, column 3:6 are [vx vy vz] viewing directions
data = np.genfromtxt(csvData, delimiter=",")
pts = data[778:1555, 3:6]

# convert to spherical coordinate in [r=1, theta, phi] in radian
xyz = pts
rtp_main = cart2sph(xyz)

# Mollweide projections, from 3d to 2d
from geometry_copy import sph2Mollweide

adjacent_ommatid_locations = np.genfromtxt(csvNeighbors, delimiter=",")

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
        # plt.quiver(xy[:, 0], xy[:, 1], quiver_coord_diff_x, quiver_coord_diff_y)
        plt.quiver(xy[:, 0], xy[:, 1], 1, 1)

        plt.xlabel("azimuth")
        plt.ylabel("elevation")

        # plt.show()
        plt.savefig(
            "OutputData/"
            + videoName
            + "/HR_Frames/"
            + str(g)
            + "-frame,"
            + str(h)
            + "-direction.png"
        )
        plt.clf()
