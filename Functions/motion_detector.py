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
csvNeighborsLeft = config.get("data", "csvNeighborsLeft")
csvNeighborsRight = config.get("data", "csvNeighborsRight")
direction = int(config.get("data", "direction"))

data = np.genfromtxt(csvData, delimiter=",")
num_omm_right = data[np.flatnonzero(data[:, 7] == 1), :].shape[0]
num_omm_left = data[np.flatnonzero(data[:, 7] == 0), :].shape[0]

config.read(readPath)
videoName = config.get("variables", "videoName")

def onclick(event):
    if ax := event.inaxes:
        print(ax.figure.axes.index(ax), ax._position.bounds, sep='\n')


with open("OutputData/" + videoName + "/f_de_l.pkl", "rb") as handle:
    final_result_3D_left = np.array(pickle.load(handle))

with open("OutputData/" + videoName + "/f_de_r.pkl", "rb") as handle:
    final_result_3D_right = np.array(pickle.load(handle))

if not os.path.exists("OutputData/" + videoName + "/HR_FramesRight"):
    os.makedirs("OutputData/" + videoName + "/HR_FramesRight")

if not os.path.exists("OutputData/" + videoName + "/HR_FramesLeft"):
    os.makedirs("OutputData/" + videoName + "/HR_FramesLeft")

# transformation between Cartesian <-> spherical coordinates
from geometry import cart2sph, sph2cart

# read in eyemap data, column 3:6 are [vx vy vz] viewing directions
data = np.genfromtxt(csvData, delimiter=",")
pts_l = data[num_omm_right + 1 :, 3:6]
pts_r = data[1 : num_omm_right + 1, 3:6]

adjacent_ommatid_locations_left = np.genfromtxt(csvNeighborsLeft, delimiter=",")
adjacent_ommatid_locations_right = np.genfromtxt(csvNeighborsRight, delimiter=",")


# convert to spherical coordinate in [r=1, theta, phi] in radian
def motion_d(pts, aol, fr3, dir):
    direction = dir
    adjacent_ommatid_locations = aol
    final_result_3D = fr3
    xyz = pts
    rtp_main = cart2sph(xyz)

    # Mollweide projections, from 3d to 2d
    from geometry import sph2Mollweide

    adjacent_ommatid_locations = adjacent_ommatid_locations[1:, :]

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

    for g in range(final_result_3D.shape[2]):
        for h in range(final_result_3D.shape[1]):
            # compute directional arrows

            quiver_coord_diff_x1 = []
            quiver_coord_diff_y1 = []

            quiver_coord_diff_x2 = []
            quiver_coord_diff_y2 = []

            for i in range(xy.shape[0]):
                real_i = np.where(adjacent_ommatid_locations[:, 0] == i + 1)[
                    0
                ].tolist()[
                    0
                ]  # Change to in order of ommatidia in csv
                current_ommatid_position = adjacent_ommatid_locations[real_i, 0] - 1
                current_ommatid_positions = adjacent_ommatid_locations[real_i, :] - 1

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
                        vx0 = (
                            xy[int(current_ommatid_positions[0]), 0]
                            + xy[int(current_ommatid_positions[2]), 0]
                            + xy[int(current_ommatid_positions[5]), 0]
                        ) / 3
                        vx1 = (
                            xy[int(current_ommatid_positions[3]), 0]
                            + xy[int(current_ommatid_positions[4]), 0]
                        ) / 2
                        vx = vx0 - vx1

                        vy0 = (
                            xy[int(current_ommatid_positions[0]), 1]
                            + xy[int(current_ommatid_positions[2]), 1]
                            + xy[int(current_ommatid_positions[5]), 1]
                        ) / 3
                        vy1 = (
                            xy[int(current_ommatid_positions[3]), 1]
                            + xy[int(current_ommatid_positions[4]), 1]
                        ) / 2
                        vy = vy0 - vy1

                    elif h == 1:
                        vx0 = (
                            xy[int(current_ommatid_positions[0]), 0]
                            + xy[int(current_ommatid_positions[2]), 0]
                            + xy[int(current_ommatid_positions[5]), 0]
                        ) / 3
                        vx1 = (
                            xy[int(current_ommatid_positions[1]), 0]
                            + xy[int(current_ommatid_positions[6]), 0]
                        ) / 2
                        vx = vx0 - vx1

                        vy0 = (
                            xy[int(current_ommatid_positions[0]), 1]
                            + xy[int(current_ommatid_positions[2]), 1]
                            + xy[int(current_ommatid_positions[5]), 1]
                        ) / 3
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

                    vx = -1*vx / divide_factor
                    vy = -1*vy / divide_factor

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
            plt.quiver(xy[:, 0], xy[:, 1], quiver_coord_diff_x, quiver_coord_diff_y)

            plt.xlabel("azimuth")
            plt.ylabel("elevation")
            if direction == 0:
                # if(g == 60):
                #     #cid = fig.canvas.mpl_connect('button_press_event', onclick)
                #     plt.show()
                plt.savefig(
                    "OutputData/"
                    + videoName
                    + "/HR_FramesLeft/"
                    + str(g)
                    + "-frame,"
                    + str(h)
                    + "-direction.png"
                )
            elif direction == 1:
                # plt.show()
                plt.savefig(
                    "OutputData/"
                    + videoName
                    + "/HR_FramesRight/"
                    + str(g)
                    + "-frame,"
                    + str(h)
                    + "-direction.png"
                )

            plt.clf()


motion_d(pts_l, adjacent_ommatid_locations_left, final_result_3D_left, 0)
motion_d(pts_r, adjacent_ommatid_locations_right, final_result_3D_right, 1)
