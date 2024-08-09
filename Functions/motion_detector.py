# Combines HR values with ommatidia directions and plots output
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import os
import pickle
import configparser

config = configparser.ConfigParser()

# MetaTxt to retrieve csv files and scene txt (later replace with a function)
config.read("MetaTxt.txt")
readPath = config.get("data", "path")
csvData = config.get("data", "csvData")
csvNeighborsl = config.get("data", "csvNeighborsLeft")
csvNeighborsr = config.get("data", "csvNeighborsRight")

# Load number of ommatidia in each eye (split)
data = np.genfromtxt(csvData, delimiter=",")
num_omm_r = data[np.flatnonzero(data[:, 7] == 1), :].shape[0]
num_omm_l = data[np.flatnonzero(data[:, 7] == 0), :].shape[0]

# Read in scene txt
config.read(readPath)
videoName = config.get("variables", "videoName")  # name of scene folder

# Load in hr output
with open("OutputData/" + videoName + "/f_de_l.pkl", "rb") as handle:
    f_result_3D_l = np.array(pickle.load(handle))
with open("OutputData/" + videoName + "/f_de_r.pkl", "rb") as handle:
    f_result_3D_r = np.array(pickle.load(handle))

# Create folders to save plots
if not os.path.exists("OutputData/" + videoName + "/HR_FramesRight"):
    os.makedirs("OutputData/" + videoName + "/HR_FramesRight")
if not os.path.exists("OutputData/" + videoName + "/HR_FramesLeft"):
    os.makedirs("OutputData/" + videoName + "/HR_FramesLeft")

# Transformation between Cartesian <-> spherical coordinates
from geometry import cart2sph, sph2cart

# Read in eyemap data, column 3:6 are [vx vy vz] viewing directions
data = np.genfromtxt(csvData, delimiter=",")
pts_l = data[num_omm_r + 1 :, 3:6]
pts_r = data[1 : num_omm_r + 1, 3:6]

# Indicies of neighbors
l_adj_omm_loc = np.genfromtxt(csvNeighborsl, delimiter=",")
l_adj_omm_loc = l_adj_omm_loc[1:, :]
r_adj_omm_loc = np.genfromtxt(csvNeighborsr, delimiter=",")
r_adj_omm_loc = r_adj_omm_loc[1:, :]


def motion_d(pts, aol, fr3, dir):
    xy_return = []
    q_coord_diff_x_no_mult = []
    q_coord_diff_y_no_mult = []
    q_coord_diff_x_mult = []
    q_coord_diff_y_mult = []

    # other_test = []

    direction = dir
    adj_omm_loc = aol
    f_result_3D = fr3
    xyz = pts

    # Convert to spherical coordinate in [r=1, theta, phi] in radian
    rtp_main = cart2sph(xyz)

    # Mollweide projections, from 3d to 2d
    from geometry import sph2Mollweide

    # Define guidelines
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

    # f_result_3D.shape[2]
    for g in range(f_result_3D.shape[2]):  # num of frames
        q_coord_diff_x_mult_temp = []
        q_coord_diff_y_mult_temp = []

        for h in range(f_result_3D.shape[1]):  # num of directions

            q_coord_diff_x = []
            q_coord_diff_y = []
            # other_test_temp1 = []

            for i in range(xy.shape[0]):  # num of ommatidia
                real_i = np.where(adj_omm_loc[:, 0] == i + 1)[0].tolist()[
                    0
                ]  # Change to in order of ommatidia in csv
                omm_position = adj_omm_loc[real_i, 0] - 1
                omm_pos = adj_omm_loc[real_i, :] - 1

                # Make sure all 6 neighbored exist
                nan_flag = False
                for z in range(7):
                    co = omm_pos[z]
                    if np.isnan(co):
                        nan_flag = True
                    elif np.isnan(xy[int(co), 0]) or np.isnan(xy[int(co), 1]):
                        nan_flag = True

                # Append nan if no neighbor
                if nan_flag:
                    q_coord_diff_x.append(np.nan)
                    q_coord_diff_y.append(np.nan)

                # Averaging for 4 neighbors, hardcoded
                else:
                    if h == 0:
                        vx0 = (
                            xy[int(omm_pos[0]), 0]
                            + xy[int(omm_pos[2]), 0]
                            + xy[int(omm_pos[5]), 0]
                        ) / 3
                        vx1 = (xy[int(omm_pos[3]), 0] + xy[int(omm_pos[4]), 0]) / 2
                        vx = vx0 - vx1
                        vy0 = (
                            xy[int(omm_pos[0]), 1]
                            + xy[int(omm_pos[2]), 1]
                            + xy[int(omm_pos[5]), 1]
                        ) / 3
                        vy1 = (xy[int(omm_pos[3]), 1] + xy[int(omm_pos[4]), 1]) / 2
                        vy = vy0 - vy1
                    elif h == 1:
                        vx0 = (
                            xy[int(omm_pos[0]), 0]
                            + xy[int(omm_pos[2]), 0]
                            + xy[int(omm_pos[5]), 0]
                        ) / 3
                        vx1 = (xy[int(omm_pos[1]), 0] + xy[int(omm_pos[6]), 0]) / 2
                        vx = vx0 - vx1
                        vy0 = (
                            xy[int(omm_pos[0]), 1]
                            + xy[int(omm_pos[2]), 1]
                            + xy[int(omm_pos[5]), 1]
                        ) / 3
                        vy1 = (xy[int(omm_pos[1]), 1] + xy[int(omm_pos[6]), 1]) / 2
                        vy = vy0 - vy1
                    elif h == 2:
                        vx = xy[int(omm_position), 0] - xy[int(omm_pos[2]), 0]
                        vy = xy[int(omm_position), 1] - xy[int(omm_pos[2]), 1]
                    else:
                        vx = xy[int(omm_position), 0] - xy[int(omm_pos[5]), 0]
                        vy = xy[int(omm_position), 1] - xy[int(omm_pos[5]), 1]

                    # Normalize
                    divide_factor = np.sqrt(np.power(vx, 2) + np.power(vy, 2))
                    vx = -1 * vx / divide_factor
                    vy = -1 * vy / divide_factor

                    # Multiply normalized direction by hr output
                    # vx = vx * f_result_3D[i, h, g]
                    # vy = vy * f_result_3D[i, h, g]

                    q_coord_diff_x.append(vx)
                    q_coord_diff_y.append(vy)

            # Plot
            plt.plot(meridians_xy[:, 0], meridians_xy[:, 1], "-k", linewidth=1.0)
            plt.plot(n45_xy[:, 0], n45_xy[:, 1], "-k", linewidth=1)
            plt.plot(eq_xy[:, 0], eq_xy[:, 1], "-k", linewidth=1)
            plt.plot(s45_xy[:, 0], s45_xy[:, 1], "-k", linewidth=1)

            plt.quiver(
                xy[:, 0],
                xy[:, 1],
                (q_coord_diff_x * f_result_3D[:, h, g]),
                (q_coord_diff_y * f_result_3D[:, h, g]),
            )
            plt.xlabel("azimuth")
            plt.ylabel("elevation")

            # Save plots
            if direction == "left":
                plt.savefig(
                    "OutputData/"
                    + videoName
                    + "/HR_FramesLeft/"
                    + str(g)
                    + "-frame,"
                    + str(h)
                    + "-direction.png"
                )
            elif direction == "right":
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

            # other_test_temp1.append(xy[:, 0])
            # other_test_temp1.append(xy[:, 1])
            # other_test_temp1.append(q_coord_diff_x*f_result_3D[:, h, g])
            # other_test_temp1.append(q_coord_diff_y*f_result_3D[:, h, g])

            plt.clf()

            if g == 0:
                q_coord_diff_x_no_mult.append(q_coord_diff_x)
                q_coord_diff_y_no_mult.append(q_coord_diff_y)

            # q_coord_diff_x_mult_temp.append(q_coord_diff_x * f_result_3D[:, h, g])
            # q_coord_diff_y_mult_temp.append(q_coord_diff_y * f_result_3D[:, h, g])
            q_coord_diff_x_mult_temp.append(q_coord_diff_x * f_result_3D[:, h, g])
            q_coord_diff_y_mult_temp.append(q_coord_diff_y * f_result_3D[:, h, g])
            # print("q", np.array(q_coord_diff_x_mult_temp).shape)

        # other_test.append(other_test_temp1)

        q_coord_diff_x_mult.append(q_coord_diff_x_mult_temp)
        q_coord_diff_y_mult.append(q_coord_diff_y_mult_temp)

    xy_return = xy

    xy_return = np.array(xy_return)
    q_coord_diff_x_no_mult = np.array(q_coord_diff_x_no_mult)
    q_coord_diff_y_no_mult = np.array(q_coord_diff_y_no_mult)
    q_coord_diff_x_mult = np.array(q_coord_diff_x_mult)
    q_coord_diff_y_mult = np.array(q_coord_diff_y_mult)

    q_coord_diff_x_no_mult = np.transpose(q_coord_diff_x_no_mult)
    q_coord_diff_y_no_mult = np.transpose(q_coord_diff_y_no_mult)
    q_coord_diff_x_mult = np.transpose(q_coord_diff_x_mult)
    q_coord_diff_y_mult = np.transpose(q_coord_diff_y_mult)

    print("xy_return ", xy_return.shape)
    print("q_coord_diff_x_no_mult ", q_coord_diff_x_no_mult.shape)
    print("q_coord_diff_y_no_mult ", q_coord_diff_y_no_mult.shape)
    print("q_coord_diff_x_mult ", q_coord_diff_x_mult.shape)
    print("q_coord_diff_y_mult ", q_coord_diff_y_mult.shape)
    # print("other_test ", np.array(other_test).shape)

    # Save output to "Pkls folder" for lptc_func
    with open("Pkls/xy_return_" + direction + ".pkl", "wb") as handle:
        pickle.dump(xy_return, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("Pkls/q_coord_diff_x_no_mult_" + direction + ".pkl", "wb") as handle:
        pickle.dump(q_coord_diff_x_no_mult, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("Pkls/q_coord_diff_y_no_mult_" + direction + ".pkl", "wb") as handle:
        pickle.dump(q_coord_diff_y_no_mult, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("Pkls/q_coord_diff_x_mult_" + direction + ".pkl", "wb") as handle:
        pickle.dump(q_coord_diff_x_mult, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("Pkls/q_coord_diff_y_mult_" + direction + ".pkl", "wb") as handle:
        pickle.dump(q_coord_diff_y_mult, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Run method for left and right eye
motion_d(pts_l, l_adj_omm_loc, f_result_3D_l, "left")
motion_d(pts_r, r_adj_omm_loc, f_result_3D_r, "right")
