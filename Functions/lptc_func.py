# Saves signal and plot of signal for every neuron to LPTC_Output folder
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import pickle
import os


def lptc_func(lptc_csv, dir, number):
    lptc_data = lptc_csv
    direction = dir
    folder_name = number

    # Create folders to save plots
    if not os.path.exists("LPTC_Output/LPTC_Figures/" + folder_name + direction):
        os.makedirs("LPTC_Output/LPTC_Figures/" + folder_name + direction)
    if not os.path.exists(
        "LPTC_Output/LPTC_Figures_No_Mult/" + folder_name + direction
    ):
        os.makedirs("LPTC_Output/LPTC_Figures_No_Mult/" + folder_name + direction)
    if not os.path.exists("LPTC_Output/LPTC_Signals"):
        os.makedirs("LPTC_Output/LPTC_Signals")

    # Alter LPTC_Data
    for i in range(lptc_data.shape[0]):
        # Change max to 1, other value 2, for the first two columns
        # Both remain 0 if they are both 0
        lptc_row_max = 0
        lptc_row_max_pos = -1
        for j in range(0, 2):
            if lptc_data[i, j] > lptc_row_max:
                lptc_row_max = lptc_data[i, j]
                lptc_row_max_pos = j
        for k in range(0, 2):
            if k != lptc_row_max_pos or lptc_row_max_pos == -1:
                lptc_data[i, k] = 0
            else:
                lptc_data[i, k] = 1

        # Repeat for the next two columns
        lptc_row_max = 0
        lptc_row_max_pos = -1
        for j in range(2, 4):
            if lptc_data[i, j] > lptc_row_max:
                lptc_row_max = lptc_data[i, j]
                lptc_row_max_pos = j
        for k in range(2, 4):
            if k != lptc_row_max_pos or lptc_row_max_pos == -1:
                lptc_data[i, k] = 0
            else:
                lptc_data[i, k] = 1

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
    print(lptc_data.shape)

    # all_frame_all_omm = []
    # for i in range(q_coord_diff_x_mult.shape[2]):
    #     single_frame_all_omm = []
    #     for j in range(q_coord_diff_x_mult.shape[1]):
    #         single_frame_single_omm = []
    #         for k in range(q_coord_diff_x_mult.shape[0]):
    #             lptc_value_x = q_coord_diff_x_no_mult[k, j] * lptc_data[k, j]
    #             lptc_value_y = q_coord_diff_y_no_mult[k, j] * lptc_data[k, j]
    #             hr_value_x = q_coord_diff_x_mult[k, j, i]
    #             hr_value_y = q_coord_diff_y_mult[k, j, i]
    #             cos_comp = np.dot(
    #                 [lptc_value_x, lptc_value_y], [hr_value_x, hr_value_y]
    #             ) / (
    #                 np.linalg.norm([lptc_value_x, lptc_value_y])
    #                 * np.linalg.norm([hr_value_x, hr_value_y])
    #             )
    #             single_frame_single_omm.append(cos_comp)
    #         single_frame_all_omm.append(single_frame_single_omm)
    #     all_frame_all_omm.append(single_frame_all_omm)

    # all_frame_all_omm = np.array(all_frame_all_omm)
    # print(all_frame_all_omm.shape)

    struct_data = np.genfromtxt("InputData/lens_opticAxis_20240701.csv", delimiter=",")
    pts_l = struct_data[1:778, 3:6]
    pts_r = struct_data[778:, 3:6]

    from geometry import cart2sph, sph2cart

    # Convert to spherical coordinate in [r=1, theta, phi] in radian
    rtp_main = cart2sph(pts_l)

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

    # Multiplies signal with lptc_data 0s and 1s
    for i in range(300):
        q_coord_diff_x_mult[:, :, i] = q_coord_diff_x_mult[:, :, i] * lptc_data
        q_coord_diff_y_mult[:, :, i] = q_coord_diff_y_mult[:, :, i] * lptc_data

    magnitudes = np.sqrt(
        np.power(q_coord_diff_x_mult, 2) + np.float_power(q_coord_diff_y_mult, 2)
    )

    # magnitudes = q_coord_diff_x_mult

    # Calculates signal value for each frame
    mag_vals = 0
    num_vals = 0
    signal = []
    for i in range(magnitudes.shape[2]):
        for j in range(magnitudes.shape[1]):
            for k in range(magnitudes.shape[0]):
                if magnitudes[k, j, i] != 0 and np.isnan(magnitudes[k, j, i]) == False:
                    mag_vals += magnitudes[k, j, i]
                    num_vals += 1
        if num_vals == 0:
            signal.append(0)
        else:
            signal.append(mag_vals / num_vals)
    # pickle "LPTC_Output/LPTC_Signals"
    with open(
        "LPTC_Output/LPTC_Signals/" + folder_name + direction + ".pkl", "wb"
    ) as handle:
        pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plots 4 normalized plots multiplied by lptc_data to visualize
    for a in range(4):
        # Plot
        plt.plot(meridians_xy[:, 0], meridians_xy[:, 1], "-k", linewidth=1.0)
        plt.plot(n45_xy[:, 0], n45_xy[:, 1], "-k", linewidth=1)
        plt.plot(eq_xy[:, 0], eq_xy[:, 1], "-k", linewidth=1)
        plt.plot(s45_xy[:, 0], s45_xy[:, 1], "-k", linewidth=1)

        # plt.quiver(
        #     xy_return[:, 0],
        #     xy_return[:, 1],
        #     all_frame_all_omm[0, a, :],
        #     all_frame_all_omm[0, a, :],
        # )

        plt.quiver(
            xy_return[:, 0],
            xy_return[:, 1],
            q_coord_diff_x_no_mult[:, a] * lptc_data[:, a],
            q_coord_diff_y_no_mult[:, a] * lptc_data[:, a],
        )

        # plt.quiver(
        #     xy_return[:, 0],
        #     xy_return[:, 1],
        #     q_coord_diff_x_no_mult[:, a],
        #     q_coord_diff_y_no_mult[:, a],
        # )

        plt.xlabel("azimuth")
        plt.ylabel("elevation")

        if direction == "left":
            plt.savefig(
                "LPTC_Output/LPTC_Figures_No_Mult/"
                + folder_name
                + direction
                + "/"
                + str(a)
                + "-direction.png"
            )
        elif direction == "right":
            # plt.show()
            plt.savefig(
                "LPTC_Output/LPTC_Figures_No_Mult/"
                + folder_name
                + direction
                + "/"
                + str(a)
                + "-direction.png"
            )

        # plt.show()
        plt.clf()

    comb_q_mult_x = []
    comb_q_mult_y = []

    # Add up the 4 directions to a single direction for the input signal (changes shape from 3D to 2D)
    for i in range(300):
        comb_q_mult_x.append(
            q_coord_diff_x_mult[:, 0, i]
            + q_coord_diff_x_mult[:, 1, i]
            + q_coord_diff_x_mult[:, 2, i]
            + q_coord_diff_x_mult[:, 3, i]
        )
        comb_q_mult_y.append(
            q_coord_diff_y_mult[:, 0, i]
            + q_coord_diff_y_mult[:, 1, i]
            + q_coord_diff_y_mult[:, 2, i]
            + q_coord_diff_y_mult[:, 3, i]
        )

    comb_q_mult_x = np.array(comb_q_mult_x)
    comb_q_mult_y = np.array(comb_q_mult_y)

    # Plots 1 non-normalized plot multiplied by lptc_data 300 times, 1 per frame
    for b in range(300):

        # Plot
        plt.plot(meridians_xy[:, 0], meridians_xy[:, 1], "-k", linewidth=1.0)
        plt.plot(n45_xy[:, 0], n45_xy[:, 1], "-k", linewidth=1)
        plt.plot(eq_xy[:, 0], eq_xy[:, 1], "-k", linewidth=1)
        plt.plot(s45_xy[:, 0], s45_xy[:, 1], "-k", linewidth=1)

        # plt.quiver(
        #     xy_return[:, 0],
        #     xy_return[:, 1],
        #     all_frame_all_omm[0, a, :],
        #     all_frame_all_omm[0, a, :],
        # )

        plt.quiver(
            xy_return[:, 0],
            xy_return[:, 1],
            (comb_q_mult_x[b, :]),
            (comb_q_mult_y[b, :]),
        )

        # plt.quiver(
        #     xy_return[:, 0],
        #     xy_return[:, 1],
        #     q_coord_diff_x_no_mult[:, a],
        #     q_coord_diff_y_no_mult[:, a],
        # )

        plt.xlabel("azimuth")
        plt.ylabel("elevation")

        if direction == "left":
            plt.savefig(
                "LPTC_Output/LPTC_Figures/"
                + folder_name
                + direction
                + "/"
                + str(b)
                + "-frame.png"
            )

        elif direction == "right":
            # plt.show()
            plt.savefig(
                "LPTC_Output/LPTC_Figures/"
                + folder_name
                + direction
                + "/"
                + str(b)
                + "-frame.png"
            )
        # plt.show()
        plt.clf()


# Run function for all 58 neurons
for i in range(58):
    data = np.genfromtxt("LPTC_Data/LPTC_rf_" + str(i + 1) + ".csv", delimiter=",")
    lptc_func(np.array(data)[1:, :], "left", str(i + 1))
    print(i)


# data = np.genfromtxt("LPTC_Data/LPTC_rf_1.csv", delimiter=",")
# lptc_data = np.array(data)[1:, :]
# direction = "left"
# folder_name = "1"
