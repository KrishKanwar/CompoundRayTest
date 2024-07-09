import numpy as np
from cmath import pi
import matplotlib.pyplot as plt


def low_pass_filter(t, u, tau):

    # plot the time series
    plt.plot(t, u)

    # plt.show()

    # low pass filter
    from scipy.signal import butter, lfilter

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyq  # cutoff frequency
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)  # get the filter coefficients
        y = lfilter(b, a, data)  # apply the filter
        return u

    # filter the time series
    fs = 1000  # sampling frequency
    cutoff = 400  # cutoff frequency
    y = butter_lowpass_filter(u, cutoff, fs, order=2)

    # plot the filtered time series with original in different colors
    plt.plot(t, u, "b-", label="data")
    plt.plot(t, y, "g-", linewidth=2, label="filtered data")

    # low pass filter
    T = 0.05 / 23  # sampling period
    tau_HR = tau  # time constant of the low pass filter
    A = 1 - 2 * tau_HR / T
    B = 1 + 2 * tau_HR / T

    yf = np.zeros(t.shape)  # initialize the filtered time series
    for i in range(1, len(t)):
        yf[i] = (u[i] + u[i - 1] - A * yf[i - 1]) / B  # apply the filter

    # plot the filtered time series
    plt.plot(t, u, "b-", label="data")
    plt.plot(t, yf, "g-", linewidth=2, label="filtered data")

    # plt.show()
    return yf


# # a time series
# t = np.linspace(0, 1000, 1000)

# # # a sine wave with noise
# u = np.sin(2*pi*t/100)

# low_pass_filter(t, u)
