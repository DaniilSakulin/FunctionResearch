import pandas as pandas
import numpy as numpy
from pandas import read_csv
from matplotlib import pyplot
from scipy.fftpack import fft
from scipy import signal
import pywt
import pywt.data


# №1
def print_mathematical_expected_value(ser):
    print("Mathematical expected value ", ser.mean())


# №2
def print_dispersion(ser):
    print("Dispersion ", series.std())


# №3
def print_scope(ser):
    ser_max = ser.max()
    ser_min = ser.min()
    print("Maximum ", ser_max)
    print("Minimum", ser_min)
    print("Scope", abs(ser_max - ser_min))


# №4
def plot_autocorrelation_plot(ser):
    pandas.plotting.autocorrelation_plot(ser).plot()
    pyplot.show()


# №5
def plot_fur_window(ser):
    X = fft(ser)
    N = len(X)
    n = numpy.arange(N)
    # get the sampling rate
    sr = 1 / (60 * 60)
    T = N / sr
    freq = n / T

    # Get the one-sided specturm
    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    # pyplot.figure(figsize=(60, 12))
    pyplot.plot(f_oneside, numpy.abs(X[:n_oneside]), 'b')
    pyplot.show()


# № 6
def plot_wavelet_haar(ser):
    cA, cD = pywt.dwt(ser, "haar")
    pyplot.plot(cA, label="approximation coefficients")
    pyplot.show()


def csv_check(ser):
    print(type(ser))
    print(ser.head())
    pyplot.plot(ser)
    pyplot.show()


if __name__ == '__main__':
    filename = "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0" \
               "/Data/S9/Final/IBI.csv"
    series = read_csv(filename, header=0, index_col=0)
    # csv_check(series)
    # print_mathematical_expected_value(series)
    # print_dispersion(series)
    # print_scope(series)
    # plot_autocorrelation_plot(series)
    # plot_fur_window(series)
    # plot_wavelet_haar(series)
