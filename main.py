import pandas as pandas
import numpy as numpy
import scipy.fft
from numpy import nanmax
from pandas import read_csv
from matplotlib import pyplot
from scipy.fftpack import fft
import pywt
import pywt.data


# № 1
def print_mathematical_expected_value(ser):
    print("Mathematical expected value ", ser.mean())


# № 2
def print_dispersion(ser):
    print("Dispersion ", series.std())


# № 3
def print_scope(ser):
    ser_max = ser.max()
    ser_min = ser.min()
    print("Maximum ", ser_max)
    print("Minimum", ser_min)
    print("Scope", abs(ser_max - ser_min))


# № 4
def plot_autocorrelation_plot(ser):
    pandas.plotting.autocorrelation_plot(ser).plot()
    pyplot.show()


# № 5
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
    c_a, c_d = pywt.dwt(ser, "haar")
    pyplot.plot(c_a, label="approximation coefficients")
    pyplot.show()


# № 8
def plot_filtered_signal(ser):
    thresh = 0.6 * nanmax(ser)
    wavelet = "bior4.4"
    mode = "per"
    coeff = pywt.wavedec(ser, wavelet, level=5, mode=mode)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode=mode)
    pyplot.plot(reconstructed_signal)
    pyplot.show()


# № 9
def plot_recovered_signal(r_signal):
    pyplot.plot(r_signal)
    pyplot.show()


# № 10
def print_signal_diff(ser_old, ser_rec):
    print(ser_old.std())
    print(ser_rec[dataset_name].std(ddof=10))


def print_recovered_v2():
    t = scipy.fft.fft(recovered_series)
    pyplot.plot(t)
    pyplot.show()


def get_recovered_signal(ser):
    c_a, c_d = pywt.dwt(ser, "db2", "smooth")
    r_signal = pywt.idwt(c_a, c_d, 'db2', 'smooth')
    # return pandas.Series(pandas.DataFrame(r_signal).values[:, 1], name=dataset_name)
    r_df = pandas.DataFrame(r_signal)
    r_df.columns = ["dots", dataset_name]
    return r_df


def csv_check(ser):
    print(type(ser))
    print(ser.head())
    pyplot.plot(ser)
    pyplot.show()


if __name__ == '__main__':
    filename = "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0" \
               "/Data/S9/Final/IBI.csv"
    dataset_name = "Dataset"
    series = read_csv(filename, header=0, index_col=0, names=[dataset_name])
    recovered_series = get_recovered_signal(series)
    csv_check(series)
    print_mathematical_expected_value(series)
    print_dispersion(series)
    print_scope(series)
    plot_autocorrelation_plot(series)
    plot_fur_window(series)
    plot_wavelet_haar(series)
    plot_filtered_signal(series)
    plot_recovered_signal(recovered_series)
    print_signal_diff(series, recovered_series)
