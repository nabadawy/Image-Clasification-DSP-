#!/usr/bin/env python

"""
FIR Filter design functions
"""

__author__ = "Kasper Kiis Jensen"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "kkj@encida.dk"



import numpy as np
from scipy import signal


def FIRDesign(taps: int, cutoff: float, width: float, type='lowpass', fs=48000, window='Rectangle', plot=False):
    """
    Design FIR Filter based on taps and cutoff frequency
    :param taps: number of taps in filter
    :param cutoff: the cutoff frequency
    :param width: the width of the transition area
    :param type: the type of filter: lowpass | highpass
    :param fs: the sample rate
    :param window: the type of window: Rectangle | Boxcar | Triang | Blackman | Hamming | Hann | Bartlett | Flattop | Parzen | Bohman | Blackmanharris | Nuttall | Barthann
    :return:
    """
    window = window.lower()
    if window == 'rectangle':
        window = 'boxcar'

    if type == 'lowpass':
        b = signal.firwin(taps, cutoff, width=width, pass_zero=True, window=window, fs=fs)
    elif type == 'highpass':
        b = signal.firwin(taps, cutoff, width=width, pass_zero=False, window=window, fs=fs)

    if plot:
        w, h = signal.freqz(b, 1, worN=fs)
        plt.figure('Filter Frequency Response')
        plt.subplot(2, 1, 1)
        plt.title('Impulse response')
        plt.plot(b)
        plt.grid()
        plt.xlabel('Taps [.]')
        plt.ylabel('Amplitude [.]')
        #frequency response
        plt.subplot(2, 1, 2)
        plt.title('Frequency response')
        plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)))
        plt.grid()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')
        plt.show()

    return b


def kaiserDesign(fpass: float, fstop: float, gpass: float, gstop: float, fs=48000, plot=False):
    """
    Design a FIR filter based in kaiser design
    :param fpass: the passband frequency
    :param fstop: the stopband frequency
    :param gpass: maximum ripple passband
    :param gstop: minimum stopband attenuation
    :return: filter coefficients
    """
    wp = fpass * 2 / fs  # normalised passband frequency to nyquist frequency
    ws = fstop * 2 / fs  # normalized stopband frequency to nyquist frequency

    omp = wp * np.pi  # angular passband frequency
    oms = ws * np.pi  # angular stopband frequency

    # normalized cutoff
    omc = (omp + oms) / 2

    # Difference in angular frequency
    if omp > oms:
        deltac = omp - oms  # Highpass
    else:
        deltac = oms - omp  # Lowpass

    # Determine beta
    A = np.array([gpass, gstop]).max()

    # find beta of kaiser window
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif 21 <= A and A <= 50:
        beta = 0.5842 * ((A - 21) ** 0.4) + 0.07886 * (A - 21)
    elif A < 21:
        beta = 0

    # find order
    M = int(np.ceil((A - 8) / (2.285 * deltac)))
    if (M % 2) == 0: # is even
        pass
    else:
        M = M +1


    # Impulse reponse
    h = np.zeros(M + 1)
    if omp > oms:
        for n in range(M + 1):  # Highpass
            h[n] = np.sin(np.pi * (n - M / 2)) / (np.pi * (n - M / 2)) - np.sin(omc * (n - (M / 2))) / (np.pi * (n - (M / 2)))
    else:
        for n in range(M + 1):  # Lowpass
            h[n] = np.sin(omc * (n - (M / 2))) / (np.pi * (n - (M / 2)))

    # L hopspital rule for sin x / x
    if omp > oms:
        h[int((M) / 2)] = 1 - omc / np.pi  # Highpass
    else:
        h[int((M) / 2)] = omc / np.pi  # Lowpass

    # Apply window
    w = signal.kaiser(M + 1, beta)  # create kaiser window
    b = np.multiply(h, w)  # multiply window with impulse response


    if plot:
        w, h = signal.freqz(b, 1, worN=fs)
        plt.figure('Filter Response')
        plt.subplot(2,1,1)
        plt.title('Impulse response')
        plt.plot(b)
        plt.grid()
        plt.xlabel('Taps [.]')
        plt.ylabel('Amplitude [.]')
        plt.subplot(2,1,2)
        plt.title('Frequency response')
        # frequency response
        plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)))
        # passband
        plt.plot([(fpass - fpass / 2), fpass], [gpass, gpass], color='red')
        plt.plot([(fpass - fpass / 2), fpass, fpass], [-gpass, -gpass, -gpass - 20], color='red')
        # stopband
        plt.plot([fstop, fstop, fstop + fstop / 2], [-gstop + 20, -gstop, -gstop], color='red')
        plt.grid()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')
        plt.legend(['Filter response', 'Kaiser Design Specifications'])
        plt.show()

    return b


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000

    fcut = 4000
    taps = 54
    width = 2000

    b = FIRDesign(taps, fcut, width, 'lowpass', fs, 'Rectangle', True)
    w1, h1 = signal.freqz(b, 1, worN=fs)


    # kaiser designs based on specifications
    fpass = 3000
    fstop = 5000
    gpass = 1
    gstop = 40

    b = kaiserDesign(fpass, fstop, gpass, gstop, fs, True)
    w2, h2 = signal.freqz(b, 1, worN=fs)

    plt.figure('Comparison of Filter design Frequency Response')
    plt.semilogx((fs * 0.5 / np.pi) * w1, 20 * np.log10(abs(h1)))
    plt.semilogx((fs * 0.5 / np.pi) * w2, 20 * np.log10(abs(h2)))
    # passband
    plt.plot([(fpass - fpass / 2), fpass], [gpass, gpass], color='red')
    plt.plot([(fpass - fpass / 2), fpass, fpass], [-gpass, -gpass, -gpass - 20], color='red')
    # stopband
    plt.plot([fstop, fstop, fstop + fstop / 2], [-gstop + 20, -gstop, -gstop], color='red')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.legend(['FIR Design', 'Kaiser Design', 'Kaiser Design Specifications'])
    plt.show()

