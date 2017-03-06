#!/usr/bin/python

"""
    Spectrogram class + cent scale transformation

    Author: Reinhard Sonnleitner (reinhard.sonnleitner@jku.at)

"""

import numpy as np
import sys
import Image
import ImageOps
sys.path.insert(0, "../util")
import wave


def cent2hz(centval):
    " cent to hz mapping "
    if centval==0: return 0
    return round(2**(float(centval) / 1200.0) * (440.0 * 2**((3.0/12.0)-5.0)), 4)

def mel2hz(mel):
    return 700.0 * (np.exp(mel/1125.0)-1.0)

def hz2mel(hz):
    return 1125.0 * np.log(1.0 + hz/700.0)

def frq2fftbin(hzval, fs, ws):
    #print "Hz: %.3f: hzval, bin: %.3f"%(hzval, (hzval / (float(fs) / float(ws))),)
    #return int(round(hzval / (float(fs) / float(ws))))
    return int(hzval / (float(fs) / float(ws)))


def triangle_filter(fidx, center_freqs_Hz):
    """
    Returns triangle filter for given filter index fidx, corresponding to the 
    list of center frequencies. 
    If the subsequent filterbank contains N triangle filters, 
    center_freqs_Hz has a length of N+2 (first value is the start freq, followed by N centerfrequencies, last
    value is the highest considered frequency).

    """
    hz_range = np.arange(center_freqs_Hz[-1]+1)
    trifilter = np.zeros(len(hz_range))
    for k in range(int(center_freqs_Hz[fidx]), int(center_freqs_Hz[fidx+1]) + 1):
        trifilter[k] = (k - center_freqs_Hz[fidx]) / (center_freqs_Hz[fidx+1] - center_freqs_Hz[fidx])
    for k in range(int(center_freqs_Hz[fidx+1]), int(center_freqs_Hz[fidx+2]) + 1):
        trifilter[k] = (center_freqs_Hz[fidx+2] -k) / (center_freqs_Hz[fidx+2] - center_freqs_Hz[fidx+1])
    return trifilter


def mel_centerfreqs(num_bands, fs, startfreq_hz=300):
    """
    Returns the mel scale center frequencies in mel.
    """
    startfreq_mel = hz2mel(startfreq_hz)
    stopfreq_mel = hz2mel(fs/2.0)
    center_freqs = [startfreq_mel + i * (stopfreq_mel - startfreq_mel) / float(num_bands) for i in range(num_bands)]
    center_freqs.append(stopfreq_mel)
    return center_freqs

def mel_centerfreqs_hz(num_bands, fs, startfreq_hz=300):
    return [mel2hz(i) for i in mel_centerfreqs(num_bands, fs, startfreq_hz)]


def triangle_filterbank(center_freqs_Hz):
    filters = []
    for i in range(len(center_freqs_Hz) - 2):
        filters.append(triangle_filter(i, center_freqs_Hz))
    return filters


def triangle_filterbank_freqbins(tri_fbank, fs, ws):
    """
    Map the triangle filterbank to frequency bins
    """
    binres = float(fs) / float(ws)
    bin_centerfreqs = np.array([int(round(binres/2.0+i*binres)) for i in range(ws/2)])
    mapped_filters = []
    for trifilter in tri_fbank:
        mapped_filters.append(np.array(trifilter)[bin_centerfreqs])
    return mapped_filters


def cent_filterbank(fs, ws, cent_start, cent_hop, num_linear_filters):
    " new cent filterbank "
    freqs = np.array([i*fs/ws for i in range(ws)])
    cent_borders = np.arange(cent_start, hz2cent(freqs[ws/2+1]), cent_hop)
    freqs_at_cent_borders = map(cent2hz, cent_borders)
    bins = np.array([frq2fftbin(frq, fs, ws) for frq in freqs_at_cent_borders])

    bin_indices = np.nonzero(np.diff(bins))[0]
    fftbin_borders = bins[bin_indices]
    if fftbin_borders[-1] < ws/2:
        fftbin_borders = np.append(fftbin_borders, ws/2)
    fftbin_borders = np.append(0, fftbin_borders)
    #for i in range(len(bins)):
    #    print "cent: %d, freq: %.2f, bin_idx: %d"%(cent_borders[i], freqs_at_cent_borders[i], bins[i])
    return fftbin_borders 
    

def bark_filterbank(fs, ws):
    cbw = [52548.0 / (z**2 - 52.56*z + 690.39) for z in np.arange(1,24)]
    borders = []
    off = 0
    for i in cbw:
        borders.append(off+i)
        off = off+i
    bins = np.array([frq2fftbin(frq, fs, ws) for frq in borders])
    bins = bins[bins < ws/2]
    if bins[-1] < ws/2-1:
        bins = np.append(bins, ws/2-1)
    return bins


def get_center_frequencies(fb):
    """
    Return center frequencies between filterbank bounds.
    """
    centerfreqs = []
    off = 0
    for i in range(1, len(fb)):
        centerfreqs.append(int(round((off+fb[i])/2.0)))
        off = fb[i]
    return centerfreqs


def get_triangle_filters(fb):
    """
    Computes triangle filters according to the filterbank bounds. 
    Triangles are overlapping from center-frequency of a band.
    """
    cf = get_center_frequencies(fb)
    triangle_borders = [(fb[0], fb[1])]
    for i in range(2,len(cf)-1):
        triangle_borders.append((cf[i-1], cf[i+1]))
    triangle_borders.append((fb[-2], fb[-1])) # last fb bin
    return triangle_borders


def compress(spec, borders):
    " Frequency compression by summing bins according to an array holding the corresponding indices. "
    cspec = np.zeros((len(borders), spec.shape[1]))
    for i in range(len(borders)-1):
        #cspec[i,:] = np.sum(spec[borders[i] : borders[i+1], :], axis=0) / float(borders[i+1]-borders[i])
        cspec[i,:] = np.sum(spec[borders[i] : borders[i+1], :], axis=0)
    return cspec


def map_trianglefilterbank_spectrogram(spec, tri_fbank):
    compressed = np.zeros((len(tri_fbank), spec.shape[1]))
    for i,f in enumerate(tri_fbank):
        bins = np.nonzero(f)[0]
        summed_bin = np.zeros((1, spec.shape[1]))
        for b in bins:
            summed_bin += spec[b,:] * f[b] 
        compressed[i,:] = summed_bin
    return compressed

def frameidx2time(frameidx, fs, ws, hs):
    " map STFT block index to absolute time within audio "
    return float(ws)/fs + ((frameidx-1)*hs)/float(fs)


def seconds2frames(sec, fs, ws, hs):
    return (float(sec) * fs - (ws-hs))/float(hs)


def hanningz(n):
    " HanningZ window "
    values = [0.5 * (1.0 - np.cos((2.0 * np.pi * float(i))/float(n))) for i in range(n)]
    return values


def spectrogram(signal, ws, hs):
    """
    Compute and return the complex spectrogram for a given signal.
    Winsize must be a power of two.
    To get the magnitude spectrogram just use abs(spec)
    """
    #window = np.array(hanningz(ws), dtype=np.float32)
    window = np.array(np.hanning(ws), dtype=np.float32)
    winhalf = ws / 2
    num_win = int(np.floor((len(signal) - ws) / hs))
    div_numwin_100 = 100.0/num_win
    print "Allocating memory for spectrogram: " + str(((winhalf+1) * (num_win+1) * np.dtype(np.complex).itemsize) / (1.0*1024*1024)) + "MB"
    spectrogram = np.zeros((winhalf+1, num_win+1), dtype=complex)
    signal = np.array(signal)
    rfft = np.fft.rfft
    for i in np.arange(num_win):
        if i%1000 == 0:
            sys.stdout.write("Magnitude spectrogram: %f%%\r" % (i*div_numwin_100, ))
            sys.stdout.flush()
        frame = rfft(signal[i * hs : i * hs + ws] * window)
        spectrogram[:,i] = frame
    print "Spectrogram: 100%                       "
    return spectrogram


def istft_frames(spec, ws, hs, prev_values=None):
    """
    prev values is of length winsize-hopsize.
    It consists of the last few sample values from the previous istft.
    Some of those values that are computed in this run must be added to 
    the parts of the previous values.
    .e.g: x[i*hs : i*hs + ws] += r * hwsdow
    when merging istfts, a new range cannot be written ontop of zero values, 
    but must be added to parts of previous results.
    """

    x = zeros(ws + hs * (spec.shape[1]-1))
    if prev_values is not None:
        x[0:len(prev_values)] = prev_values
    for i in range(spec.shape[1]):
        r = real(fft.irfft(spec[:, i]))
        x[i * hs : i * hs + ws] += r
    return x


def istft(spec, fs, ws, hs, t_sec):
    " compute ISTFT from spec, over duration of t_sec seconds " 
    t = t_sec
    x = zeros(floor(t * fs))
    for i in range((len(x)-ws)/hs):
        col = spec[:,i]
        r = real(fft.irfft(col))
        x[i * hs : i * hs + ws] += r
    return x


def scale_signal(signal, maxval=16384):
    signal_s = signal - signal.min()
    smax = signal_s.max()
    if smax == 0:
        smax = 0.000001
    signal_scaled = (maxval / smax * signal_s)
    return signal_scaled


def writewav(filename, signal, fs):
    " write .wav from signal "
    wavfile = wave.open(filename, 'wb')
    wavfile.setparams((1, 2, fs, len(signal), 'NONE', 'noncompressed'))
    signal_final = ''
    #scale
    signal_s = signal - signal.min()
    #signal_s = signal
    print signal.min()
    print signal.max()

    smax = signal_s.max()
    if smax == 0:
        smax = 0.000001
    signal_scaled = (32767 / smax) * signal_s
    #signal_scaled = 32767 * signal_s
    print signal_scaled.min()
    print signal_scaled.max()

    for i in range(len(signal)):
        signal_final += wave.struct.pack('h', signal_scaled[i])
    wavfile.writeframes(signal_final)
    wavfile.close()


def hz2cent(hzval):
    " hz to cent mapping "
    if hzval == 0: return 0
    return 1200.0 * np.log2(float(hzval) / (440.0 * 2**((3.0/12.0) - 5.0)))


def compress_spectrum_fast(spec, filterbank_bounds):
    """
    Compresses the frequency bands of a spectrum according to 
    an array of frequency bin indices.
    """
    compressed = np.zeros((len(filterbank_bounds), spec.shape[1]))

    for i in range(len(filterbank_bounds)-1):
        compressed[i, :] = np.sum(spec[filterbank_bounds[i, :] : filterbank_bounds[i + 1], :], axis=0)
    return compressed


def decompress_spectrum(spec, filterbank_bounds):
    """
    Approximate decompression of compressed spectrogramm.
    Decompresses according to filterbank_bounds.
    Returns decompressed spectrogram
    """
    #TODO: implement
    pass


def get_db_magnitudes(spec):
    """
    Computes spectrogram with db magnitudes from input spectrogram (linear magnitudes)
    """
    spectrogram = spec.copy()
    spectrogram[spectrogram<=0.000001] = 0.000001
    spectrogram = 20*log10(spectrogram)
    return spectrogram


def ndarray2image(data, filename):
    """ 
    writes content of float ndarray as .png file.
    normalizes content and writes uint8 picture
    """
    #scale image to uint8
    data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromstring("L", (data.shape[1], data.shape[0]), raw.tostring())
    im = ImageOps.autocontrast(im)
    im = ImageOps.invert(im)
    im.save(filename)




