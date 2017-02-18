"""
Script to get spectrogram from audio.

Takes any audio format, decodes audio to memory, computes magnitude spec, shows magnitude spec.
"""

import argparse
import sys
import numpy as np
import Spectrogram
sys.path.insert(0, "../audio_converter")
import audio_converter
import matplotlib.pyplot as plt
import ntpath
from scipy import stats

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', action="store", help='The audio input file.')
    parser.add_argument('fs', action="store", type=int, help='Sampling rate.')
    parser.add_argument('ws', action="store", type=int, help='STFT windowsize.')
    parser.add_argument('hs', action="store", type=int, help='STFT hopsize.')

    parser.add_argument('-start', action="store", type=int, help="Start second of audio.")
    parser.add_argument('-stop', action="store", type=int, help="Stop second of audio.")
    return parser.parse_args()


def run():
    args = parse_arguments()
    if not args.start: args.start = 0
    if args.stop:
        signal = np.frombuffer(audio_converter.decode_to_memory(args.filename, sample_rate=args.fs, skip=args.start, maxlen=args.stop-args.start),\
                dtype=np.float32)
    else: 
        signal = np.frombuffer(audio_converter.decode_to_memory(args.filename, sample_rate=args.fs), dtype=np.float32)

    magspec = abs(Spectrogram.spectrogram(signal, ws=args.ws, hs=args.hs))
    print "magpsec shape: %s"%(magspec.shape,)

    # save magspec to /tmp/
    np.savez('/tmp/magspec.npz', magspec)

    # set ticks, ticklabels in seconds
    length = magspec.shape[1]
    length_sec = Spectrogram.frameidx2time(length, ws=args.ws, hs=args.hs, fs=args.fs)
    tickdist_seconds = 60 # one tick every n seconds
    tickdist_labels_in_minutes = 60 # for seconds use 1; for minutes 60
    numticks = length_sec/tickdist_seconds
    tick_per_dist = int(round(length / numticks))
    xtickrange = range(length)[::tick_per_dist]
    xticklabels = ["%d"%(round(Spectrogram.frameidx2time(i, ws=args.ws, hs=args.hs, fs=args.fs))/tickdist_labels_in_minutes) for i in xtickrange]

    plt.subplot(611)
    plt.imshow(magspec, aspect='auto', origin='lower', interpolation='nearest')
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("magnitude spectrogram")

    plt.subplot(612)
    plt.imshow(np.log(magspec+1), aspect='auto', origin='lower', interpolation='nearest')
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("log magnitude spectrogram")
    

    # spectral flatness
    magspec_without_last = magspec[:,0:magspec.shape[1]-1] # remove the last entry because it always contains 0
    print "calculating geomethric mean"
    gmean = stats.gmean(magspec_without_last, axis=0)
    print "calculating arithmetic mean"
    amean = np.mean(magspec_without_last, axis=0)
    spectral_flatness = gmean / amean

    spectrogram_xscale = plt.xlim()  # just to scale it the same way the spectrograms were scaled
    plt.subplot(613)
    plt.plot(spectral_flatness)
    plt.xlim(spectrogram_xscale)
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("spectral flatness")

    idx = np.where(spectral_flatness < 0.3)[0]
    spectral_flatness[idx] = 0.0
    plt.subplot(614)
    plt.plot(spectral_flatness)
    plt.xlim(spectrogram_xscale)
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("spectral flatness > 0.3")

    plt.subplot(615)
    plt.plot(amean)
    plt.xlim(spectrogram_xscale)
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("arithmetic mean")

    idx = np.where(amean > 0.001)[0]
    amean[idx] = 1.0
    plt.subplot(616)
    plt.plot(amean)
    plt.xlim(spectrogram_xscale)
    plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("arithmetic mean < 0.001")

    
    plt.suptitle("File: " + ntpath.basename(args.filename) + "  |  Time: " + str(round(args.start/60,1)) + " - " + str(round((args.start+length_sec)/60, 1)) + " [min]")
    plt.show()


run()
