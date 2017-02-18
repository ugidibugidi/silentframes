"""
Script to get spectrogram from audio.

Takes any audio format, decodes audio to memory, computes magnitude spec, shows magnitude spec.
"""

import sys
import numpy as np
import Spectrogram
sys.path.insert(0, "../audio_converter")
import audio_converter
import matplotlib.pyplot as plt
import ntpath
from scipy import stats

def read_commercial_annotations(file, commercial_id='2;'):
    """
    Returns the a float-array which contains the start time and the duration of the commercial.
    (Non-commercials are discarded).
    """
    import numpy as np
    import csv
    
    annotations = np.empty( shape=(0, 0) )
    
    f = open(file, 'rb') # opens the csv file
    try:
        reader = csv.reader(f)
        comm = [c for c in reader if c[2] == commercial_id]
        if len(comm) > 0:
            annotations = (np.array(comm)[:,0:2]).astype(np.float)
    finally:
        f.close()
    
    return annotations


def run():
    fs = 22050 # sampling rate
    ws = 1024 # window size
    hs = 512 # hop size
    data_path = "/home/user/Desktop/masterarbeit/data/" # where the mp3-files are
    #filenames = [ 'RTL-h5', 'RTL-h17', 'Sat1-h16' ]
    filenames = [ 'RTL-h5', 'RTL-h17', 'ZDF-h17' ]
    
    total_len_sec = 3600.0
    total_len_min = total_len_sec / 60.0
        
    clip_size_seconds = 30.0
    yMax = 70.0
    
    plt.rcParams.update({'font.size': 14})
    
    f, axarr = plt.subplots(len(filenames), sharex=True)

    for file_num, filename in enumerate(filenames):
        loudness = np.loadtxt(data_path + "LoudnessTotal/" + filename + ".LoudnessTotal", delimiter=',')[ : ,1]
        annotation_file = data_path + "annotations_block/" + filename + ".label"


        # calculate mean
        clip_size = int(clip_size_seconds * (loudness.shape[0] / total_len_sec))
        print str(clip_size_seconds) + " seconds = " + str(clip_size) + " frames"
        loudness_shortend = loudness[:(loudness.shape[0] / clip_size) * clip_size] # remove elements at the end in order to get an shape that is a multiple of clip_size
        loudness_reshaped = loudness_shortend.reshape(-1, clip_size)
        loudness_clip = np.mean(loudness_reshaped, axis=1)
    
        axarr[file_num].plot(np.linspace(0, total_len_min, loudness_clip.shape[0]), loudness_clip, color='blue',  linewidth=2.0)
        axarr[file_num].plot(np.linspace(0, total_len_min, loudness.shape[0]), loudness,  color='blue', linewidth=0.2, alpha=0.10)
        axarr[file_num].set_ylim(0, yMax)
        axarr[file_num].set_title("Hour: " + ntpath.basename(filename))
        axarr[file_num].set_ylabel("Loudness")
        #axarr[file_num].plot([0, total_len_min], [np.mean(loudness), np.mean(loudness)], color='red')
        axarr[file_num].fill_between(np.linspace(0, total_len_min, loudness_clip.shape[0]), loudness_clip, np.mean(loudness))
    
    
        # read annotion file
        anno = read_commercial_annotations(annotation_file, commercial_id='2;')
        #print annotations
        for a in anno:
            a_start = a[0] * total_len_min / total_len_sec
            a_end= (a[0]+a[1]) * total_len_min / total_len_sec
            axarr[file_num].fill_between([a_start, a_end], 0, yMax, facecolor='gray', alpha=0.4)
    
    
    f.subplots_adjust(hspace=0.25)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlabel("Time [min]")
    f.set_size_inches(13, 11)
    
    
    plt.savefig('/home/user/Desktop/loudness_during_commercial_block.png', bbox_inches='tight')    
    plt.show()

    
    exit()
    
    
    
    
    
    
    
    if stop:
        signal = np.frombuffer(audio_converter.decode_to_memory(filename, sample_rate=fs, skip=start, maxlen=stop-start),\
                dtype=np.float32)
    else: 
        signal = np.frombuffer(audio_converter.decode_to_memory(filename, sample_rate=fs), dtype=np.float32)









    magspec = abs(Spectrogram.spectrogram(signal, ws=ws, hs=hs))
    print "magpsec shape: %s"%(magspec.shape,)
    print "signal shape: %s"%(signal.shape,)

    # save magspec to /tmp/
    np.savez('/tmp/magspec.npz', magspec)
    
    # downsample
    signal_downsampled = signal[::16384]
    print "signal_downsampled shape: %s"%(signal_downsampled.shape,)

    # set ticks, ticklabels in seconds
    length = magspec.shape[1]
    
    length_sec = Spectrogram.frameidx2time(length, ws=ws, hs=hs, fs=fs)
    tickdist_seconds = 60 # one tick every n seconds
    tickdist_labels_in_minutes = 60 # for seconds use 1; for minutes 60
    numticks = length_sec/tickdist_seconds
    tick_per_dist = int(round(length / numticks))
    xtickrange = range(length)[::tick_per_dist]
    xticklabels = ["%d"%(round(Spectrogram.frameidx2time(i, ws=ws, hs=hs, fs=fs))/tickdist_labels_in_minutes) for i in xtickrange]

    #plt.subplot(211)
    #plt.plot(signal_downsampled)
    #plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    #plt.title("signal")
    

    # energy
    clip_size = 512
    energy = np.sum(magspec, axis=0)
    print "energy shape: %s"%(energy.shape)
    energy_shortend = energy[:(energy.shape[0] / clip_size) * clip_size] # remove elements at the end in order to get an shape that is a multiple of clip_size
    print "energy_shortend shape: %s"%(energy_shortend.shape)
    energy_reshaped = energy_shortend.reshape(-1, clip_size)
    print "energy_reshaped shape: " + str(energy_reshaped.shape)
    energy_clip = np.mean(energy_reshaped, axis=1)
    print "energy_reshaped shape: " + str(energy_clip.shape)
    
    #TODO: use data of RTL-h8.LoudnessTotal

    #spectrogram_xscale = plt.xlim()  # just to scale it the same way the spectrograms were scaled
    plt.subplot(111)
    plt.plot(energy_clip)
    #plt.xlim(spectrogram_xscale)
    #plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=8)
    plt.title("energy")
    
    plt.suptitle("File: " + ntpath.basename(filename) + "  |  Time: " + str(round(start/60,1)) + " - " + str(round((start+length_sec)/60, 1)) + " [min]")
    plt.show()


run()
