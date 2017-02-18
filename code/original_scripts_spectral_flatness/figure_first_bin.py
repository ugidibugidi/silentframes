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
    data_path = "/home/user/Desktop/masterarbeit/data/" # where the mp3-files are
    filenames = [ 'RTL-h5', 'Sat1-h7' ]
    feature_name = "freqBin1_median"
    feature_ylabel = "First Frequency Bin"
    
    yMax = 0.007
    
    total_len_sec = 3600.0
    total_len_min = total_len_sec / 60.0

    
    plt.rcParams.update({'font.size': 14})
    
    f, axarr = plt.subplots(len(filenames), sharex=True)

    for file_num, filename in enumerate(filenames):
        ffb = np.loadtxt(data_path + feature_name + "/" + filename + "." + feature_name, delimiter=',')[ : ,1]
        annotation_file = data_path + "annotations_block/" + filename + ".label"
    
        axarr[file_num].plot(np.linspace(0, total_len_min, ffb.shape[0]), ffb,  color='blue')
        axarr[file_num].set_ylim(0, yMax)
        axarr[file_num].set_title("Hour: " + ntpath.basename(filename))
        axarr[file_num].set_ylabel(feature_ylabel)
    
    
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
    f.set_size_inches(13, 7)
    
    
    plt.savefig('/home/user/Desktop/' + feature_name + '.png', bbox_inches='tight')    
    plt.show()


run()
