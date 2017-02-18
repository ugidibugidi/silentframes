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
import os
import re
from scipy import stats

#from: http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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
    #filenames = [ 'RTL-h5', 'RTL-h17', 'Sat1-h16' ]
    tmp = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]# take only non-folders
    filenames = natural_sort(tmp)

    louder_than_mean_total = 1
    silent_than_mean_total = 1
    
    print "file \t\t louder\t silent\t ratio    \t commercial ratio"
    
    for file_num, filename in enumerate(filenames):
        filename = filename.replace(".mp3", "")
        time_loudness = np.loadtxt(data_path + "LoudnessTotal/" + filename + ".LoudnessTotal", delimiter=',')
        annotation_file = data_path + "annotations_block/" + filename + ".label"
        anno = read_commercial_annotations(annotation_file, commercial_id='2;')
        
        loudness = time_loudness[ : ,1]
        loudnessTime = time_loudness[ : ,0]
        
        mean_hour = np.mean(loudness)
        
        loudness_only_commercials_crit = np.zeros((loudness.shape[0]), dtype=bool)
        for start_time, duration in anno:
            end_time = start_time + duration
            criterion = (time_loudness[ : ,0] >= start_time) & (time_loudness[ : ,0] <= end_time)
            loudness_only_commercials_crit = np.logical_or(loudness_only_commercials_crit, criterion)
        
        loudness_only_commercials = time_loudness[loudness_only_commercials_crit]

        louder_than_mean = np.where(loudness_only_commercials[ : ,1] > mean_hour)[0].size
        silent_than_mean = loudness_only_commercials.shape[0] - louder_than_mean
        
        print ntpath.basename(filename) + "  \t " + str(louder_than_mean) + \
                " \t " + str(silent_than_mean) + " \t " + \
                "{0:.2f}".format((louder_than_mean*1.0+1.0)/ (louder_than_mean+silent_than_mean+1.0)*100.0) + \
                "%   \t " + \
                "{0:.2f}".format(loudness_only_commercials.shape[0] *100.0 / loudness.shape[0]) + "%"
        
        louder_than_mean_total += louder_than_mean
        silent_than_mean_total += silent_than_mean

    print "-------------------------------------------------------------------------------"
    print "TOTAL    \t " + str(louder_than_mean_total) +  \
            " \t " + str(silent_than_mean_total) + " \t " +  \
            "{0:.2f}".format(louder_than_mean_total*1.0/(louder_than_mean_total+silent_than_mean_total)*100.0) + "%"



run()
