"""
Script to get spectrogram from audio.

Takes any audio format, decodes audio to memory, computes magnitude spec, shows magnitude spec.


user@ubuntu:~/Desktop/masterarbeit/spectrogram_testcode/spectrogram$ python checker.py -start 2700 -stop 3599 /home/user/Desktop/masterarbeit/data/RTL-h20.mp3 22050 512 496

"""

import argparse
import sys
import numpy as np
import Spectrogram
sys.path.insert(0, "../audio_converter")
import audio_converter
import matplotlib.pyplot as plt
import os
import re
import math

def rms_energy(array):
    """
    Returns the root mean squared energy.
    """
    import numpy as np
    
    return np.sqrt(np.sum(np.power(array, 2), axis=0))


def arithmetic_mean(array, axis=0):
    """
    Returns the arithmetic mean over the given axis (default=0).
    """
    import numpy as np
    
    return np.mean(array, axis=axis)


def geometric_mean(array, axis=0):
    """
    Returns the geometric mean over the given axis (default=0).
    """
    from scipy import stats
    
    return stats.gmean(array, axis=axis)


def spectral_flatness(array, axis=0):
    """
    Returns the spectral flatness over the given axis (default=0).
    """
    
    return geometric_mean(array, axis=axis) / arithmetic_mean(array, axis=axis)



    
#from: http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)



def run():    
    fs = 22050 # sampling rate
    ws = 1024 # window size
    hs = 512 # hop size
    data_path = "/home/user/Desktop/masterarbeit/data/" # where the mp3-files are
    feature_names = ["sfCount", "sfConsecutive_Median", "sfConsecutive_Max"]
    silence_threshold = 1.0 # classified as silence if rms < silence_threshold

    clip_windowsize = 352 #704 #1408 #2816
    clip_stepsize = 9
    
    # add configuration to filename
    for i in range(len(feature_names)):
        feature_names[i] = feature_names[i]  + "_" + str(int(silence_threshold*10)) + "e1_" + str(clip_stepsize) + "_" + str(clip_windowsize)
        print feature_names[i] 
    
    # remove existing output directory and build directory path
    feature_descriptions = []
    for feature_name in feature_names:
        feature_output_dir = data_path + feature_name +"/"
        if not os.path.exists(feature_output_dir):
            os.makedirs(feature_output_dir)
        feature_descriptions.append([feature_name, feature_output_dir])
    # only silent frames (just as reference)
    feature_output_dir_sf = data_path + "sf/"
    if not os.path.exists(feature_output_dir_sf):
        os.makedirs(feature_output_dir_sf)
    
    # process hours in steps (otherwise too much memory is used!)
    stepsize = 900 # in seconds
    max_size = 3600 # in seconds (1h=3600s)
    steps = np.ceil(max_size*1.0/stepsize)
   
    tmp = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]# take only non-folders
    files = natural_sort(tmp)
    
    for file_num, f in enumerate(files):
        abs_f = data_path + f
        print  "############ " + f +  " ###### " + str(file_num+1) + "/" + str(len(files)) + " ##############################"
        
        # remove existing output file and build filename
        out_fs = []
        for feature_name, feature_output_dir in feature_descriptions:
            out_filename = feature_output_dir + f.replace(".mp3", "." + feature_name)
            if os.path.exists(out_filename):
                os.remove(out_filename)
            out_fs.append(open(out_filename , 'ab'))
        # only silent frames (just as reference)
        out_filename_sf = feature_output_dir_sf + f.replace(".mp3", ".sf") 
        if os.path.exists(out_filename_sf):
            os.remove(out_filename_sf)
        out_f_sf = open(out_filename_sf , 'ab')
        
        # convert file and make spectrogram
        for step in xrange(0, max_size, stepsize):
            signal = np.frombuffer(audio_converter.decode_to_memory(abs_f, sample_rate=fs, skip=step, maxlen=stepsize), dtype=np.float32)
            magspec = abs(Spectrogram.spectrogram(signal, ws=ws, hs=hs))
            print "magspec shape: %s"%(magspec.shape,)
        
            # calculate (frame-level) feature: silence frames
            rms = rms_energy(magspec)
            silent_frames = np.where(rms < silence_threshold, 1.0, 0.0)
            
            
            # only silent frames (just as reference)
            tmp_time = np.linspace(np.float(step), np.float(step+stepsize), num=silent_frames.shape[0], endpoint=False)
            for i in range(silent_frames.shape[0]):
                out_f_sf.write('{0:.8f}'.format(tmp_time[i]).rstrip('0').rstrip('.') + "," + repr(silent_frames[i]) + "\n")
            
            
            # clip
            clip_feature = np.empty((len(feature_names), np.ceil(silent_frames.shape[0]*1.0/clip_stepsize)))
            for i in range(clip_feature.shape[1]):
                start_frame = i*clip_stepsize - clip_windowsize/2
                # if clip_windowsize is odd then add one to the integer division
                stop_frame = i*clip_stepsize + clip_windowsize/2 + (1 if clip_windowsize%2 == 1 else 0)
                
                # set limits to prevent index out of bounds
                if start_frame < 0:
                    start_frame = 0
                if stop_frame > silent_frames.shape[0]:
                    stop_frame = silent_frames.shape[0]
                                
                # consecutive silent frames
                consecutive_sf = []
                consecutive_len = 0
                for j in range(start_frame, stop_frame):
                    if silent_frames[j] == 1:
                        consecutive_len += 1
                    else:
                        if consecutive_len != 0: # add only consecutive silent frames
                            consecutive_sf.append(consecutive_len)
                        consecutive_len = 0

                if not consecutive_sf: # if list is empty add dummy for easier calculations later
                    consecutive_sf.append(0)
                
                # calculate clip-level feature
                clip_feature[0][i] = silent_frames[start_frame:stop_frame].sum()
                clip_feature[1][i] = np.median(consecutive_sf)
                clip_feature[2][i] = np.max(consecutive_sf)

                #TODO: other features?
                
            # write feature to output
            tmp_time = np.linspace(np.float(step), np.float(step+stepsize), num=clip_feature.shape[1], endpoint=False)
            for i in range(clip_feature.shape[1]):
                for j, out_f in enumerate(out_fs):
                    out_f.write('{0:.8f}'.format(tmp_time[i]).rstrip('0').rstrip('.') + "," + repr(clip_feature[j][i]) + "\n")
        
        
        for out_f in out_fs:
            out_f.close()
        out_f_sf.close() # only silent frames (just as reference)
        
run()
