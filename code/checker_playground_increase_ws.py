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
    
    
def extract_local_maxima(list, order=3):
    """
    Returns a new list of the same size where all not local maxima are 0.0.
    """
    
    from scipy.signal import argrelextrema

    pos_maxima = argrelextrema(list, np.greater, order=order)[0]
    local_maxima = np.zeros(list.shape)
    
    for pos in pos_maxima:
        local_maxima[pos] = list[pos]
    
    return local_maxima


#from: http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def filter_below_threshold(list, cutoff_threshold, fixed_pass_threshold, variable_pass_th_variance=1.1, top_n_largest=50):
    """
    Returns the a new list which sets all the entries that are smaller than the thresholds to 0.0.
    """
    import numpy as np
    import heapq
    
    list_copy = np.copy(list)
    variable_pass_threshold = heapq.nlargest(top_n_largest, list_copy)[-1] / variable_pass_th_variance
    condition = (list_copy < cutoff_threshold) | ((list_copy < fixed_pass_threshold) & (list_copy < variable_pass_threshold))
    list_copy[np.where(condition)[0]] = 0.0
    
    return variable_pass_threshold, list_copy
    
    
def filter_above_threshold(list, cutoff_threshold, fixed_pass_threshold, variable_pass_th_variance=1.1, top_n_smallest=50):
    """
    Returns the a new list which sets all the entries that are larger than the thresholds to 0.0.
    """
    import numpy as np
    import heapq
    
    list_copy = np.copy(list)
    variable_pass_threshold = heapq.nsmallest(top_n_smallest, list_copy)[-1] * variable_pass_th_variance
    condition = (list_copy > cutoff_threshold) | ((list_copy > fixed_pass_threshold) & (list_copy > variable_pass_threshold))
    list_copy[np.where(condition)[0]] = 0.0
    
    return variable_pass_threshold, list_copy
    
    
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
    

def print_annotations(src_plot, annotations, len_per_hour, max_size, from_len, to_len, y_axis_max_size):
    """
    Print the annotations to a plot.
    """

    for a in annotations:
        a_start = a[0]*len_per_hour/max_size
        a_end= (a[0]+a[1])*len_per_hour/max_size
        if (from_len < a_start) and (a_start < to_len): # if the hour is splitted
            a_start -= from_len
            a_end -= from_len
            src_plot.fill_between([a_start, a_end], 0, y_axis_max_size, facecolor='gray', alpha=0.3)
    

def check_results_start_points(prediction, true_class, decision_rate=0.2):
    """
    Compute True-Positive, False-Positive, True-Negative and False-Negative of the prediction and true_class.
    The decision rate is 
    """
    max_size = 3600.0 # in s
    len_per_hour = 38757.0
    time_diff = 0.3 # in s
    
    # convert prediction to other format
    prediction_converted = np.array([idx for idx, p in enumerate(prediction) if p > 0.0], dtype=np.float)
    prediction_converted *= max_size/len_per_hour
    
    prediction_cleared = [] # create a list for easier appending
    actual = prediction_converted[0]
    prediction_cleared.append(actual)
    for val in prediction_converted:
        if (val - actual) > time_diff:
            actual = val
            prediction_cleared.append(actual)

    # convert true_class to other format
    true_class_cleared = [] # create a list for easier appending
    for start,duration in true_class:
        true_class_cleared.append(start)
        true_class_cleared.append(start+duration)
    true_class_cleared = set(true_class_cleared) # remove duplicates
    
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # calculate TP and FP
    for pred in prediction_cleared:
        
        found = 0
        for tru in true_class_cleared:
            if np.absolute(pred - tru) < decision_rate:
                found = 1
                break
        
        if found == 1:
            tp += 1
        else:
            fp += 1
            
    # calculate FN
    for tru in true_class_cleared:
        
        found = 0
        for pred in prediction_cleared:
            if np.absolute(pred - tru) < decision_rate:
                found = 1
                break
        
        if found == 0:
            fn += 1
    
    # calculate TN = all time slices - TP - FP - FN
    # --> all time slices = 60 * 60 / decision_rate
    tn = int((60 * 60 / decision_rate) - tp - fp - fn)
    
    return tp, fp, tn, fn


def masked_area_between_points(list, min_distance, max_distance, set_to_val):
    """
    Sets the area between values greater than 0.0 to val if they are at least min_distance
    away from each other and at most max_distance.
    """
    max_size = 3600.0 # in s
    len_per_hour = 38757.0
    time_diff = 0.3 # in s
    frame_to_time = max_size/len_per_hour
    time_to_frame = len_per_hour/max_size
    
    list1 = np.zeros(list.shape)
    threshold = 0.0
    
    idx_of_positives = np.array([idx for idx, p in enumerate(list) if p > threshold])
    
    for i1 in idx_of_positives:
        t1 = i1 * frame_to_time
        for i2 in idx_of_positives:
            t2 = i2 * frame_to_time
            diff = t2 - t1
            if min_distance < diff and diff < max_distance: # implicitly ignore all cases where t2 < t1
                for j in range(i1, i2): # mask area between
                    list1[j] = set_to_val               

    return list1


def check_results_mask(prediction, true_class, decision_rate=0.2):
    """
    Compute True-Positive, False-Positive, True-Negative and False-Negative of the prediction and true_class.
    The decision rate is 
    """
    max_size = 3600.0 # in s
    len_per_hour = 38757.0
    time_diff = 0.3 # in s
    frame_to_time = max_size/len_per_hour
    time_to_frame = len_per_hour/max_size
    
    # convert true_class to other format
    true_class_expanded = np.zeros(prediction.shape)
    for start,duration in true_class:
        start_frame = int(np.floor(start * time_to_frame))
        end_frame = int(np.floor((start+duration) * time_to_frame))
        for i in range(start_frame, end_frame):
            true_class_expanded[i-1] = 1.0
    
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    delta = int(np.floor(decision_rate * time_to_frame))
    prediction_length = prediction.shape[0]
    
    # calculate TP and FP
    for i,pred in enumerate(prediction):
        if pred > 0.0:
            found = 0
            for j in range(max(0, i-delta), min(i+delta,prediction_length)):
                if true_class_expanded[j] > 0.0:
                    found = 1
                    break
        
            if found == 1:
                tp += 1
            else:
                fp += 1

    # calculate FN
    for i,tru in enumerate(true_class_expanded):
        if tru > 0.0:
            found = 0
            for j in range(max(0, i-delta), min(i+delta,prediction_length)):
                if prediction[j] > 0.0:
                    found = 1
                    break
        
            if found == 0:
                fn += 1
    
    # calculate TN = all time slices - TP - FP - FN
    tn = prediction_length - tp - fp - fn
    
    return tp, fp, tn, fn
    
    
#from: http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def gaussian(x, mu, sig):
    gaus = np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))
    gaus *= 2.0 # to flatten it at the top
    gaus += 0.05
    if np.isscalar(x):
        if gaus > 1.0:
            gaus = 1.0
    else:
        gaus[gaus > 1.0] = 1.0
    return gaus
    """
    for mu, sig in [(14, 0.15), (17.0, 1.0)]:
        x = np.linspace(0.0, 24.0, 1000)
        y = gaussian(x, mu, sig)
        
        plt.grid(True)
        plt.ylim(ymax=1.01)
        plt.xlim(xmax=24)
        plt.plot(x, y, linewidth=6)
        plt.yticks(np.arange(0.0, 1.01, 0.1))
        plt.xticks(np.arange(min(x), max(x)+1, 2))
        plt.xlabel("Time of the day")
    plt.show()
    return
    """
    
def get_threshold(channel, daytime):
    #daytime in full hours (0-23)
    cutoff_threshold = 0.25
    fixed_pass_threshold = 0.4
    variable_pass_th_variance = 1.2
    top_n_largest = 30
    top_n_min = 1
    
    #uncomment for heuristic threshold
    '''
    distribution = np.array([ 14, 0.15 ])
    if channel == 'ARD': distribution = np.array([ 17.0, 1.0 ])
    elif channel == 'RTL': distribution = np.array([ 14, 0.15 ])
    elif channel == 'Sat1': distribution = np.array([ 14, 0.15 ])
    elif channel == 'ZDF': distribution = np.array([ 17.0, 1.0 ])
    
    factor = gaussian(daytime, distribution[0], distribution[1])
    top_n_largest = top_n_min + int(top_n_largest*factor)
    '''
    
    return cutoff_threshold, fixed_pass_threshold, variable_pass_th_variance, top_n_largest


def run():    
    fs = 22050
    ws = 4096
    hs = 2048
    data_path = "/home/user/Desktop/masterarbeit/data/" # where the mp3-files are
    annotation_path = data_path + "annotations/"
    stepsize = 3600 # in seconds
    max_size = 3600 # in seconds (1h=3600s)
    y_axis_max_size = 0.8001 # maximal value of y axis (to get a uniform scale over all plots)
    y_axis_min_size = 0.0 # minimal value of y axis (to get a uniform scale over all plots)
    x_axis_max_size = 38757 # maximal value of x axis (to get a uniform scale over all plots)
    x_axis_min_size = 0 # minimal value of x axis (to get a uniform scale over all plots)
    len_per_hour = 38757 #fixed for plotting regions
    
    show_figures = 0

    steps = np.ceil(max_size/stepsize)
    
    summary = [] 

    tmp = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]# take only non-folders
    files = natural_sort(tmp)
    
    for file_num, f in enumerate(files):
        abs_f = data_path + f
        print  "############ " + f +  " ###### " + str(file_num+1) + "/" + str(len(files)) + " ############"
        
        # read annotion file
        annotation_file = annotation_path + os.path.splitext(f)[0] + ".label"
        anno = read_commercial_annotations(annotation_file, commercial_id='2;')
        
        #make one figure per file
        if show_figures:
            plt.ion() # turns on interactive mode 
            plt.figure() # create a new figure

        for (j, step) in enumerate(xrange(0, max_size, stepsize)):
            #print  "###### from " + str(step) + " to " + str(step+stepsize) + " ###"
            
            # convert file and make spectrogram
            signal = np.frombuffer(audio_converter.decode_to_memory(abs_f, sample_rate=fs, skip=step, maxlen=stepsize), dtype=np.float32)
            magspec = abs(Spectrogram.spectrogram(signal, ws=ws, hs=hs))
            print "magpsec shape: %s"%(magspec.shape,)
            magspec_without_last = magspec[:,0:magspec.shape[1]-1] # remove the last entry because it always contains 0

            # arithmetic mean
            #amean = arithmetic_mean(magspec_without_last, axis=0)
            #(tmp,amean_masked) = filter_above_threshold(amean, 0.001, 0.0001, variable_pass_th_variance=1.1, top_n_smallest=50)

            # spectral flatness
            sflatness = spectral_flatness(magspec_without_last, axis=0)
            
            # extract local maxima
            # --> maybe smooth it before numpy.convolve() ... http://wiki.scipy.org/Cookbook/SignalSmooth
            #sflatness_smoothed = smooth(sflatness, window_len=11)
            #sflatness_local_maxima = extract_local_maxima(sflatness_smoothed, order=2)
             
            print "1"
                       
            # maske-out all non-canidates
            channel =  f.split('-')[0] 
            daytime = int(((f.split('-')[1]) .split('h')[1]).split('.')[0])
            cutoff_th, fixed_pass_th, variable_pass_th_var, top_n_largest = get_threshold(channel, daytime)
            (variable_pass_th, sflatness_masked) = filter_below_threshold(sflatness, #sflatness_local_maxima, 
                        cutoff_th, fixed_pass_th, variable_pass_th_variance=variable_pass_th_var, top_n_largest=top_n_largest)

            # set area between candidates to 1.0
            sflatness_area = masked_area_between_points(sflatness_masked, 3.0, 90.0, 0.6)
            
            # calculate statistics
            #tp, fp, tn, fn = check_results_start_points(sflatness_masked, anno, decision_rate=0.2)
            #baseline = np.zeros(sflatness.shape)
            #tp, fp, tn, fn = check_results_mask(baseline, anno, decision_rate=0.2)
            tp, fp, tn, fn = check_results_mask(sflatness_area, anno, decision_rate=0.2)	    
            summary.append( (os.path.splitext(f)[0], tp, fp, tn, fn) )
            
            print "2"
            
            # NEW
            #set area between  to 1.0
            sflatness_area = masked_area_between_points(sflatness_masked, 3.0, 90.0, 1.0)

            tmp_time = np.linspace(0.0, np.float(3600.0), num=sflatness_area.shape[0], endpoint=False)

            out_f = open('/home/user/Desktop/masterarbeit/data/betweenSilentFrames/' + f.replace(".mp3", ".betweenSilentFrames") , 'wb')
            for i in range(sflatness_area.shape[0]):
                out_f.write('{0:.8f}'.format(tmp_time[i]).rstrip('0').rstrip('.') + "," + repr(sflatness_area[i]) + "\n")
            out_f.close()
            # NEW
            
            print "3"
            
            
            # plot data
            if show_figures:
                # set ticks, ticklabels in seconds
                length = magspec.shape[1]
                length_sec = Spectrogram.frameidx2time(length, ws=ws, hs=hs, fs=fs)
                tickdist_seconds = 120 # one tick every n seconds
                tickdist_labels_in_minutes = 60 # for seconds use 1; for minutes 60
                numticks = length_sec/tickdist_seconds
                tick_per_dist = int(round(length / numticks))
                xtickrange = range(length)[::tick_per_dist]
                xticklabels = ["%d"%((round(Spectrogram.frameidx2time(i, ws=ws, hs=hs, fs=fs)+j*stepsize)/tickdist_labels_in_minutes)) for i in xtickrange]
                
                #first subplot
                plt.subplot(3,steps,j+1)
                plt.plot(sflatness, alpha=0.8, linewidth=1)
                plt.axhline(y=cutoff_th, linewidth=2.5, color='g')
                plt.axhline(y=fixed_pass_th, linewidth=2.5, color='r')
                plt.axhline(y=variable_pass_th, linewidth=2.5, color='k')
                print_annotations(plt, anno, len_per_hour, max_size, j*len(sflatness), (j+1)*len(sflatness), y_axis_max_size)
                plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=11)
                #plt.title("Spectral Flatness  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                plt.ylim(ymin=y_axis_min_size, ymax=y_axis_max_size) # set constant y scale
                plt.xlim(xmin=x_axis_min_size, xmax=x_axis_max_size)
                plt.yticks(np.arange(0.0, y_axis_max_size, 0.1))
                
                #second subplot
                plt.subplot(3,steps,steps+j+1)
                plt.plot(sflatness_masked, alpha=0.7, linewidth=2.5)
                print_annotations(plt, anno, len_per_hour, max_size, j*len(sflatness), (j+1)*len(sflatness), y_axis_max_size)	    
                plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=11)
                #plt.title("Spectral Flatness (selected)  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                plt.ylim(ymin=y_axis_min_size, ymax=y_axis_max_size) # set constant y scale
                plt.xlim(xmin=x_axis_min_size, xmax=x_axis_max_size)
                plt.yticks(np.arange(0.0, y_axis_max_size, 0.1))

               #third subplot PROTOTYP only for whole hours
                plt.subplot(3,steps,3)
                plt.plot(sflatness_area, alpha=0.7, linewidth=2.5)
                print_annotations(plt, anno, len_per_hour, max_size, j*len(sflatness), (j+1)*len(sflatness), y_axis_max_size)	    
                plt.xticks(xtickrange, xticklabels, rotation=70, fontsize=11)
                plt.fill_between(range(sflatness_area.shape[0]), 0, sflatness_area, facecolor='blue', alpha=0.3)
                #plt.title("Connect area between  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                plt.ylim(ymin=y_axis_min_size, ymax=y_axis_max_size) # set constant y scale
                plt.xlim(xmin=x_axis_min_size, xmax=x_axis_max_size)
                plt.yticks(np.arange(0.0, y_axis_max_size, 0.1))


        if show_figures:
            plt.suptitle("File: " + os.path.splitext(f)[0])
            plt.show()
        

    # calculating statistics
    summary_names = (np.array(summary)[:,:1])
    summary_values = (np.array(summary)[:,1:]).astype(np.integer)
    
    total_tp = summary_values.sum(0)[0]
    total_fp = summary_values.sum(0)[1]
    total_tn = summary_values.sum(0)[2]
    total_fn = summary_values.sum(0)[3]

    summary_names = np.vstack((summary_names, np.array([ 'total' ])))
    summary_values = np.vstack((summary_values, np.array([ total_tp, total_fp, total_tn, total_fn ])))
    
    print "NAME      \t TP \t FP \t TN \t FN \t PREC \t RECL \t F1 \t ACC "
    for idx, entry in enumerate(summary_values):
        tp = entry[0]
        fp = entry[1]
        tn = entry[2]
        fn = entry[3]
        prec = tp / (tp+fp+0.0001)
        recall = tp / (tp+fn+0.0001)
        fmeasure = 2.0 * (prec * recall) / (prec + recall+0.0001)
        accuracy = (tp+tn) / (tp+tn+fp+fn+0.0001)
        print ( str(summary_names[idx]) + " \t " + str(tp) + " \t " + str(fp) + " \t " + 
                    str(tn) + " \t " + str(fn) + " \t " + str(round(prec,3)) + " \t " + 
                    str(round(recall,3)) + " \t " + str(round(fmeasure,3)) + " \t " + str(round(accuracy,3)) )

    if show_figures:
        raw_input('Press Enter to continue...')

run()
