"""
Script to get spectrogram from audio.

Takes any audio format, decodes audio to memory, computes magnitude spec, shows magnitude spec.


user@ubuntu:~/Desktop/masterarbeit/spectrogram_testcode/spectrogram$ python checker.py -start 2700 -stop 3599 /home/user/Desktop/masterarbeit/data/RTL-h20.mp3 22050 512 496

"""

import argparse
import sys
import numpy as np
import Spectrogram
sys.path.insert(0, "audio_converter")
import audio_converter
import matplotlib.pyplot as plt
import os
import re
import math


def rms_energy(array, axis=0):
    """
    Returns the root mean squared energy over the given axis (default=0).
    """
    import numpy as np
    
    return np.sqrt(np.sum(np.power(array, 2), axis=axis))
    

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
    
def reset_value(list, old_value, new_value):
    """
    Returns the a new list which sets all the entries that are equal to old_value to the new_value.
    """
    import numpy as np
    
    list_copy = np.copy(list)
    condition = list_copy == old_value
    list_copy[np.where(condition)[0]] = new_value
    
    return list_copy
    
    
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
    

def print_annotations(src_plot, annotations, elements_per_hour, max_size, from_len, to_len, y_axis_max_size):
    """
    Print the annotations to a plot.
    """

    for a in annotations:
        a_start = a[0]*elements_per_hour/max_size
        a_end= (a[0]+a[1])*elements_per_hour/max_size
        if (from_len < a_start) and (a_start < to_len): # if the hour is splitted
            a_start -= from_len
            a_end -= from_len
            src_plot.fill_between([a_start, a_end], 0, y_axis_max_size, facecolor='gray', alpha=0.3)
    

def check_results_start_points(elements_per_hour, prediction, true_class, decision_rate=0.2):
    """
    Compute True-Positive, False-Positive, True-Negative and False-Negative of the prediction and true_class.
    The decision rate is 
    """
    max_size = 3600.0 # in s
    time_diff = 0.3 # in s
    
    # convert prediction to other format
    prediction_converted = np.array([idx for idx, p in enumerate(prediction) if p > 0.0], dtype=np.float)
    prediction_converted *= max_size/float(elements_per_hour)
    
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


def masked_area_between_points(elements_per_hour, list, min_distance, max_distance, set_to_val):
    """
    Sets the area between values greater than 0.0 to val if they are at least min_distance
    away from each other and at most max_distance.
    """
    max_size = 3600.0 # in s
    frame_to_time = max_size/float(elements_per_hour)
    time_to_frame = float(elements_per_hour)/max_size
    
    list1 = np.zeros(list.shape)
    threshold = 0.0
    
    idx_of_positives = np.array([idx for idx, p in enumerate(list) if p > threshold])

    for i,idx_of_positive1 in enumerate(idx_of_positives):
        t1 = idx_of_positive1 * frame_to_time
        
        # Start at the search at i (commercials are longer than 0 seconds)
        for idx_of_positive2 in idx_of_positives[i:]:
            t2 = idx_of_positive2 * frame_to_time
            diff = t2 - t1
            
            # Stop search when the max distance has been reached
            if diff > max_distance:
                break
            
            # Set area between min and max distance to the input value
            if min_distance < diff and diff < max_distance:
                list1[idx_of_positive1:idx_of_positive2] = set_to_val

    return list1


def check_results_mask(elements_per_hour, prediction, true_class, decision_rate=0.2):
    """
    Compute True-Positive, False-Positive, True-Negative and False-Negative of the prediction and true_class.
    The decision rate is 
    """
    max_size = 3600.0 # in s
    frame_to_time = max_size/float(elements_per_hour)
    time_to_frame = float(elements_per_hour)/max_size
    
    # convert true_class to other format
    true_class_expanded = np.zeros(prediction.shape)
    for start,duration in true_class:
        start_frame = int(np.floor(start * time_to_frame))
        end_frame = min(int(np.floor((start+duration) * time_to_frame)), true_class_expanded.shape[0])
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
    cutoff_threshold = 0.002
    fixed_pass_threshold = 0.0004
    variable_pass_th_variance = 1.6
    top_n_largest = 40
    top_n_min = 1
    
    #uncomment for spectral flatness threshold
    '''
    cutoff_threshold = 0.3
    fixed_pass_threshold = 0.6
    variable_pass_th_variance = 1.1
    top_n_largest = 30
    top_n_min = 1
    '''
    
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
    ws = 4096 #512
    hs = 2048 #496
    working_dir = "/home/user/repos/silentframes/"
    data_path = working_dir + "data/" # where the mp3-files are
    annotation_path = data_path + "annotations/"
    stepsize = 3600 # in seconds
    max_size = 3600 # in seconds (1h=3600s)
    elements_per_hour = int((fs / (hs*1.0)) * max_size) # 160040 (for ws=512 und hs=496)
    y_axis_max_size = 0.7001 # maximal value of y axis (to get a uniform scale over all plots)
    y_axis_max_size_rms = 0.0025 # maximal value of y axis for the RMS feature (to get a uniform scale over all plots)
    y_axis_min_size = 0.0 # minimal value of y axis (to get a uniform scale over all plots)
    x_axis_max_size = elements_per_hour #160040 # maximal value of x axis (to get a uniform scale over all plots)
    x_axis_min_size =  0 #0 # minimal value of x axis (to get a uniform scale over all plots)
    min_comm_length = 3.0
    max_comm_length = 90.0
    
    show_figures = 0

    steps = np.ceil(max_size/stepsize)
    
    summary = [] 

    #files = ['RTL-h12.mp3']# take only non-folders
    #files = ['ARD-h16.mp3', 'Sat1-h7.mp3', 'RTL-h5.mp3', 'ZDF-h14.mp3', 'ZDF-h17.mp3']# take only non-folders
    
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
            plt.rcParams.update({'font.size': 14})

        for (j, step) in enumerate(xrange(0, max_size, stepsize)):
            #print  "###### from " + str(step) + " to " + str(step+stepsize) + " ###"
            
            # convert file and make spectrogram
            signal = np.frombuffer(audio_converter.decode_to_memory(abs_f, sample_rate=fs, skip=step, maxlen=stepsize), dtype=np.float32)
            magspec_without_last = abs(Spectrogram.spectrogram(signal, ws=ws, hs=hs))[:,:-1] 
            signal = None # unlink variable (the garbage collector can then free the memory if it is needed)
            print "magpsec shape w/o last element (always contains 0): %s"%(magspec_without_last.shape,)

            # arithmetic mean
            #amean = arithmetic_mean(magspec_without_last, axis=0)
            #(tmp,amean_masked) = filter_above_threshold(amean, 0.001, 0.0001, variable_pass_th_variance=1.1, top_n_smallest=50)

            # spectral flatness
            sflatness = spectral_flatness(magspec_without_last, axis=0)
            
            # RMS
            rms = rms_energy(magspec_without_last) / max(rms_energy(magspec_without_last))
            
            magspec_without_last = None # unlink variable (the garbage collector can then free the memory if it is needed)
          
            # extract local maxima
            # --> maybe smooth it before numpy.convolve() ... http://wiki.scipy.org/Cookbook/SignalSmooth
            #sflatness_smoothed = smooth(sflatness, window_len=11)
            #sflatness_local_maxima = extract_local_maxima(sflatness_smoothed, order=2)
            
            # maske-out all non-candidates
            channel =  f.split('-')[0] 
            daytime = int(((f.split('-')[1]) .split('h')[1]).split('.')[0])
            cutoff_th, fixed_pass_th, variable_pass_th_var, top_n_largest = get_threshold(channel, daytime)
            (variable_pass_th, rms_masked) = filter_above_threshold(rms, #sflatness_local_maxima, 
                        cutoff_th, fixed_pass_th, variable_pass_th_variance=variable_pass_th_var, top_n_smallest=top_n_largest)			
            #(variable_pass_th, sflatness_masked) = filter_below_threshold(sflatness, #sflatness_local_maxima, 
            #            cutoff_th, fixed_pass_th, variable_pass_th_variance=variable_pass_th_var, top_n_largest=top_n_largest)
            print "variable_pass_th: \t\t" + str(variable_pass_th)

            # set area between candidates to 1.0
            rms_area = masked_area_between_points(elements_per_hour, rms_masked, min_comm_length, max_comm_length, 1.0)
            
            # calculate statistics
            #tp, fp, tn, fn = check_results_start_points(elements_per_hour, sflatness_masked, anno, decision_rate=0.2)
            #baseline = np.zeros(sflatness.shape)
            #tp, fp, tn, fn = check_results_mask(elements_per_hour, baseline, anno, decision_rate=0.2)
            tp, fp, tn, fn = check_results_mask(elements_per_hour, rms_area, anno, decision_rate=0.2)	    
            summary.append( (os.path.splitext(f)[0], tp, fp, tn, fn) )
            
            # NEW
            tmp_time = np.linspace(0.0, np.float(3600.0), num=rms_area.shape[0], endpoint=False)

            out_f = open(working_dir + 'data/betweenSilentFrames/' + f.replace(".mp3", ".betweenSilentFrames") , 'wb')
            for i in range(rms_area.shape[0]):
                out_f.write('{0:.8f}'.format(tmp_time[i]).rstrip('0').rstrip('.') + "," + repr(rms_area[i]) + "\n")
            out_f.close()
            # NEW
            
            
            # plot data
            if show_figures:
                
                f, axarr = plt.subplots(4, sharex=True)
                
                # set ticks, ticklabels in seconds
                length = rms.shape[0]
                length_sec = Spectrogram.frameidx2time(length, ws=ws, hs=hs, fs=fs)
                tickdist_seconds = 120 # one tick every n seconds
                tickdist_labels_in_minutes = 60 # for seconds use 1; for minutes 60
                numticks = length_sec/tickdist_seconds
                tick_per_dist = int(round(length / numticks))
                xtickrange = range(length)[::tick_per_dist]
                xticklabels = ["%d"%((round(Spectrogram.frameidx2time(i, ws=ws, hs=hs, fs=fs)+j*stepsize)/tickdist_labels_in_minutes)) for i in xtickrange]

                #first subplot (old value: spectral flatness)
                axarr[0].plot(sflatness, alpha=0.8, linewidth=1)
                axarr[0].axhline(y=cutoff_th, linewidth=2.5, color='g')
                axarr[0].axhline(y=fixed_pass_th, linewidth=2.5, color='r')
                axarr[0].axhline(y=variable_pass_th, linewidth=2.5, color='k')
                print_annotations(axarr[0], anno, elements_per_hour, max_size, j*len(sflatness), (j+1)*len(sflatness), y_axis_max_size)
                #axarr[0].set_title("Spectral Flatness  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                axarr[0].set_ylim(ymin=y_axis_min_size, ymax=y_axis_max_size) # set constant y scale
                axarr[0].set_xlim(xmin=x_axis_min_size, xmax=x_axis_max_size)
                axarr[0].set_yticks(np.arange(0.0, y_axis_max_size, 0.1))
                axarr[0].set_title("Before peak picking (step: 1c)")
                axarr[0].set_ylabel("Spectral Flatness")
                
                #second subplot 
                axarr[1].plot(rms, alpha=0.8, linewidth=1)
                axarr[1].axhline(y=cutoff_th, linewidth=2.5, color='g')
                axarr[1].axhline(y=fixed_pass_th, linewidth=2.5, color='r')
                axarr[1].axhline(y=variable_pass_th, linewidth=2.5, color='k')
                print_annotations(axarr[1], anno, elements_per_hour, max_size, j*len(rms), (j+1)*len(rms), y_axis_max_size_rms)
                #axarr[1].set_title("Spectral Flatness  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                axarr[1].set_ylim(ymin=y_axis_min_size, ymax=y_axis_max_size_rms) # set constant y scale
                axarr[1].set_xlim(xmin=x_axis_min_size, xmax=y_axis_max_size_rms)
                axarr[1].set_yticks(np.arange(0.0, y_axis_max_size_rms, 0.0005))
                axarr[1].set_title("Before peak picking (step: 1c)")
                axarr[1].set_ylabel("RMS")
                
                #third subplot
                #axarr[2].plot(rms_masked, alpha=0.7, linewidth=2.5)
                axarr[2].plot(reset_value(rms_masked, 0.0, 1.0), alpha=0.7, linewidth=2.5)
                print_annotations(axarr[2], anno, elements_per_hour, max_size, j*len(rms), (j+1)*len(rms), y_axis_max_size_rms)
                #axarr[1].set_title("Spectral Flatness (selected)  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                axarr[2].set_ylim(ymin=y_axis_min_size, ymax=y_axis_max_size_rms) # set constant y scale
                axarr[2].set_xlim(xmin=x_axis_min_size, xmax=y_axis_max_size_rms)
                axarr[2].set_yticks(np.arange(0.0, y_axis_max_size_rms, 0.0005))
                axarr[2].set_title("After peak picking (step: 1c)")
                axarr[2].set_ylabel("RMS")

               #fourth subplot PROTOTYP only for whole hours
                axarr[3].plot(rms_area, alpha=0.7, linewidth=2.5)
                print_annotations(axarr[3], anno, elements_per_hour, max_size, j*len(rms), (j+1)*len(rms), y_axis_max_size)	    
                axarr[3].fill_between(range(rms_area.shape[0]), 0, rms_area, facecolor='blue', alpha=0.3)
                #axarr[2].set_title("Connect area between  |  Time: " + str(round(step/60,1)) + " - " + str(round((step+stepsize)/60, 1)) + " [min]")
                axarr[3].set_ylim(ymin=y_axis_min_size, ymax=y_axis_max_size) # set constant y scale
                axarr[3].set_xlim(xmin=x_axis_min_size, xmax=x_axis_max_size)
                axarr[3].set_yticks([y_axis_min_size, y_axis_max_size])
                axarr[3].set_yticklabels(['false', 'true'])
                axarr[3].set_title("At the end (step: 2)")
                axarr[3].set_ylabel("Commercial")
                

        if show_figures:
            f.subplots_adjust(hspace=0.25)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            plt.xticks(np.linspace(0, elements_per_hour, num=7), [0, 10, 20, 30, 40, 50, 60])
            plt.xlabel("Time [min]")
            f.set_size_inches(13, 11)
    
            plt.savefig('/home/user/Desktop/bsf_classification_process.png', bbox_inches='tight')    
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

    print "################################################"
    print "cutoff_th: \t\t" + str(cutoff_th)
    print "fixed_pass_th: \t\t" + str(fixed_pass_th)
    print "variable_pass_th_var: \t" + str(variable_pass_th_var)
    print "top_n_largest: \t\t" + str(top_n_largest)
    print "min_comm_length: \t" + str(min_comm_length)
    print "max_comm_length: \t" + str(max_comm_length)
    print "window_size: \t\t" + str(ws)
    print "hop_size: \t\t" + str(hs)

    if show_figures:
        raw_input('Press Enter to continue...')

run()
