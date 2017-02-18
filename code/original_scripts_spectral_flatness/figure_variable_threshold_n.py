
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

def gaussian(x, mu, sig):
    gaus = np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))
    gaus *= 2.0 # to flatten it at the top
    #gaus += 0.05
    if np.isscalar(x):
        if gaus > 1.0:
            gaus = 1.0
    else:
        gaus[gaus > 1.0] = 1.0
    return (np.maximum(1,gaus*30.0)).astype(int)

    
def get_threshold(channel, daytime):
    #daytime in full hours (0-23)
    cutoff_threshold = 0.3
    fixed_pass_threshold = 0.6
    variable_pass_th_variance = 1.1
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

    plt.ion() # turns on interactive mode 
    plt.figure() # create a new figure
    for mu, sig in [(14, 0.15), (17.0, 1.0)]:
        x = np.linspace(0.0, 24.0, 1000)
        y = gaussian(x, mu, sig)
        
        #plt.grid(True)
        plt.ylim(ymax=30.1)
        plt.xlim(xmax=24)
        plt.plot(x, y)
        plt.yticks(np.arange(0, 30.1, 5))
        plt.xticks(np.arange(min(x), max(x)+1, 4))
        plt.xlabel("Time of the day")
        plt.ylabel("n(t,b)")
    
    plt.savefig('/home/user/Desktop/variable_threshold_n.pdf', bbox_inches='tight')  
    plt.show()
    raw_input('Press Enter to continue...')
    
    
run()
