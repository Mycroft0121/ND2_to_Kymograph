# steps:
import matplotlib.pyplot as pl
import itertools
import glob  # pathname pattern

from PIL import Image


# from ND2 extractor

import nd2reader
import os

import PIL
import numpy as np
from pims import ND2_Reader
import xml.etree.cElementTree as ET
import re
import pathos.multiprocessing
from datetime import datetime
import h5py
import cv2



# case 1: have bright field
# use the numbers on chip for template matching, using standard library

# case 2: only fluorescent channels
# get x index of trenches in each view

base_dir = "/Volumes/Samsung_T3/HYSTERESIS_96WP_GC_ROBOT_/Lane_01/pos_000"
os.chdir(base_dir)
files = glob.glob(base_dir + '/*_c_GFP.tif*')


def get_time(name):
    sub_name = name.split('_t')[1]
    # print sub_name
    num = sub_name.split('_c')[0]
    return int(num)

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='both', kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind





files.sort(key=get_time)
# take the initial 20s
files = files[0:50]

y_top    = 732
y_bottom = 868
trench_width = 20
peaks = {}
# print(len(files))
peak_ind = open('peak_ind.txt', 'w')
for i in range(len(files)):
    # print i
    im_i = pl.imread(files[i])
    # crop the trench region
    im_trenches = im_i[y_top:y_bottom]
    # take percentile here, not ave
    # im_trenches_ave = im_trenches.mean(axis=0)
    im_trenches_perc = np.percentile(im_trenches,80,axis=0)
    peak = detect_peaks(im_trenches_perc, mph = 300, mpd = trench_width)
    peaks[i] = peak
    # file.write('\n'.join(str(year) for year in years))
    peak_ind.write(' '.join(str(p) for p in peak))
    peak_ind.write('\n')
    # print peak

peak_ind.close()


# input: two lists of integers of similar size and values
# goal
# output: an integer indicating shift pixes (positive in right direction, negative in left)
def pairwise_list_align(list_a, list_b):
    shift = 0
    matches = 0
    i_a = 0
    i_b = 0
    # only consider middle
    list_a = list_a[1:-1]
    len_a = len(list_a)-2
    len_b = len(list_b)
    for x in list_a:
        found = 0
        while not found:
            diff = list_b[i_b] - x
            if diff< -trench_width:
                i_b +=1
            elif diff > trench_width: # this cell is lost
                break
            else:
                found = 1
                shift += diff
                matches += 1
    shift = shift*1./matches
    # print shift
    return shift

shift = []
for i in range(49):
    list_a = peaks[i]
    list_b = peaks[i+1]
    shift.append(pairwise_list_align(list_a, list_b))



shift= np.cumsum(np.array(shift)).astype(int)

pad = 0
for i in range(1,len(files)):
    # print i
    im_i = pl.imread(files[i])
    s = shift[i-1]*-1
    if s:
        # im_new = moveImage(im_i, move_y, move_x, pad=0)
        im_new = np.zeros((im_i.shape[0], im_i.shape[1]), dtype='int16')
        # if s < 0:
        #     xbound = -s
        # else:
        xbound = im_i.shape[1]


        if s >= 0:
            im_new[:,s:] = im_i[:,im_i.shape[1]-s]
            im_new[:,0:s] = pad
        else:
            im_new[:,:xbound+s] = im_i[:,-s:]
            im_new[:,xbound+s:] = pad
    else:
        im_new = im_i
    # print(shift[i-1])
    # print(im_i == im_new)

    tiff_name = files[i].split('.tif')[0] + '_new_new.tif'
    # image = im_new.base.astype(np.uint16)
    out = PIL.Image.frombytes("I;16", (im_new.shape[1], im_new.shape[0]), im_new.tobytes())
    out.save(tiff_name)




