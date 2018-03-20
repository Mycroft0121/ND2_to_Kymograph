import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as pl
import itertools

import re
import glob  # pathname pattern
import os
from PIL import Image
import math

from scipy.signal import medfilt, argrelextrema
from scipy.misc import toimage
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

# from ND2 extractor

import nd2reader
import os
import shutil
import PIL
import numpy as np
from pims import ND2_Reader
import functools
import xml.etree.cElementTree as ET
import re
import pathos.multiprocessing
import multiprocessing
from datetime import datetime
import h5py
import cv2


# step 1, extract ND2 as usual
class ND2_extractor():
    def __init__(self, nd2_file, file_directory, xml_file=None, xml_dir=None, output_path=None):
        self.input_path = file_directory
        self.nd2_file = nd2_file
        self.nd2_file_name = nd2_file[:-4]
        self.xml_file = xml_file
        self.xml_dir = xml_dir
        self.output_path = output_path
        self.main_dir = file_directory + "/" + self.nd2_file_name
        self.nd2_f = nd2_file
        self.file_dir = file_directory
        self.pos_dict = None
        self.pos_offset = None
        self.lane_dict = None

    def lane_info(self):
        # dict for lane info
        nd2_new = ND2_Reader(self.nd2_file)
        nd2_new.iter_axes = 'm'
        lane_dict = {}
        lane_dict[0] = 1
        pos_offset = {}
        cur_lane = 1
        pos_min = 0
        pos_offset[cur_lane] = pos_min - 1
        y_prev = nd2_new[0].metadata['y_um']
        pos_num = len(nd2_new)
        for i in range(1, pos_num):
            f = nd2_new[i]
            y_now = f.metadata['y_um']
            if abs(y_now - y_prev) > 200:  # a new lane
                cur_lane += 1
                pos_min = i - 1
                pos_offset[cur_lane] = pos_min
            lane_dict[i] = cur_lane
            y_prev = y_now
        nd2_new.close()
        self.lane_dict = lane_dict
        self.pos_offset = pos_offset

    def pos_info(self):
        cur_dir = os.getcwd()
        os.chdir(self.xml_dir)
        tree = ET.ElementTree(file=self.xml_file)
        root = tree.getroot()[0]
        pos_dict = {}
        lane_dict = {}
        pos_offset = {}
        lane_count = 0
        lane_name_prev = None
        dummy_count = 0
        for i in root:
            if i.tag.startswith('Point'):
                ind = int(i.tag[5:])
                pos_name = i[1].attrib['value']
                if len(pos_name) < 1:
                    pos_name = "dummy_" + str(dummy_count)
                    dummy_count += 1
                    lane_name_cur = "dummy"
                else:
                    lane_name_cur = re.match(r'\w', pos_name).group()
                if lane_name_cur != lane_name_prev:
                    lane_name_prev = lane_name_cur
                    lane_count += 1
                    pos_offset[lane_count] = ind - 1
                lane_dict[ind] = lane_count
                pos_dict[ind] = pos_name
        os.chdir(cur_dir)
        self.pos_dict = pos_dict
        self.lane_dict = lane_dict
        self.pos_offset = pos_offset

    def tiff_extractor(self, pos):
        nd2 = nd2reader.Nd2(self.nd2_f)
        if self.pos_dict:
            new_dir = self.main_dir + "/Lane_" + str(self.lane_dict[pos]).im(2) + "/" + self.pos_dict[pos] + "/"
        else:
            lane_ind = self.lane_dict[pos]
            pos_off = self.pos_offset[lane_ind]
            new_dir = self.main_dir + "/Lane_" + str(lane_ind).zfill(2) + "/pos_" + str(pos - pos_off).zfill(3) + "/"

        # create a folder for each position
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        os.chdir(new_dir)

        if self.pos_dict:
            meta_name = self.nd2_file_name + "_" + self.pos_dict[pos] + "_t"
        else:
            meta_name = self.nd2_file_name + "_pos_" + str(pos - pos_off).zfill(3) + "_t"

        for image in nd2.select(fields_of_view=pos):
            channel = image._channel
            channel = str(channel.encode('ascii', 'ignore'))
            time_point = image.frame_number
            tiff_name = meta_name + str(time_point).zfill(4) + "_c_" + channel + ".tiff"

            # save file in 16-bit
            # thanks to http://shortrecipes.blogspot.com/2009/01/python-python-imaging-library-16-bit.html
            image = image.base.astype(np.uint16)
            out = PIL.Image.frombytes("I;16", (image.shape[1], image.shape[0]), image.tobytes())
            out.save(tiff_name)

        os.chdir(self.file_dir)

    def run_extraction(self):
        start_t = datetime.now()

        os.chdir(self.input_path)
        # get position name if xml is available
        if self.xml_file:
            if not self.xml_dir:
                self.xml_dir = self.input_path
                self.pos_info()
        # otherwise get lane info from y_um
        else:
            self.lane_info()
        os.chdir(self.input_path)

        # switch to another ND2reader for faster iterations
        nd2 = nd2reader.Nd2(self.nd2_file)

        main_dir = self.input_path + "/" + self.nd2_file_name
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

        # parallelize extraction
        poses = nd2.fields_of_view
        cores = pathos.multiprocessing.cpu_count()
        pool = pathos.multiprocessing.Pool(cores)
        pool.map(self.tiff_extractor, poses)

        time_elapsed = datetime.now() - start_t
        print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))


        # TODO
        # return # of lanes, fov, channels


#############
# will use a lot from Sadik's code
class trench_kymograph():
    def __init__(self, nd2_file, file_directory, lane, info_channel, kymo_channels, pos, trench_length, trench_width,
                 frame_start=None, frame_limit=None):
        self.main_path = file_directory
        self.lane = lane
        self.info_channel = info_channel
        self.kymo_channel = kymo_channels
        self.pos = pos
        self.frame_limit = frame_limit
        self.pos_path = file_directory + "/" + nd2_file[:-4] + "/Lane_" + str(lane).zfill(2) + "/pos_" + str(pos).zfill(
            3)
        self.trench_length = trench_length
        self.trench_width = trench_width
        self.frame_start = frame_start
        self.meta = None

    # generate stacks for each fov, find the max intensity
    def get_trenches(self):
        # get the target files
        self.channel = self.info_channel
        os.chdir(self.pos_path)
        files = glob.glob(self.pos_path + '/*' + self.channel + '.tiff')

        # a helper function
        # sort files by time
        def get_time(name):
            sub_name = name.split('_t')[1]
            # print sub_name
            num = sub_name.split('_c')[0]
            return int(num)

        files.sort(key=get_time)
        # Meta fov generated by taking the 80% percentile of each pixel
        total_t = len(files)
        self.total_t = total_t
        if self.frame_start is None:
            self.frame_start = 0
        if self.frame_limit is None:
            self.frame_limit = min(50, total_t)
        # print(total_t)
        im = pl.imread(files[0])
        [height, width] = im.shape

        max_im = np.zeros((height, width))

        for i in range(self.frame_start, self.frame_limit):
            im_i = pl.imread(files[i])
            if np.max(im_i) > 255:
                im_min = im_i.min()
                im_max = im_i.max()
                scaling_factor = (im_max - im_min)
                im_i = (im_i - im_min)
                im_i = (im_i * 255. / scaling_factor).astype(np.uint8)
                im_i = np.array(im_i)
            im_concat = np.stack((max_im, im_i))
            max_im = np.max(im_concat, axis=0)

        out_file = "frame_max_50.tiff"
        max_im = max_im.astype(np.uint16)
        out = PIL.Image.frombytes("I;16", (width, height), max_im.tobytes())
        out.save(out_file)
        self.meta = max_im.astype(np.uint8)

        # intensity scanning to find the box containing each trench

        # find the rough upper bound with horizontal intensity profile
        intensity_scan = self.meta.sum(axis=1)
        intensity_scan = intensity_scan / float(sum(intensity_scan))
        exp_int = np.exp(intensity_scan) - 1
        exp_int -= min(exp_int)
        peak_ind = np.argmax(exp_int)
        sub_intensity = exp_int[:peak_ind]
        # take the ratio of intensity/max_intensity and regard any pixel
        # lower than 14% of the max as non cells
        max_ratio = sub_intensity / sub_intensity[-1]
        self.upper_index = np.where(max_ratio > 0.12)[0][0] - 20
        self.lower_index = int(self.upper_index + self.trench_length * 0.8)

        # crop image with upper & lower indices
        im_trenches = self.meta[self.upper_index:self.lower_index]
        # print(im_trenches.shape)
        # convert to 8-bit, using the imageJ way
        im_min = im_trenches.min()
        im_max = im_trenches.max()
        scaling_factor = (im_max - im_min)
        im_trenches = (im_trenches - im_min)
        im_trenches = (im_trenches * 255. / scaling_factor).astype(np.uint8)
        intensity_scan = np.amax(im_trenches, axis=0)
        peak_ind = self.detect_peaks(intensity_scan, mph=45, mpd=self.trench_width)
        left_ind = peak_ind - int(self.trench_width / 2)

        right_ind = peak_ind + int(self.trench_width / 2)
        self.ind_list = list(zip(left_ind, right_ind))
        self.ind_list = np.array(self.ind_list)
        hf = h5py.File('box_info.h5', 'w')
        hf.create_dataset('box', data=self.ind_list)
        hf.create_dataset('upper_index', data=self.upper_index)
        hf.create_dataset('lower_index', data=self.upper_index + self.trench_length)
        hf.close()

    def kymograph(self, channel, box_info=None, frame_limit=None):
        if not os.path.isdir(self.pos_path + '/Kymographs'):
            os.system('mkdir "' + self.pos_path + '/Kymographs"')
        ori_files = glob.glob(self.pos_path + '/*' + channel + '*')
        if box_info:
            hf = h5py.File(box_info, 'r')
            self.ind_list = hf.get('box').value
            self.upper_index = hf.get('upper_index').value
            self.lower_index = hf.get('lower_index').value
            hf.close()

        if type(self.meta) == None:
            self.meta = pl.imread("frame_max_50.tiff")
            im_min = self.meta.min()
            im_max = self.meta.max()
            scaling_factor = (im_max - im_min)
            self.meta = (self.meta - im_min)
            self.meta = (self.meta * 255. / scaling_factor).astype(np.uint8)


        # a helper function
        # sort files by time
        def get_time(name):
            sub_name = name.split('_t')[1]
            num = sub_name.split('_c')[0]
            return int(num)

        ori_files.sort(key=get_time)

        self.total_t = len(ori_files)
        # if no frame limit specified use all frames
        if self.frame_start is None:
            self.frame_start = 0
        if frame_limit is None:
            self.frame_limit = self.total_t

        trench_num = len(self.ind_list)

        self.lower_index = self.upper_index + self.trench_length

        all_kymo = {}
        for i in range(trench_num):
            all_kymo[i] = np.zeros((self.total_t, self.trench_length, self.trench_width))
        file_list = ori_files[self.frame_start:self.frame_limit]

        for f_i in range(len(file_list)):
            file = file_list[f_i]
            im_t = pl.imread(file)
            im_t = pl.imread(file)
            im_min = im_t.min()
            im_max = im_t.max()
            scaling_factor = (im_max - im_min)
            im = (im_t - im_min)
            im = (im * 255. / scaling_factor).astype(np.uint8)
            # template matching

            tl, br = self.matchTemplate(im)
            if f_i == self.frame_start:
                ref_br = br
            else:
                move_x = ref_br[1] - br[1]
                move_y = ref_br[0] - br[0]
                # print(files[i-1].split('/')[-1],tl,br,move_x,move_y)
                # imc = self.moveImage(imc, move_x, move_y)
                im_t = self.moveImage(im_t, move_x, move_y, pad=0)

            for i in range(trench_num):
                trench_left, trench_right = self.ind_list[i]
                # trench_8bit = im[self.upper_index:self.lower_index, trench_left:trench_right]
                # intensity_scan = np.amax(trench_8bit, axis=0)
                # peak_loc = intensity_scan.argmax()
                # peak_shift = int(peak_loc - centre)
                trench = im_t[self.upper_index:self.lower_index, max(0, trench_left):trench_right]

                # correct for the first trench at edge
                if i == 0:
                    if trench.shape[1] < trench_width:
                        pad = np.zeros((trench_length, trench_width - trench.shape[1]))
                        trench = np.concatenate((pad, trench), axis=1).astype(np.uint16)

                # correct for the last trench at edge
                elif i == trench_num - 1:
                    if trench.shape[1] < trench_width:
                        pad = np.zeros((trench_length, trench_width - trench.shape[1]))
                        trench = np.concatenate((trench, pad), axis=1).astype(np.uint16)
                all_kymo[i][f_i] = trench.astype(np.uint16)


        for i in range(trench_num):
            this_kymo = np.concatenate(all_kymo[i], axis=1).astype(np.uint16)
            all_kymo[i] = None
            out = PIL.Image.frombytes("I;16", (this_kymo.shape[1], this_kymo.shape[0]), this_kymo.tobytes())
            trench_name = self.pos_path + "/Kymographs/Channel_" + channel + "_Lane_" + str(self.lane).zfill(
                2) + "_pos_" + str(
                self.pos).zfill(3) + "_trench_" + str(i + 1) + '.tiff'
            out.save(trench_name)

    # from Sadik
    def matchTemplate(self, img):
        """
        Takes an image and a template to search for and returns bottom right
        and top left coordinates. (top_left,bottom_right) ((int,int),(int,int))
        """
        w, h = self.meta.shape[::-1]
        # w, h = template.shape[::-1]

        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        method = cv2.TM_CCOEFF

        # Apply template Matching
        self.res = cv2.matchTemplate(img, self.meta, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        return (top_left, bottom_right)

        # # for file in ori_files[self.frame_start:self.frame_limit]:
        #     im_t = pl.imread(file)
        #     im_min = im_t.min()
        #     im_max = im_t.max()
        #     scaling_factor = (im_max - im_min)
        #     im = (im_t - im_min)
        #     im = (im * 255. / scaling_factor).astype(np.uint8)
        #     for i in range(trench_num):
        #         trench_left, trench_right = self.ind_list[i]
        #         trench_8bit = im[self.upper_index:self.lower_index, trench_left:trench_right]
        #         intensity_scan = np.amax(trench_8bit, axis=0)
        #         peak_loc = intensity_scan.argmax()
        #         peak_shift = int(peak_loc - centre)
        #         trench =  im_t[self.upper_index:self.lower_index,
        #                                          (trench_left + peak_shift):(trench_right + peak_shift)]
        #         # correct for the at edge
        #         if i == trench_num -1:
        #             if trench.shape[1] < trench_width:
        #                 pad    = np.zeros((trench_length, trench_width -trench.shape[1]))
        #                 trench = np.concatenate((trench,pad),axis=1).astype(np.uint16)
        #         # correct for last trenches at edge
        #         if i == trench_num -1:
        #             if trench.shape[1] < trench_width:
        #                 pad    = np.zeros((trench_length, trench_width -trench.shape[1]))
        #                 trench = np.concatenate((trench,pad),axis=1).astype(np.uint16)
        #         all_kymo[i][f_i] = trench
        # openCV template matching

    def run_kymo(self):
        for c in self.kymo_channel:
            if c == self.info_channel:
                self.get_trenches()
                self.kymograph(c)
            else:
                self.kymograph(c, box_info='box_info.h5')

    @staticmethod
    def moveImage(im, move_x, move_y, pad=0):
        """
        Moves the image without changing frame dimensions, and
        pads the edges with given value (default=0).
        """

        if move_y > 0:
            ybound = -move_y
        else:
            ybound = im.shape[1]

        if move_x > 0:
            xbound = -move_x
        else:
            xbound = im.shape[0]

        if move_x >= 0 and move_y >= 0:
            im[move_x:, move_y:] = im[:xbound, :ybound]
            im[0:move_x, :] = pad
            im[:, 0:move_y] = pad
        if move_x < 0 and move_y >= 0:
            im[:move_x, move_y:] = im[-move_x:, :ybound]
            im[move_x:, :] = pad
            im[:, 0:move_y] = pad
        if move_x >= 0 and move_y < 0:
            im[move_x:, :move_y] = im[:xbound, -move_y:]
            im[0:move_x, :] = pad
            im[:, move_y:] = pad
        if move_x < 0 and move_y < 0:
            im[:move_x, :move_y] = im[-move_x:, -move_y:]
            im[move_x:, :] = pad
            im[:, move_y:] = pad

        return im

    @staticmethod
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




###############
# test
if __name__ == "__main__":
    nd2_file = "NDL88_3Lane_RL5850_2Lane_8uMIPTG_DATA.nd2"
    file_directory = r"Z:\Somenath\DATA_Ti3\20180313\DATA"
    # lane, channel, pos, trench_length, trench_width
    # new_kymo = trench_kymograph(nd2_file, file_directory, 1,'GFP', 2, 526, 20)
    # os.chdir(r"Z:\Somenath\DATA_Ti3\20180313\DATA\NDL88_3Lane_RL5850_2Lane_8uMIPTG_DATA\Lane_01\pos_002")
    # new_kymo.get_trenches()

    # (self, nd2_file, file_directory, lane, info_channel,kymo_channels, pos, trench_length, trench_width, frame_start=None, frame_limit = None):
    # new_kymo.kymograph('GFP','box_info.h5')
    lanes = range(1, 6)
    poses = range(1, 38)
    info_channel = 'GFP'
    kymo_channels = ['GFP', 'RFP']

    trench_length = 526
    trench_width = 20

    for lane in lanes:
        for pos in poses:
            start_t = datetime.now()
            new_kymo = trench_kymograph(nd2_file, file_directory, lane, info_channel, kymo_channels, pos, trench_length,
                         trench_width)
            new_kymo.run_kymo()
            time_elapsed = datetime.now() - start_t
            print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))
    # new_extractor = ND2_extractor(nd2_file,file_directory)
    # new_extractor.run_extraction()
    # parallelization of extraction part
    # new_kymo = trench_kymograph(nd2_file, file_directory, 1, info_channel, kymo_channels, 7, trench_length,
    #                             trench_width)




