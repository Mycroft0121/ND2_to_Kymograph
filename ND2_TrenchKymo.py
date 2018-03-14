import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as pl
import itertools

import re
import glob    # pathname pattern
import os
import json
from PIL import Image
from tifffile import imsave
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import io as skio
from skimage.segmentation import active_contour
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from sklearn import linear_model
from scipy.signal import medfilt, argrelextrema
from scipy.misc import toimage
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
#from multiprocessing import Pool



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

from datetime import datetime


# step 1, extract ND2 as usual
class ND2_extractor():
    def __init__(self, nd2_file, file_directory, xml_file=None, xml_dir=None, output_path=None):
        self.input_path    = file_directory
        self.nd2_file      = nd2_file
        self.nd2_file_name = nd2_file[:-4]
        self.xml_file      = xml_file
        self.xml_dir       = xml_dir
        self.output_path   = output_path
        self.main_dir      = file_directory+"/"+self.nd2_file_name
        self.nd2_f         = nd2_file
        self.file_dir      = file_directory
        self.pos_dict      = None
        self.pos_offset    = None
        self.lane_dict     = None

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
        self.lane_dict  = lane_dict
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
        self.pos_dict   = pos_dict
        self.lane_dict  = lane_dict
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
            os.makedirs(new_dir, 0755)
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
            os.makedirs(main_dir, 0755)

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
    def __init__(self, nd2_file, file_directory, lane, channel, pos, trench_length, trench_width, frame_start=None, frame_limit = None):
        self.main_path     = file_directory
        self.lane          = lane
        self.channel       = channel
        self.pos           = pos
        self.frame_limit   = frame_limit
        self.pos_path      = file_directory + "/"+ nd2_file[:-4] + "/Lane_" + str(lane).zfill(2)  + "/pos_" + str(pos).zfill(3)
        self.trench_length = trench_length
        self.trench_width  = trench_width
        self.frame_start   = frame_start

    # generate stacks for each fov, find the max intensity
    def get_trenches(self):
        # get the target files
        os.chdir(self.pos_path)
        files = glob.glob(self.pos_path + '/*'+ self.channel + '.tiff')


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
        if self.frame_start is None:
            self.frame_start = 0
        if self.frame_limit is None:
            self.frame_limit = total_t


        #
        # im_ave = 0
        # for i in xrange(self.frame_start, self.frame_limit):
        #     im_i    = pl.imread(files[i])
        #     if np.max(im_i) > 255:
        #         im_min = im_i.min()
        #         im_max = im_i.max()
        #         scaling_factor = (im_max - im_min)
        #         im_i = (im_i - im_min)
        #         im_i = (im_i * 255. / scaling_factor).astype(np.uint8)
        #     im_ave += im_i/50.0
        # out_file = "frame_average.tiff"
        # im_ave = im_ave.astype(np.uint16)
        # out = PIL.Image.frombytes("I;16", (im_ave.shape[1], im_ave.shape[0]), im_ave.tobytes())
        # out.save(out_file)
        im = pl.imread(files[0])
        [height, width]=im.shape
        h_list = range(height)
        w_list = range(width)
        coord_list = list(itertools.product(h_list, w_list))
        int_list   = range(0,256)
        int_dict   = {intensity:0 for intensity in int_list}
        pix_int_dict = {pix_loc:int_dict for pix_loc in coord_list}
        # only consider 200 frames
        t_step = max(1, (self.frame_limit-self.frame_start) / 200)

        total_frame = len(range(self.frame_start-1, total_t, t_step))
        # print total_frame

        self.im_stack = np.zeros((total_frame,height,width))
        count = 0
        for i in xrange(self.frame_start, total_t, t_step):
            im = pl.imread(files[i])
            if np.max(im) > 255:
                im_min = im.min()
                im_max = im.max()
                scaling_factor = (im_max - im_min)
                im = (im - im_min)
                im = (im * 255. / scaling_factor).astype(np.uint8)
            self.im_stack[count] = im
            count +=1
            # print i

        self.int_perc = np.percentile(self.im_stack,80,axis=0)
        out_file = "frame_80_pct.tiff"
        self.int_perc = self.int_perc.astype(np.uint16)
        out = PIL.Image.frombytes("I;16", (self.int_perc.shape[1], self.int_perc.shape[0]), self.int_perc.tobytes())
        out.save(out_file)


        # for i in xrange(self.frame_start, total_t+1, t_step):
        #     im = pl.imread(files[self.frame_start])
        #     if np.max(im) > 255:
        #         im_min = im.min()
        #         im_max = im.max()
        #         scaling_factor = (im_max - im_min)
        #         im = (im - im_min)
        #         im = (im * 255. / scaling_factor).astype(np.uint8)
        #     for idx_1, idx_2 in coord_list:
        #         idx_int = im[idx_1,idx_2]
        #         (pix_int_dict[(idx_1, idx_2)])[idx_int] +=1
        #     print(i)
        # self.dict = pix_int_dict
        # # get the 80? percentile intensity at each pixel
        # meta_pix_int = {pix_loc:0 for pix_loc in coord_list}
        # for idx_1, idx_2 in coord_list:
        #
        #     count = 0
        #     for key in sorted(pix_int_dict[(idx_1, idx_2)].iterkeys()):
        #
        # # return
        #


        # intensity scanning to find the box containing each trench

        # find the rough upper bound with horizontal intensity profile
        intensity_scan = self.int_perc.sum(axis=1)
        intensity_scan = intensity_scan / float(sum(intensity_scan))
        exp_int = np.exp(intensity_scan) - 1
        exp_int -= min(exp_int)
        peak_ind = np.argmax(exp_int)
        sub_intensity = exp_int[:peak_ind]
        # take the ratio of intensity/max_intensity and regard any pixel
        # lower than 14% of the max as non cells
        max_ratio = sub_intensity / sub_intensity[-1]
        self.upper_index = np.where(max_ratio > 0.12)[0][0]
        self.lower_index = self.upper_index + self.trench_length


        # crop image with upper & lower indices
        im_trenches = self.int_perc[self.upper_index:self.lower_index]
        # print(im_trenches.shape)
        # convert to 8-bit, using the imageJ way
        im_min = im_trenches.min()
        im_max = im_trenches.max()
        scaling_factor = (im_max - im_min)
        im_trenches = (im_trenches - im_min)
        im_trenches = (im_trenches * 255. / scaling_factor).astype(np.uint8)
        intensity_scan = np.amax(im_trenches, axis=0)
        peak_ind = self.detect_peaks(intensity_scan, mph=45, mpd=self.trench_width)
        left_ind = peak_ind - self.trench_width/2

        right_ind = peak_ind + self.trench_width/2
        self.ind_list = zip(left_ind, right_ind)


    def kymograph(self):
        if not os.path.isdir(self.pos_path + '/Kymographs'):
            os.system('mkdir "' + self.pos_path + '/Kymographs"')
        ori_files = glob.glob(self.pos_path + '/*'+self.channel+'*')

        # a helper function
        # sort files by time
        def get_time(name):
            sub_name = name.split('_t')[1]
            num = sub_name.split('_c')[0]
            return int(num)

        ori_files.sort(key=get_time)
        print(ori_files)

        # if no frame limit specified use all frames
        if self.frame_start is None:
            frame_start = 1
        if self.frame_limit is None:
            frame_limit = len(ori_files)
        trench_num = len(self.ind_list)

        for i in xrange(trench_num):
            # generate trench
            # trench in T0
            im_t = pl.imread(ori_files[frame_start - 1])

            trench_left,trench_right = self.ind_list[i]
            trench = im_t[self.upper_index:self.lower_index,trench_left:trench_right]
            centre = self.trench_width/2
            for file in ori_files[frame_start:frame_limit]:
                im_t = pl.imread(file)
                im_min = im_t.min()
                im_max = im_t.max()
                scaling_factor = (im_max - im_min)
                im = (im_t - im_min)
                im = (im * 255. / scaling_factor).astype(np.uint8)

                trench_8bit = im[self.upper_index:self.lower_index,trench_left:trench_right]

                intensity_scan = np.amax(trench_8bit, axis=0)

                peak_loc = intensity_scan.argmax()
                peak_shift = peak_loc-centre
                trench = np.concatenate((trench, im_t[self.upper_index:self.lower_index, trench_left+peak_shift:trench_right+peak_shift]),axis=1)

            out = PIL.Image.frombytes("I;16", (trench.shape[1], trench.shape[0]), trench.tobytes())
            trench_name = self.pos_path + "/Kymographs/Lane_" + str(self.lane).zfill(2)  + "_pos_" + str(self.pos).zfill(3) + "_trench_" + str(i+1)+'.tiff'
            out.save(trench_name)










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
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
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
nd2_file = "HYSTERESIS_GC_COLLECTION_INOCULATION.nd2"
file_directory = "/Volumes/Samsung_T3"
new_kymo = trench_kymograph(nd2_file, file_directory, 1,'MCHERRY', 15, 248, 14)
new_kymo.get_trenches()
# new_kymo.kymograph()

# new_extractor = ND2_extractor(nd2_file,file_directory)
# new_extractor.run_extraction()