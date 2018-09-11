# ND2 extractor, Kymograph generator
# author: Emanuele Leoncini, Suyang Wan, Sadik Yidik
# product manager: Emanuele Leoncini, Somenath Bakshi

#
#
# Library dependence:
# use nd2reader 2.1.3, don't use the new version!!!!!
# library install instructions:
# In terminal, type:
# nd2reader: In terminal, type: "pip install "nd2reader==2.1.3"" or "pip3 install "nd2reader==2.1.3""
# PIL: In terminal, type: "pip install Pillow" or "pip3 install Pillow"
# pims: In terminal, type: "pip install pims_nd2" or "pip3 install pims_nd2"

#
# # Todo: create a GUI


import matplotlib.pyplot as pl
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
import multiprocessing

from datetime import datetime
import h5py
from tifffile import imsave
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import io as skio
from skimage.segmentation import active_contour
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn import linear_model
from scipy.signal import medfilt, argrelextrema
from scipy.misc import toimage
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt, binary_closing, binary_dilation
from multiprocessing import Pool
from skimage import util
from PIL import Image, ImageEnhance
from skimage.exposure import equalize_adapthist
import shutil


#############
# todo: deal with trenches at bottom & one fov with 2 trenches
# todo: incorporate Sadik's Phase Contrast channel
# todo: rotation correction for poor aligned chips
# todo: trench identification with multiple channels
class trench_kymograph():
    def __init__(self, nd2_file, main_directory, lane, pos, channel, trench_width, frame_start=None, frame_limit=None,
                 output_dir=None,file_path=None, other_channels = None):
        self.prefix = nd2_file[:-4]
        self.main_path = main_directory
        self.lane = lane
        self.channel = channel
        self.pos = pos
        self.trench_width = trench_width
        self.frame_start = frame_start
        self.frame_limit = frame_limit
        self.file_list = None
        self.frame_end = None
        self.file_length = None
        self.other_channels = other_channels
        # deal with stacks
        self.is_stack = None
        self.stack_length = None
        self.stack = None



        # TODO: change the path pattern if you didn't extract the ND2 with my extractor
        if file_path == None:
            self.file_path = self.main_path + "/" + self.prefix + "/Lane_" + str(self.lane).zfill(2) + "/pos_" + str(
                self.pos).zfill(3)
        else:
            self.file_path = file_path
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.file_path


    # TODO: change the path pattern if you didn't extract the ND2 with my extractor
    # Stack compatible
    def get_file_list(self, file_path=None, channel=None):
        self.is_stack = 0
        # TODO: to deal with multiple stacks
        if not file_path:
            file_path = self.file_path
        if not channel:
            channel = self.channel
        os.chdir(file_path)

        # special case for testing
        self.file_list = glob.glob('*' + channel + '*.tif*')
        first_file = skio.imread(self.file_list[0])
        first_shape = first_file.shape
        if len(first_shape) == 3:
            self.is_stack = 1

            self.stack_length = first_shape[0]
            if self.frame_start is None:
                self.frame_start = 0
            if self.frame_limit is None:
                self.frame_end = self.stack_length - self.frame_start
            else:
                self.frame_end = self.frame_start + self.frame_limit
            self.stack = first_file[self.frame_start:self.frame_end, :, :]
            [self.height, self.width] = [first_shape[1], first_file.shape[2]]
            self.file_length = self.stack.shape[0]


        else:
            def get_time(name):
                sub_name = name.split('_t')[1]
                num = sub_name.split('_c')[0]
                return int(num)

            self.file_list.sort(key=get_time)

            if self.frame_start is None:
                self.frame_start = 0
            if self.frame_limit is None:
                self.frame_end = len(self.file_list)
            else:
                self.frame_end = self.frame_start + self.frame_limit

            self.file_list = self.file_list[self.frame_start:self.frame_end]
            [self.height, self.width] = [first_shape[0], first_file.shape[1]]
            self.file_length = len(self.file_list)
            return

    def get_frame(self, i):
        if self.is_stack:
            return self.stack[i, :, :]
        else:
            return pl.imread(self.file_list[i])

    # from Sadik
    def background_enhance(self):
        self.get_file_list()  # run on original data
        self.enhanced_path = self.file_path + '/enhanced'
        # print(self.enhanced_path)
        if not os.path.exists(self.enhanced_path):
            os.makedirs(self.enhanced_path)
            # for i in range(self.file_length):
            for i in range(50):
                im_i = self.get_frame(i)
                if np.max(im_i) > 255:
                    im_i = self.to_8_bit(im_i)

                im_g = 255 * filters.gaussian(im_i, sigma=10)
                im_i = (im_i - im_g)
                im_i[im_i < 0] = 0
                if self.is_stack:
                    new_name = self.file_list[0].split('/')[-1]
                    new_name = new_name.split('.')[0] + '_t_' + str(i).zfill(3) + '.tiff'
                else:
                    new_name = self.file_list[i].split('/')[-1]
                cim = Image.fromarray((im_i).astype(np.uint8))
                contrast = ImageEnhance.Contrast(cim)
                cim = contrast.enhance(3)
                cim.save(os.path.join(self.enhanced_path, new_name))
            return

    def auto_crop(self):
        cropped_path = self.file_path + '/cropped'
        self.cropped_path = cropped_path
        if not os.path.exists(cropped_path):
            os.makedirs(cropped_path)
        self.get_file_list(self.file_path + '/enhanced')
        # for i in range(self.file_length):
        for i in range(50):
            im_i = self.get_frame(i)
            if np.max(im_i) > 255:
                im_i = self.to_8_bit(im_i)
            if i == 0:
                im_i_left = im_i[:, :self.width / 4]

                self.spread = list((map(self.N2spread, im_i_left > threshold_otsu(im_i_left) // 2)))
                self.spread = np.array(self.spread) / (self.width / 4)
                self.spread = medfilt(self.spread, 29)
                self.ytop = min(np.where(self.spread > 0.2)[0])  # leave some space
                self.ybot = max(np.where(self.spread > 0.45)[0]) - 30

            cim = Image.fromarray(im_i[self.ytop:self.ybot, :].astype(np.uint8))
            # get file name
            if self.is_stack:
                new_name = self.file_list[0].split('/')[-1]
                new_name = new_name.split('.')[0] + '_t_' + str(i).zfill(3) + '.tiff'
            else:
                new_name = self.file_list[i].split('/')[-1]
            cim.save(os.path.join(cropped_path, new_name))
        return

    def mask_all_trenches(self):
        self.cropped_path = self.file_path + "/cropped"
        self.get_file_list(self.cropped_path)
        # use the first 50 frames to get rough trench mask
        # convert to binary using max_entropy threshold

        im_stack = np.zeros((min(50, self.file_length), self.height, self.width))
        for i in range(min(50, self.file_length)):
            im_i = self.get_frame(i)
            thresh = threshold_otsu(im_i)
            im_i = self.make_binary(im_i, thresh)
            im_stack[i] = im_i
        # take median
        self.im_projected = np.ceil(np.percentile(im_stack, 70, axis=0)).astype(np.int8)
        # thresh = threshold_otsu(im_projected)
        out_file = "rough_mask.tiff"
        out = PIL.Image.frombytes("L", (self.width, self.height), self.im_projected.tobytes())

        out.save(out_file)


        # only analysis the tip
        sub_height = 50
        im_tip = self.im_projected[:sub_height, :]
        # remove small elements
        tip_trench = label(im_tip)
        reg_props = regionprops(tip_trench)
        for reg in reg_props:
            if reg.area < 50:
                reg_loc = reg.bbox
                filled_reg = np.zeros((reg_loc[2] - reg_loc[0], reg_loc[3] - reg_loc[1]))
                tip_trench[reg_loc[0]:reg_loc[2], reg_loc[1]:reg_loc[3]] = filled_reg
        out_file = "small_particle_removed.tiff"
        out = PIL.Image.frombytes("L", (self.width, sub_height), im_tip.tobytes())
        out.save(out_file)

        self.im_projected = im_tip

        # # close
        # # run("Options...", "iterations=5 count=1 black pad edm=8-bit do=Close stack");
        # im_closed = binary_closing(self.im_projected/255,structure=np.ones((4,4)),iterations=5)
        # im_closed = (255*im_closed).astype(np.int8)
        # out_file = "closed_mask.tiff"
        # out = PIL.Image.frombytes("L", (self.width, sub_height), im_closed.tobytes())
        # out.save(out_file)

        # vertical dilation
        structure = np.zeros((9, 9))
        structure[4, 4] = 1
        structure[5, 4] = 1
        structure[6, 4] = 1
        structure[7, 4] = 1
        structure[8, 4] = 1

        # im_dilated = binary_dilation(im_closed,structure=structure, iterations=20)
        im_dilated = binary_dilation(im_tip, structure=structure, iterations=20)
        im_dilated = (255 * im_dilated).astype(np.int8)
        out_file = "dilated_mask.tiff"
        out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
        out.save(out_file)

        # # close
        # # run("Options...", "iterations=5 count=1 black pad edm=8-bit do=Close stack");
        # im_closed = binary_closing(im_dilated/255,structure=np.ones((5,5)),iterations=5)
        # im_closed = (255*im_closed).astype(np.int8)
        # out_file = "closed_mask_after_dilation.tiff"
        # out = PIL.Image.frombytes("L", (self.width, sub_height), im_closed.tobytes())
        # out.save(out_file)

        # vertical dilation down
        structure = np.zeros((9, 9))
        structure[4, 4] = 1
        structure[5, 4] = 1
        structure[6, 4] = 1
        structure[7, 4] = 1
        structure[8, 4] = 1

        # im_dilated = binary_dilation(im_closed,structure=structure, iterations=200)
        im_dilated = binary_dilation(im_tip, structure=structure, iterations=200)
        im_dilated = (255 * im_dilated).astype(np.int8)
        out_file = "dilated_mask_after_closing_down.tiff"
        out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
        out.save(out_file)

        # vertical dilation up
        structure = np.zeros((5, 5))
        structure[0, 2] = 1
        structure[1, 2] = 1
        structure[2, 2] = 1
        im_dilated = binary_dilation(im_dilated, structure=structure, iterations=4)
        im_dilated = (255 * im_dilated).astype(np.int8)
        out_file = "dilated_mask_after_closing_down_up.tiff"
        out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
        out.save(out_file)

        # find binding box
        trench_ccomp = label(im_dilated)
        self.reg_props = regionprops(trench_ccomp)
        self.bbox_list = []
        for reg in self.reg_props:
            reg_width = reg.bbox[3] - reg.bbox[1]
            if self.trench_width * 0.7 < reg_width < self.trench_width * 1.3:
                self.bbox_list.append(reg.bbox)

        self.bbox_list.sort(key=lambda x: x[1])
        self.bbox_list = [(int(a + self.ytop), self.ybot, int(b), int(d)) for a, b, c, d in self.bbox_list]

        # exclude edges
        most_left = self.bbox_list[0]
        most_right = self.bbox_list[-1]
        if most_left[2] == 0:
            self.bbox_list = self.bbox_list[1:]
        if most_right[3] == self.width:
            self.bbox_list = self.bbox_list[:-1]
        # print(self.bbox_list)
        return

    def get_kymos(self):
        trench_num = len(self.bbox_list)
        trench_dict = {}
        # create empty stacks for each trenches
        kymo_path = self.file_path + '/kymograph'
        if self.other_channels:
            self.other_channels.append(self.channel)
            all_channels = self.other_channels
        else:
            all_channels = [self.channel]
        print(all_channels)

        # self.get_file_list()
        # self.get_file_list(self.enhanced_path)
        if not os.path.exists(kymo_path):
            os.makedirs(kymo_path)

        for c in all_channels:
            self.get_file_list(channel=c)
            print(self.file_list)
            for i in range(trench_num):
                cur_box = self.bbox_list[i]
                # print(self.ytop, cur_box[0], self.ybot,cur_box[1])
                trench_dict[i] = np.zeros((self.file_length, cur_box[1] - cur_box[0], cur_box[3] - cur_box[2]))
            for f_i in range(self.file_length):
                try:
                    file_i = self.file_list[f_i]
                except:
                    print("something is wrong")
                    continue
                im_t = pl.imread(file_i)

                for t_i in range(trench_num):
                    cur_box = self.bbox_list[t_i]
                    trench_dict[t_i][f_i] = im_t[cur_box[0]:self.ybot, cur_box[2]:cur_box[3]]

            for t_i in range(trench_num):
                trench_stack_name = kymo_path + "/Stack_Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(
                    3) + "_trench_" + str(t_i + 1).zfill(2) + "_top_c_" + c + ".tiff"

                imsave(trench_stack_name, trench_dict[t_i].astype(np.uint16))
                trench_dict[t_i] = None
        return

    def clean_up(self):
        shutil.rmtree(self.enhanced_path)
        # shutil.rmtree(self.cropped_path)
        return

    def run_kymo(self):
        self.get_file_list()
        self.background_enhance()
        self.auto_crop()
        self.mask_all_trenches()
        self.get_kymos()
        self.clean_up()
        return

    @staticmethod
    def to_8_bit(im):
        im_min = im.min()
        im_max = im.max()
        scaling_factor = (im_max - im_min)
        im = (im - im_min)
        im = (im * 255. / scaling_factor).astype(np.uint8)
        return im

    # Max entropy algorithm
    @staticmethod
    def max_entropy(data):
        # flatten the data
        data = data.flatten()
        data = np.sort(data)

        # histogram
        hist_v = np.histogram(data, bins=data.max())[0]
        # normalize hist
        hist_v = hist_v * 1. / len(data)
        # CDF
        cdf = hist_v.cumsum()
        # print(cdf)
        max_ent, threshold = 0, 0
        for i in range(len(cdf)):
            # for i in range(255):
            # low range
            cl = cdf[i]
            sub_hist = hist_v[:i + 1] / cl
            tot_ent = - np.sum(sub_hist * np.log(sub_hist))

            # high range
            ch = 1 - cl
            if ch > 0:
                sub_hist = hist_v[i:] / ch
                tot_ent -= np.sum(sub_hist * np.log(sub_hist))

                if tot_ent > max_ent:
                    max_ent, threshold = tot_ent, i

        return threshold

    @staticmethod
    def make_binary(im, threshold):
        im = im > threshold
        im = im.astype(int) * 255
        return im

    @staticmethod
    def N2spread(x):
        # get nonzero elements
        ix = np.where(x > 0)[0]
        # except empty signal
        if len(ix) < 1:
            return 0
        return np.mean(abs(ix[0] - ix))


###############
# test
if __name__ == "__main__":
    def run_kymo_generator(nd2_file, main_directory, lanes, poses, channel, trench_width, frame_start=None,
                           frame_limit=None, output_dir = None, file_path = None, other_channels = None):


        start_t = datetime.now()
        print('Kymo starts ')
        # drift correct for each lane:
        # trench identify for each pos
        for lane in lanes:
            def helper_kymo(p):
                new_kymo = trench_kymograph(nd2_file, main_directory, lane, p, channel, trench_width,
                                            frame_start, frame_limit, output_dir, file_path, other_channels)
                new_kymo.run_kymo()
                return

            cores = multiprocessing.cpu_count() / 3
            jobs = []
            batch_num = len(poses) / cores + 1

            for i in range(batch_num):
                start_ind = i * cores
                end_ind = start_ind + cores
                partial_poses = poses[start_ind:end_ind]

                for p in partial_poses:
                    j = multiprocessing.Process(target=helper_kymo, args=(p,))
                    jobs.append(j)
                    j.start()
                    # print(p, j.pid)

                for job in jobs:
                    # print(job.pid)
                    job.join()
        time_elapsed = datetime.now() - start_t
        print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))


    #
    # # TODO: Change me
    # nd2_file = "20180402--HYSTERESIS_ROBOT_RUN--Recovery001.nd2"
    # main_directory =r"/Volumes/SysBio/PAULSSON LAB/Leoncini/DATA_Ti3//20180402--HYSTERESIS_ROBOT_RUN"

    #
    # seg_channel = 'MCHERRY' # segementation channel
    # other_channels = ['GFP'] # has to be a list
    #
    # # in pixels, measure in FIJI with a rectangle
    # trench_length = 330
    # trench_width = 30 # has to be even
    #
    #
    # spatial = 0       # 0 for top trench, 1 for bottom, 2 for both
    # drift_correct = 1 # 1 for need correction, 0 for no
    #
    #
    # # TODO: Don't touch me!
    # run_kymo_generator(nd2_file, main_directory, lanes, poses, other_channels, seg_channel, trench_length, trench_width,
    #                    spatial, drift_correct)
    #
    nd2_file = "growth_expression.nd2"
    main_directory = r"/Volumes/SysBio/PAULSSON LAB/Somenath/DATA_Ti4/20180118_GC_16Strain_Barcoded/Timelapse/"
    lanes = range(1,6)  # has to be a list
    poses = range(1, 101)  # second value exclusive

    channel = 'BF'

    other_channels = ['GFP']

    # in pixels, measure in FIJI with a rectangle
    trench_width = 24

    # run_kymo_generator(nd2_file, main_directory, lanes, poses, channel, trench_width, other_channels)

    #
    new_kymo = trench_kymograph(nd2_file, main_directory, 1, 2, channel,  trench_width,frame_limit=50, other_channels=other_channels)
    new_kymo.get_kymos()
    # print(len(new_kymo.file_list))
    # # new_kymo.background_enhance()
    # new_kymo.auto_crop()
    # new_kymo.mask_all_trenches()
    # new_kymo.get_kymos()