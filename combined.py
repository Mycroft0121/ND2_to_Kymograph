# ND2 extractor, Kymograph generator
# author: Suyang Wan
# product manager: Emanuele Leoncini, Somenath Bakshi
# Special thanks for technical support: Sadik Yidik
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
from skimage import io as skio

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.signal import medfilt
from scipy.ndimage.morphology import binary_dilation
from PIL import Image, ImageEnhance
import shutil

# todo: fix extractor xml file problem
# todo: new class for segmentation & lineage tracking
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
            meta_name = self.nd2_file_name + "_" + self.pos_dict[pos] + "_Time_"  #TODO: name pattern changed
        else:
            meta_name = self.nd2_file_name + "_pos_" + str(pos - pos_off).zfill(3) + "_Time_" #TODO: name pattern changed

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


#############
# todo: rotation correction for poor aligned chips
class trench_kymograph():
    def __init__(self, nd2_file, main_directory, lane, pos, channel, seg_channel, spatial,trench_length=None, trench_width=None,
                 drift_correct=0, find_correct=0, frame_start=None, frame_limit=None, output_dir=None,
                 box_info=None, saving_option = 0, clean_up=1, chip_length=None, chip_width=None, magnification = None):
    # def __init__(self, nd2_file, main_directory, lane, pos, channel, seg_channel, trench_length, trench_width,
    #              spatial,
    #              drift_correct=0, find_correct=0, frame_start=None, frame_limit=None, output_dir=None,
    #              box_info=None):
        self.prefix = nd2_file[:-4]
        self.main_path = main_directory
        self.lane = lane
        self.channel = channel
        self.seg_channel = seg_channel
        self.pos = pos
        self.trench_length = trench_length
        self.trench_width = trench_width
        self.frame_start = frame_start
        self.frame_limit = frame_limit
        self.drift_correct = drift_correct
        self.find_correct = find_correct
        self.drift_x = None
        self.drift_y = None
        self.drift_x_txt = None
        self.drift_y_txt = None
        self.spatial = spatial  # 0 for top, 1 for bottom, 2 for both
        self.tops = []
        self.bottoms = []
        self.meta = None
        self.height = None
        self.width = None
        self.total_t = None
        self.out_file = None
        self.box_info = box_info  # file names
        self.file_list = None
        self.frame_end = None

        # new for phase
        self.is_stack = 0
        self.stack_length = None
        self.stack = None
        self.file_length = 0
        self.cropped_path = None
        self.spread = None
        self.bad_pos = [0,0]

        self.bbox_dict = {}

        self.im_projected = None

        self.bottom_cut = 0
        self.magnification = magnification


        self.saving_option = saving_option  # 0 for stack, 1 for kymo, 2 for both, default only save stacks
        self.clean = clean_up # whether delete enhanced/cropped, default yes

        self.chip_length = chip_length
        self.chip_width  = chip_width

        # 6.5 is the magic number for Ti3, Ti4
        if ((self.trench_length and self.chip_length) is None) or ((self.trench_width and self.chip_width) is None):
            print("Error: trench dimension not specified")
            exit()
        if not self.trench_length:
            if not self.magnification:
                print("Error: magnification not specified")
                exit()
            self.trench_length = int(self.chip_length/6.5*self.magnification)
        if not self.trench_width:
            if not self.magnification:
                print("Error: magnification not specified")
                exit()
            self.trench_width = int((self.chip_width/6.5*self.magnification))
            if self.seg_channel != 'BF':   # if not phase contrast, dilate trench width
                self.trench_width *= 1.2

        # TODO: change the path pattern if you didn't extract the ND2 with my extractor
        self.file_path = self.main_path + "/" + self.prefix + "/Lane_" + str(self.lane).zfill(2) + "/pos_" + str(
            self.pos).zfill(3)
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.file_path

    ###
    # TODO: change the path pattern if you didn't extract the ND2 with my extractor
    def get_file_list(self, file_path=None, channel=None, spatial=''):
        self.is_stack = 0
        # TODO: to deal with multiple stacks
        if not file_path:
            file_path = self.file_path
        if not channel:
            channel = self.channel
        os.chdir(file_path)

        # special case for testing
        self.file_list = glob.glob(spatial+'*' + channel + '*.tif*')
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
                sub_name = name.split('_Time_')[1]
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

    # only for stacks
    def get_frame(self, i):
        if self.is_stack:
            return self.stack[i, :, :]
        else:
            return pl.imread(self.file_list[i])

    def find_drift(self):
        lane_path = self.main_path + "/" + self.prefix + "/Lane_" + str(self.lane).zfill(2)
        tops = []
        peaks = []
        file_num = len(self.file_list)
        drift_y = open(lane_path + '/drift_y.txt', 'w')
        drift_x = open(lane_path + '/drift_x.txt', 'w')
        y_shift = [0]
        for i in range(len(self.file_list)):
            # print(self.find_top(i))
            tops.append(self.find_top(i))

        for i in range(len(tops)-1):
            diff = 0
            # diff = tops[i+1] - tops[i]
            # if diff > 10:
            #     diff = 0
            y_shift.append(diff)

        for i in range(len(self.file_list)):
            peaks.append(self.find_peaks(i, tops))

        # positive: downwards drift
        drift_y.write(' '.join(map(str, y_shift)))
        # print(y_shift)


        x_shift = [0]

        for i in range(file_num - 1):
            list_a = peaks[i]
            list_b = peaks[i + 1]
            move = self.pairwise_list_align(list_a, list_b, self.trench_width * 0.75)
            x_shift.append(move)

        # positive: drift to the right
        x_shift = np.cumsum(np.array(x_shift)).astype(int)

        drift_x.write(' '.join(map(str, x_shift.tolist())))

        self.drift_x = x_shift
        self.drift_y = y_shift
        self.drift_x_txt = 'drift_x.txt'
        self.drift_y_txt = 'drift_y.txt'
        return

    def read_drift(self):
        self.drift_x_txt = 'drift_x.txt'
        self.drift_y_txt = 'drift_y.txt'
        lane_path = self.main_path + "/" + self.prefix + "/Lane_" + str(self.lane).zfill(2)
        self.drift_x_txt = lane_path + "/" + self.drift_x_txt
        self.drift_y_txt = lane_path + "/" + self.drift_y_txt
        # read files into np array
        self.drift_x = np.loadtxt(self.drift_x_txt, dtype=int, delimiter=' ')
        self.drift_y = np.loadtxt(self.drift_y_txt, dtype=int, delimiter=' ')
        return

    def find_top(self, i):
        im_i = pl.imread(self.file_list[i])
        x_per = np.percentile(im_i, 95, axis=1)
        intensity_scan = x_per
        intensity_scan = intensity_scan / float(sum(intensity_scan))
        # normalize intensity
        im_min = intensity_scan.min()
        im_max = intensity_scan.max()
        scaling_factor = (im_max - im_min)
        intensity_scan = (intensity_scan - im_min)
        intensity_scan = (intensity_scan / scaling_factor)

        if self.spatial == 1:
            # actually  bottoms, but mie..
            top = np.where(intensity_scan > 0.2)[0][-1]
        else:
            top = np.where(intensity_scan > 0.2)[0][0]
        return top

    def find_peaks(self, i, tops):
        im_i = pl.imread(self.file_list[i])
        # crop the trench region
        im_trenches = im_i[tops[0]:tops[0] + self.trench_length]
        im_trenches_perc = np.percentile(im_trenches, 80, axis=0)
        # normalize intensity
        im_min = im_trenches_perc.min()
        im_max = im_trenches_perc.max()
        scaling_factor = (im_max - im_min)
        im_trenches_perc = (im_trenches_perc - im_min)
        im_trenches_perc = (im_trenches_perc / scaling_factor)
        peak = self.detect_peaks(im_trenches_perc, mph=0.15, mpd=self.trench_width)
        new_peak = self.peak_correct(peak, im_trenches_perc)
        return new_peak

    def peak_correct(self, old_peak, im_intensity):
        half_trench_width = self.trench_width/2
        new_peaks = [old_peak[0]]
        for p in old_peak[1:-1]:
            half_p_height = im_intensity[p]/2 # int
            full_peak = im_intensity[p - half_trench_width:p + half_trench_width+1]
            p_tops  = np.where(full_peak>half_p_height)
            p_left  = p - half_trench_width + p_tops[0][0]
            p_right = p - half_trench_width + p_tops[0][-1]
            p_corrected = (p_left + p_right)/2

            new_peaks.append(p_corrected)
        new_peaks.append(old_peak[-1])
        return new_peaks

    def get_trenches(self):
        os.chdir(self.file_path)
        # use the first 50 frames to identify trench relation
        # TODO: change this part to add more flexibility, like backwards trench identification for persistors
        frame_num = len(self.file_list)
        # using the 85 percentile of the intensity of the first 50 frames as the meta-representation
        im_stack = np.zeros((min(50, frame_num), self.height, self.width))
        for i in range(min(50, frame_num)):
            im_i = pl.imread(self.file_list[i])
            if np.max(im_i) > 255:
                im_i = self.to_8_bit(im_i)
            if self.drift_correct == 1:
                # correct for drift
                move_x = self.drift_x[i]
                temp = np.zeros((self.height, self.width))
                if move_x>0:
                    temp[:, :self.width-move_x] = im_i[:,move_x:]
                else:
                    temp[:,(-move_x):] = im_i[:,:self.width+move_x]
                im_i = temp

            im_stack[i] = im_i
        perc = np.percentile(im_stack, 85, axis=0).astype(np.uint8)
        out_file = "perc_85_frame_50.tiff"

        # convert to 8-bit, using the imageJ way
        out = PIL.Image.frombytes("L", (self.width, self.height), perc.tobytes())
        out.save(out_file)

        # identify tops & bottoms
        x_per = np.percentile(perc, 90, axis=1)
        intensity_scan = x_per
        intensity_scan = intensity_scan / float(sum(intensity_scan))
        # normalize intensity
        im_min = intensity_scan.min()
        im_max = intensity_scan.max()
        scaling_factor = (im_max - im_min)
        intensity_scan = (intensity_scan - im_min)
        intensity_scan = (intensity_scan / scaling_factor)

        if self.spatial != 1:  # top
            top = np.where(intensity_scan > 0.2)[0][0] - 30
            bottom = top + self.trench_length + 60
            self.tops.append(top)
            self.bottoms.append(bottom)
        if self.spatial != 0:  # bottom
            bottom = np.where(intensity_scan > 0.2)[0][-1] + 30
            top = bottom - self.trench_length - 60
            self.tops.append(top)
            self.bottoms.append(bottom)


        # if self.spatial == 0:  # top
        #     top = np.where(intensity_scan > 0.2)[0][0] - 30
        #     bottom = top + self.trench_length + 60
        #     self.tops.append(top)
        #     self.bottoms.append(bottom)
        # elif self.spatial == 1:  # bottom
        #     bottom = np.where(intensity_scan > 0.2)[0][-1] + 30
        #     top = bottom - self.trench_length - 60
        #     self.tops.append(top)
        #     self.bottoms.append(bottom)
        # else:  # both
        #     # top one
        #     top = np.where(intensity_scan > 0.2)[0][0] - 30
        #     bottom = top + self.trench_length + 60
        #     self.tops.append(top)
        #     self.bottoms.append(bottom)
        #     # bottom one
        #     bottom = np.where(intensity_scan > 0.2)[0][-1] + 30
        #     top = bottom - self.trench_length - 60
        #     self.tops.append(top)
        #     self.bottoms.append(bottom)

        # identify trenches
        peak_ind_dict = {}
        if self.spatial == 2:
            for i in range(2):
                im_trenches = perc[self.tops[i]:self.bottoms[i]]
                im_trenches_perc = np.percentile(im_trenches, 80, axis=0)

                # normalize intensity
                im_min = im_trenches_perc.min()
                im_max = im_trenches_perc.max()
                scaling_factor = (im_max - im_min)
                im_trenches_perc = (im_trenches_perc - im_min)
                im_trenches_perc = (im_trenches_perc / scaling_factor)
                peak_ind = self.detect_peaks(im_trenches_perc, mph=0.35, mpd=self.trench_width)

                # corrected
                peak_ind = np.array(self.peak_correct(peak_ind,im_trenches_perc))

                if peak_ind[0] < (self.trench_length / 2):
                    peak_ind = peak_ind[1:]
                if (self.width - peak_ind[-1]) < (self.trench_length / 2):
                    peak_ind = peak_ind[:-1]
                left_ind = np.array(peak_ind) - int(self.trench_width / 2)
                right_ind = peak_ind + int(self.trench_width / 2)
                ind_list = list(zip(left_ind, right_ind))
                ind_list = np.array(ind_list)
                peak_ind_dict[i] = ind_list
        else:
            im_trenches = perc[self.tops[0]:self.bottoms[0]]
            im_trenches_perc = np.percentile(im_trenches, 80, axis=0)
            # normalize intensity
            im_min = im_trenches_perc.min()
            im_max = im_trenches_perc.max()
            scaling_factor = (im_max - im_min)
            im_trenches_perc = (im_trenches_perc - im_min)
            im_trenches_perc = (im_trenches_perc / scaling_factor)
            peak_ind = self.detect_peaks(im_trenches_perc, mph=0.35, mpd=self.trench_width)
            if peak_ind[0] < (self.trench_length / 2):
                peak_ind = peak_ind[1:]
            if (self.width - peak_ind[-1]) < (self.trench_length / 2):
                peak_ind = peak_ind[:-1]
            left_ind = peak_ind - int(self.trench_width / 2)
            right_ind = peak_ind + int(self.trench_width / 2)
            ind_list = list(zip(left_ind, right_ind))
            ind_list = np.array(ind_list)
            peak_ind_dict[0] = ind_list

        self.box_info = []
        if self.spatial == 2:
            h5_name_top = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_top.h5"
            self.box_info.append(h5_name_top)
            hf_t = h5py.File(h5_name_top, 'w')
            hf_t.create_dataset('box', data=peak_ind_dict[0])
            hf_t.create_dataset('upper_index', data=self.tops[0])
            hf_t.create_dataset('lower_index', data=self.bottoms[0])
            hf_t.close()
            h5_name_bottom = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_bottom.h5"
            self.box_info.append(h5_name_bottom)
            hf_b = h5py.File(h5_name_bottom, 'w')
            hf_b.create_dataset('box', data=peak_ind_dict[1])
            hf_b.create_dataset('upper_index', data=self.tops[1])
            hf_b.create_dataset('lower_index', data=self.bottoms[1])
            hf_b.close()
            # print(peak_ind_dict)
        else:
            local = ['top', 'bottom']
            h5_name = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_" + local[
                self.spatial] + ".h5"
            self.box_info.append(h5_name)
            hf = h5py.File(h5_name, 'w')
            hf.create_dataset('box', data=peak_ind_dict[0])
            hf.create_dataset('upper_index', data=self.tops[0])
            hf.create_dataset('lower_index', data=self.bottoms[0])
            hf.close()
        return

    # Phase contrast trench detection

    def background_enhance(self):
        self.get_file_list()  # run on original data
        self.enhanced_path = self.file_path + '/enhanced'
        # print(self.enhanced_path)
        if not os.path.exists(self.enhanced_path):
            os.makedirs(self.enhanced_path)
            # for i in range(self.file_length):
            for i in range(min(50,self.file_length)):
                im_i = self.get_frame(i)
                if np.max(im_i) > 255:
                    im_i = self.to_8_bit(im_i)

                im_g = 255 * filters.gaussian(im_i, sigma=10)
                im_i = (im_i - im_g)
                im_i[im_i < 0] = 0
                if self.is_stack:
                    new_name = self.file_list[0].split('/')[-1]
                    new_name = new_name.split('.')[0] + '_Time_' + str(i).zfill(3) + '.tiff'
                else:
                    new_name = self.file_list[i].split('/')[-1]
                cim = Image.fromarray((im_i).astype(np.uint8))
                contrast = ImageEnhance.Contrast(cim)
                cim = contrast.enhance(3)
                cim.save(os.path.join(self.enhanced_path, new_name))
            return

    # add spatial support
    def auto_crop(self):
        cropped_path = self.file_path + '/cropped'
        self.cropped_path = cropped_path
        if not os.path.exists(cropped_path):
            os.makedirs(cropped_path)
        self.get_file_list(file_path=self.file_path + '/enhanced')
        # for i in range(self.file_length):
        for i in range(self.file_length):
            im_i = self.get_frame(i)
            if np.max(im_i) > 255:
                im_i = self.to_8_bit(im_i)
            if i == 0:
                self.spread = list((map(self.N2spread, im_i > threshold_otsu(im_i) // 2)))
                self.spread = np.array(self.spread) / self.width
                self.spread = medfilt(self.spread, 29)
                if self.spatial != 1:  # both have top
                    self.ytop_t = min(np.where(self.spread > 0.2)[0])  # leave some space
                    # self.ybot_t = max(np.where(self.spread > 0.45)[0]) - 30
                    # to have consistent height
                    self.ybot_t = self.ytop_t + self.trench_length

                    # deal with bad position
                    if self.ybot_t > self.height:
                        out_string = "Top of lane " + self.lane + " position " + self.pos + " may be a bad position. no kymograph will be generated on it."
                        print(out_string)
                        self.bad_pos[0] = 1

                # For bottom, need a new attribute to store the original index
                if self.spatial != 0:  # both have bottom
                    self.ybot_b = max(np.where(self.spread > 0.2)[0])  # leave some space
                    # to have consistent height
                    self.ytop_b = self.ybot_b - self.trench_length

                    # deal with bad position
                    if self.ytop_b < 0:
                        out_string = "Bottom of lane " + self.lane + " position " + self.pos + " may be a bad position. no kymograph will be generated on it."
                        print(out_string)
                        self.bad_pos[1] = 1
                    self.bottom_cut = self.ytop

            if self.spatial != 1:  # both have top
                cim = Image.fromarray(im_i[self.ytop:self.ybot, :].astype(np.uint8))
                # get file name
                if self.is_stack:
                    new_name = self.file_list[0].split('/')[-1]
                    new_name = 'Top_' + new_name.split('.')[0] + '_Time_' + str(i).zfill(3) + '.tiff'
                else:
                    new_name = 'Top_' + self.file_list[i].split('/')[-1]
                cim.save(os.path.join(cropped_path, new_name))

            if self.spatial != 0:  # both have bottom
                cim = Image.fromarray(im_i[self.ytop:self.ybot, :].astype(np.uint8))
                # get file name
                if self.is_stack:
                    new_name = self.file_list[0].split('/')[-1]
                    new_name = 'Bottom_' + new_name.split('.')[0] + '_Time_' + str(i).zfill(3) + '.tiff'
                else:
                    new_name = 'Bottom_' + self.file_list[i].split('/')[-1]
                cim.save(os.path.join(cropped_path, new_name))
        return

    def mask_all_trenches(self):
        self.cropped_path = self.file_path + "/cropped"
        if self.spatial != 1: # top
            self.get_file_list(file_path=self.cropped_path, spatial='Top')
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
            out_file = "Top_rough_mask.tiff"
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
            out_file = "Top_small_particle_removed.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_tip.tobytes())
            out.save(out_file)
            self.im_projected = im_tip
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
            out_file = "Top_dilated_mask.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
            out.save(out_file)

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
            out_file = "Top_dilated_mask_after_closing_down.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
            out.save(out_file)

            # vertical dilation up
            structure = np.zeros((5, 5))
            structure[0, 2] = 1
            structure[1, 2] = 1
            structure[2, 2] = 1
            im_dilated = binary_dilation(im_dilated, structure=structure, iterations=4)
            im_dilated = (255 * im_dilated).astype(np.int8)
            out_file = "Top_dilated_mask_after_closing_down_up.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
            out.save(out_file)

            # find binding box
            trench_ccomp = label(im_dilated)
            self.reg_props = regionprops(trench_ccomp)
            self.bbox_list = []
            for reg in self.reg_props:
                reg_width = reg.bbox[3] - reg.bbox[1]
                # if the identified box is really a trench
                if self.trench_width * 0.7 < reg_width < self.trench_width * 1.3:
                    self.bbox_list.append(reg.bbox)

            self.bbox_list.sort(key=lambda x: x[1])
            self.bbox_list = [(int(a + self.ytop_t), self.ybot_t, int(b), int(d)) for a, b, c, d in self.bbox_list]

            # exclude edges
            most_left = self.bbox_list[0]
            most_right = self.bbox_list[-1]
            if most_left[2] == 0:
                self.bbox_list = self.bbox_list[1:]
            if most_right[3] == self.width:
                self.bbox_list = self.bbox_list[:-1]
            self.bbox_dict[0] = self.bbox_list

        if self.spatial != 0: # bottom
            self.get_file_list(file_path=self.cropped_path, spatial='Bottom')
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
            out_file = "Top_rough_mask.tiff"
            out = PIL.Image.frombytes("L", (self.width, self.height), self.im_projected.tobytes())

            out.save(out_file)
            # only analysis the bot
            sub_height = 50
            im_bot = self.im_projected[self.height - sub_height:, :]
            # remove small elements
            bot_trench = label(im_bot)
            reg_props = regionprops(bot_trench)
            for reg in reg_props:
                if reg.area < 50:
                    reg_loc = reg.bbox
                    filled_reg = np.zeros((reg_loc[2] - reg_loc[0], reg_loc[3] - reg_loc[1]))
                    bot_trench[reg_loc[0]:reg_loc[2], reg_loc[1]:reg_loc[3]] = filled_reg
            out_file = "Bottom_small_particle_removed.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_bot.tobytes())
            out.save(out_file)
            self.im_projected = im_bot
            # vertical dilation
            structure = np.zeros((9, 9))
            # structure[4, 4] = 1
            # structure[5, 4] = 1
            # structure[6, 4] = 1
            # structure[7, 4] = 1
            # structure[8, 4] = 1
            structure[0, 4] = 1
            structure[1, 4] = 1
            structure[2, 4] = 1
            structure[3, 4] = 1
            structure[4, 4] = 1


            im_dilated = binary_dilation(im_bot, structure=structure, iterations=200)
            im_dilated = (255 * im_dilated).astype(np.int8)
            out_file = "Bottom_dilated_mask.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
            out.save(out_file)

            # vertical dilation up
            structure = np.zeros((5, 5))
            # structure[0, 2] = 1
            # structure[1, 2] = 1
            # structure[2, 2] = 1
            structure[2, 2] = 1  #down
            structure[3, 2] = 1
            structure[4, 2] = 1
            im_dilated = binary_dilation(im_dilated, structure=structure, iterations=4)
            im_dilated = (255 * im_dilated).astype(np.int8)
            out_file = "Bottom_dilated_mask_after_closing_down_up.tiff"
            out = PIL.Image.frombytes("L", (self.width, sub_height), im_dilated.tobytes())
            out.save(out_file)

            # find binding box
            trench_ccomp = label(im_dilated)
            self.reg_props = regionprops(trench_ccomp)
            self.bbox_list = []
            for reg in self.reg_props:
                reg_width = reg.bbox[3] - reg.bbox[1]
                # if the identified box is really a trench
                if self.trench_width * 0.7 < reg_width < self.trench_width * 1.3:
                    self.bbox_list.append(reg.bbox)

            self.bbox_list.sort(key=lambda x: x[1])
            # self.bbox_list = [(int(a + self.ytop), self.ybot, int(b), int(d)) for a, b, c, d in self.bbox_list]

            self.bbox_list = [(int(c + self.bottom_cut-self.trench_length), int(c + self.bottom_cut), int(b), int(d))
                              for a, b, c, d in self.bbox_list]

            # exclude edges
            most_left = self.bbox_list[0]
            most_right = self.bbox_list[-1]
            if most_left[2] == 0:
                self.bbox_list = self.bbox_list[1:]
            if most_right[3] == self.width:
                self.bbox_list = self.bbox_list[:-1]
            self.bbox_dict[1] = self.bbox_list

        # save box info:
        self.box_info = []
        if self.spatial !=1:
            top_box = self.bbox_dict[0]
            h5_name_top = str(self.seg_channel) + "_lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_top.h5"
            self.box_info.append(h5_name_top)

            hf_t = h5py.File(h5_name_top, 'w')
            hf_t.create_dataset('box', data=top_box)

            hf_t.close()
        if self.spatial !=0:
            bot_box = self.bbox_dict[1]
            h5_name_bot = str(self.seg_channel) + "_lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_bottom.h5"
            self.box_info.append(h5_name_bot)

            hf_b = h5py.File(h5_name_bot, 'w')
            hf_b.create_dataset('box', data=bot_box)
            hf_b.close()
        return

    def clean_up(self):
        shutil.rmtree(self.enhanced_path)

        return

    def run_kymo_phase(self):
        self.get_file_list()
        self.background_enhance()
        self.auto_crop()
        self.mask_all_trenches()
        self.kymograph()
        if self.clean:
            self.clean_up()
        return

    def kymograph(self):
        if self.seg_channel != 'BF':
            if self.box_info is None:
                self.box_info = []
                if self.spatial == 2:
                    h5_name_top = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_top.h5"
                    self.box_info.append(h5_name_top)
                    h5_name_bottom = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_bottom.h5"
                    self.box_info.append(h5_name_bottom)
                else:
                    local = ['top', 'bottom']
                    h5_name = "Lane_" + str(self.lane).zfill(2) + "_pos_" + str(self.pos).zfill(3) + "_" + local[
                        self.spatial] + ".h5"
                    self.box_info.append(h5_name)
        os.chdir(self.file_path)

        kymo_path = os.path.join(self.main_path, self.prefix)
        kymo_path = kymo_path + "/Lane_" + str(self.lane)
        if not os.path.exists(kymo_path):
            os.makedirs(kymo_path)

        kymo_path = kymo_path + "/pos_" + str(self.pos)
        if not os.path.exists(kymo_path):
            os.makedirs(kymo_path)

        kymo_path_stack = os.path.join(kymo_path, "Stack")
        kymo_path_kymo = os.path.join(kymo_path, "Kymograph")
        if not os.path.exists(kymo_path_stack):
            os.makedirs(kymo_path_stack)
        if not os.path.exists(kymo_path_kymo):
            os.makedirs(kymo_path_kymo)

        for ii in range(len(self.box_info)):
            hf = h5py.File(self.box_info[ii], 'r')
            if self.seg_channel !='BF':
                ind_list = hf.get('box').value
                upper_index = hf.get('upper_index').value
                lower_index = hf.get('lower_index').value + 20
                hf.close()
                trench_num = len(ind_list)
                if trench_num > 0:
                    all_kymo = {}
                    for t_i in range(trench_num):
                        all_kymo[t_i] = np.zeros((len(self.file_list), lower_index - upper_index, self.trench_width))
                    # file_list = ori_files[self.frame_start:self.frame_limit]
                    for f_i in range(len(self.file_list)):
                        try:
                            file_i = self.file_list[f_i]
                        except:
                            print("something is wrong")
                            continue

                        im_t = pl.imread(file_i)
                        if self.drift_correct == 1:
                            # correct for drift
                            move_x = self.drift_x[f_i]
                            move_y = self.drift_y[f_i]
                        else:
                            move_x = 0
                            move_y = 0
                        for t_i in range(trench_num):

                            trench_left, trench_right = ind_list[t_i]
                            trench = np.zeros((lower_index - upper_index, self.trench_width))
                            trench[:,:max(0, trench_left+move_x)+self.trench_width] = im_t[upper_index+move_y:lower_index+move_y, max(0, trench_left+move_x):max(0, trench_left+move_x)+self.trench_width]
                            all_kymo[t_i][f_i] = trench.astype(np.uint16)

                    for t_i in range(trench_num):
                        if "_top" in self.box_info[ii]:  # top trench
                            if self.saving_option !=0:  # save kymo
                                trench_name = kymo_path_kymo + "/Kymo_Lane_" + str(self.lane).zfill(
                                    2) + "_pos_" + str(
                                    self.pos).zfill(3) + "_trench_" + str(t_i + 1).zfill(2) + "_top_c_"+ self.channel+".tiff"
                            if self.saving_option !=1: # save stacks
                                trench_name_stack = kymo_path_stack + "/Stack_Lane_" + str(self.lane).zfill(
                                    2) + "_pos_" + str(
                                    self.pos).zfill(3) + "_trench_" + str(t_i + 1).zfill(2) + "_top_c_" + self.channel + ".tiff"
                        else:   # bottom trench
                            if self.saving_option != 0:  # save kymo
                                trench_name = kymo_path + "/Kymo_Lane_" + str(self.lane).zfill(
                                    2) + "_pos_" + str(
                                    self.pos).zfill(3) + "_trench_" + str(t_i + 1).zfill(2) + "_bottom_c_"+ self.channel+".tiff"
                            if self.saving_option != 1: # save stack
                                trench_name_stack = kymo_path + "/Stack_Lane_" + str(self.lane).zfill(
                                    2) + "_pos_" + str(
                                    self.pos).zfill(3) + "_trench_" + str(t_i + 1).zfill(
                                    2) + "_bottom_c_" + self.channel + ".tiff"
                        # TODO
                        if self.saving_option != 1:  # save stacks
                            imsave(trench_name_stack,all_kymo[t_i].astype(np.uint16))
                        if self.saving_option != 0:  # save kymo
                            this_kymo = np.concatenate(all_kymo[t_i], axis=1).astype(np.uint16)
                            out = PIL.Image.frombytes("I;16", (this_kymo.shape[1], this_kymo.shape[0]), this_kymo.tobytes())
                            out.save(trench_name)
                        all_kymo[t_i] = None
                else:
                    print("no trenches detected")
            else: # segmented with phase
                if self.spatial !=1: #top
                    if not self.bad_pos[0]:
                        self.bbox_list = self.bbox_dict[0]
                        trench_num = len(self.bbox_list)
                        trench_dict = {}
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
                                trench_dict[t_i][f_i] = im_t[cur_box[0]:cur_box[1], cur_box[2]:cur_box[3]]

                        for t_i in range(trench_num):
                            # save stack
                            if self.saving_option != 1:
                                trench_stack_name = kymo_path + "/Stack_Lane_" + str(self.lane).zfill(2) + "_pos_" + str(
                                    self.pos).zfill(
                                    3) + "_trench_" + str(t_i + 1).zfill(2) + "_channel_" + self.channel + "_top.tiff"

                                imsave(trench_stack_name, trench_dict[t_i].astype(np.uint16))
                            # save kymo
                            if self.saving_option != 0:
                                trench_kymo_name = kymo_path + "/Kymo_Lane_" + str(self.lane).zfill(2) + "_pos_" + str(
                                    self.pos).zfill(
                                    3) + "_trench_" + str(t_i + 1).zfill(2) + "_channel_" + self.channel + "_top.tiff"
                                this_kymo = np.concatenate(trench_dict[t_i], axis=1).astype(np.uint16)

                                out = PIL.Image.frombytes("I;16", (this_kymo.shape[1], this_kymo.shape[0]), this_kymo.tobytes())
                                out.save(trench_kymo_name)
                            trench_dict[t_i] = None

                if self.spatial !=0: #bottom
                    if not self.bad_pos[1]:
                        self.bbox_list = self.bbox_dict[1]
                        trench_num = len(self.bbox_list)
                        trench_dict = {}
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
                                trench_dict[t_i][f_i] = im_t[cur_box[0]:cur_box[1], cur_box[2]:cur_box[3]]

                        for t_i in range(trench_num):
                            # save stack
                            if self.saving_option != 1:
                                trench_stack_name = kymo_path + "/Stack_Lane_" + str(self.lane).zfill(2) + "_pos_" + str(
                                    self.pos).zfill(
                                    3) + "_trench_" + str(t_i + 1).zfill(2) + "_channel_" + self.channel + "_bottom.tiff"

                                imsave(trench_stack_name, trench_dict[t_i].astype(np.uint16))
                            # save kymo
                            if self.saving_option != 0:
                                trench_kymo_name = kymo_path + "/Kymo_Lane_" + str(self.lane).zfill(2) + "_pos_" + str(
                                    self.pos).zfill(
                                    3) + "_trench_" + str(t_i + 1).zfill(2) + "_channel_" + self.channel + "_bottom.tiff"
                                this_kymo = np.concatenate(trench_dict[t_i], axis=1).astype(np.uint16)

                                out = PIL.Image.frombytes("I;16", (this_kymo.shape[1], this_kymo.shape[0]),
                                                          this_kymo.tobytes())
                                out.save(trench_kymo_name)
                            trench_dict[t_i] = None
        return

    def run_kymo(self):
        if self.seg_channel != 'BF':
            self.get_file_list()
            # identify
            if self.find_correct == 1:
                self.find_drift()
            else:
                if self.drift_correct == 1:
                    self.read_drift()
                if self.channel == self.seg_channel:
                    self.get_trenches()
                    self.kymograph()
                else:
                    self.kymograph()
        else:
            self.run_kymo_phase()
        return

    @staticmethod
    def to_8_bit(im):
        im_min = im.min()
        im_max = im.max()
        scaling_factor = (im_max - im_min)
        im = (im - im_min)
        im = (im * 255. / scaling_factor).astype(np.uint8)
        return im

    @staticmethod
    def moveImage(im, move_x, move_y, pad=0):
        """
        Moves the image without changing frame dimensions, and
        pads the edges with given value (default=0).
        """
        im_new = np.ones((im.shape[0], im.shape[1]), dtype=np.uint16) * pad
        xbound = im.shape[1]
        ybound = im.shape[0]
        if move_x >= 0:
            im_new[:, move_x:] = im[:, :xbound - move_x]
        else:
            im_new[:, :xbound + move_x] = im[:, -move_x:]
        if move_y >= 0:
            im_new[move_y:, :] = im[:ybound - move_y, :]
        else:
            im_new[:ybound + move_y, :] = im[-move_y:, :]
        # print(im == im_new)
        return im_new

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

    @staticmethod
    def pairwise_list_align(list_a, list_b, max_gap):
        # print(list_b)
        # print(max_gap)
        shift = 0
        matches = 0
        i_b = 0
        len_b = len(list_b)
        # only consider middle
        list_a = list_a[1:-1]
        for x in list_a:
            found = 0
            while (not found) and (i_b < len_b):
                # print("list_b ", list_b[i_b])
                # print("list_a ", x)
                diff = list_b[i_b] - x

                if diff < -max_gap:
                    i_b += 1
                    len_b -= 1
                elif diff > max_gap:  # this cell is lost
                    break
                else:
                    found = 1
                    shift += diff
                    matches += 1
                    i_b += 1  # don't compare with the matched one for the next cell
                    len_b -= 1

        if matches:
            shift = shift * 1. / matches

        return shift

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

    # (self, nd2_file, main_directory, lane, pos, channel, seg_channel, spatial,trench_length=None, trench_width=None,
    #                  drift_correct=0, find_correct=0, frame_start=None, frame_limit=None, output_dir=None,
    #                  box_info=None, saving_option = 0, clean_up=1, chip_lenght=None, chip_width=None, magnification = None)
    def run_kymo_generator(nd2_file, main_directory, lanes, poses, other_channels, seg_channel,  trench_length, trench_width,
                           spatial, drift_correct, frame_start=None, frame_limit=None, output_dir=None, box_info=None,
                           saving_option = 0, clean_up=1, chip_length=None, chip_width=None, magnification = None):

        start_t = datetime.now()
        print('Kymo starts ')
        # drift correct for each lane:
        for lane in lanes:
            pos = poses[0]
            channel = seg_channel
            find_correct = 1
            new_kymo = trench_kymograph(nd2_file, main_directory, lane, pos, channel, seg_channel,spatial, trench_length,
                                        trench_width,  drift_correct, find_correct, frame_start, frame_limit,
                                        output_dir, box_info, saving_option, clean_up, chip_length, chip_width, magnification)
            new_kymo.run_kymo()

        # trench identify for each pos
        for lane in lanes:
            channel = seg_channel
            def helper_kymo(p):
                new_kymo = trench_kymograph(nd2_file, main_directory, lane, pos, channel, seg_channel,spatial, trench_length,
                                            trench_width,  drift_correct, find_correct, frame_start, frame_limit,
                                            output_dir, box_info, saving_option, clean_up, chip_length, chip_width, magnification)
                new_kymo.run_kymo()
                return 0

            cores = multiprocessing.cpu_count()
            jobs = []
            batch_num = len(poses)/cores + 1

            for i in range(batch_num):
                start_ind = i * cores
                end_ind   = start_ind + cores
                partial_poses = poses[start_ind:end_ind]

                for p in partial_poses:
                    j = multiprocessing.Process(target=helper_kymo, args=(p,))
                    jobs.append(j)
                    j.start()
                    print(p, j.pid)

                for job in jobs:
                    print(job.pid)
                    job.join()


        for lane in lanes:
            for channel in other_channels:
                print("lane channel ", lane, channel)
                def helper_kymo(p):
                    new_kymo = trench_kymograph(nd2_file, main_directory, lane, pos, channel, seg_channel,spatial, trench_length,
                                                trench_width,  drift_correct, find_correct, frame_start, frame_limit, output_dir,
                                                box_info, saving_option, clean_up, chip_length, chip_width, magnification)
                    new_kymo.run_kymo()

                cores = multiprocessing.cpu_count()
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
                        print(p, j.pid)

                    for job in jobs:
                        print(job.pid)
                        job.join()

            print("woohoo")

        time_elapsed = datetime.now() - start_t
        print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))



    nd2_file = "PERSISTER_GC_LONG_STATIONARY_PART2.nd2"
    #
    file_directory = r"/Volumes/SysBio/PAULSSON LAB/Leoncini/DATA_Ti4/20180705--PERSISTER_GC/"
    new_extractor = ND2_extractor(nd2_file, file_directory)
    new_extractor.run_extraction()
    #
    # # TODO: Change me
    # nd2_file = "20180402--HYSTERESIS_ROBOT_RUN--Recovery001.nd2"
    # main_directory =r"/Volumes/SysBio/PAULSSON LAB/Leoncini/DATA_Ti3//20180402--HYSTERESIS_ROBOT_RUN"
    #
    #
    # lanes = [1] # has to be a list
    # poses = range(2,66)  # second value exclusive
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
    # drift_correct = 0 # 1 for need correction, 0 for no
    # saving_option = 0
    # clean_up = 1
    # frame_start = None
    # frame_limit = None
    #
    # # TODO: Don't touch me!
    # run_kymo_generator(nd2_file, main_directory, lanes, poses, other_channels, seg_channel,  trench_length, trench_width,
    #                            spatial, drift_correct, frame_start, frame_limit, output_dir, box_info,
    #                            saving_option, clean_up, chip_length, chip_width, magnification)
    #

