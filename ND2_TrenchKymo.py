import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np

import re
import glob    # pathname pattern
import os
import json
from PIL import Image
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
from multiprocessing import Pool


# from ND2 extractor

import nd2reader
import os
import shutil
import PIL
import numpy as np
from pims import ND2_Reader
import multiprocessing
import functools
import xml.etree.cElementTree as ET
import re

from datetime import datetime




# step 1, extract ND2 and save as stacks
class ND2_extractor():
    def __init__(self, nd2_file, file_directory, xml_file=None, xml_dir=None, output_path=None):
        self.input_path    = file_directory
        self.nd2_file      = nd2_file
        self.nd2_file_name = nd2_file[:-4]
        self.xml_file      = xml_file
        self.xml_dir       = xml_dir
        self.output_path   =  output_path
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
            new_dir = self.main_dir + "/Lane_" + str(self.lane_dict[pos]) + "/" + self.pos_dict[pos] + "/"
        else:
            lane_ind = self.lane_dict[pos]
            pos_off = self.pos_offset[lane_ind]
            new_dir = self.main_dir + "/Lane_" + str(lane_ind) + "/pos_" + str(pos - pos_off) + "/"

        # create a folder for each position
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, 0755)
        os.chdir(new_dir)

        if self.pos_dict:
            meta_name = self.nd2_file_name + "_" + self.pos_dict[pos] + "_t"
        else:
            meta_name = self.nd2_file_name + "_pos_" + str(pos - pos_off) + "_t"

        for image in nd2.select(fields_of_view=pos):
            channel = image._channel
            channel = str(channel.encode('ascii', 'ignore'))
            time_point = image.frame_number
            if time_point < 10:
                time_point = "0" + str(time_point)
                time_point = str(time_point)
            tiff_name = meta_name + str(time_point) + "_c_" + channel + ".tiff"

            # save file in 16-bit
            # thanks to http://shortrecipes.blogspot.com/2009/01/python-python-imaging-library-16-bit.html
            image = image.base.astype(np.uint16)
            out = PIL.Image.frombytes("I;16", (image.shape[1], image.shape[0]), image.tobytes())
            out.save(tiff_name)

        os.chdir(self.file_dir)



    def run_extraction(self):
        start_t = datetime.now()

        os.chdir(self.file_directory)
        # get position name if xml is available
        if self.xml_file:
            if not self.xml_dir:
                self.xml_dir = self.file_directory
                ND2_extractor.pos_info(self)
            #pos_dict, lane_dict, pos_offset = pos_info(xml_file, xml_dir)
        # otherwise get lane info from y_um
        else:
            ND2_extractor.lane_info(self.nd2_file)
        os.chdir(self.file_directory)

        # switch to another ND2reader for faster iterations
        # nd2_file_name = nd2_file[:-4]
        nd2 = nd2reader.Nd2(self.nd2_file)

        main_dir = self.file_directory + "/" + self.nd2_file_name
        if not os.path.exists(main_dir):
            os.makedirs(main_dir, 0755)

        # parallelize extraction
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        poses = nd2.fields_of_view
        pool.map(ND2_extractor.tiff_extractor, poses)

        time_elapsed = datetime.now() - start_t
        print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))

    # # get position name if xml is available
    # if xml_file:
    #     if not xml_dir:
    #         xml_dir = file_directory
    #     pos_dict, lane_dict, pos_offset = pos_info(xml_file, xml_dir)
    # # otherwise get lane info from y_um
    # else:
    #     lane_dict, pos_offset = lane_info(nd2_file)
    # os.chdir(file_directory)
    #
    # # switch to another ND2reader for faster iterations
    # nd2_file_name = nd2_file[:-4]
    # nd2 = nd2reader.Nd2(nd2_file)
    #
    # main_dir = file_directory+"/"+nd2_file_name
    # if not os.path.exists(main_dir):
    #     os.makedirs(main_dir, 0755)
    #
    # # parallelize extraction
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # poses = nd2.fields_of_view
    # pool.map(tiff_extractor, poses)







class trench_kymograph():
    def __init__(self, path, lane, channel, time, fov, bit_info):
        self.


    # generate stacks for each fov, find the max intensity
    def get_trenches(self, stacks):


        # find the max intensity for all stacks

        # intensity scanning to find the box containing each trench

        # return a list of box coordinates


    def kymograph(self, coordinates):
        # cut each fov with the coordinates
        # generate kymograph from it








# def ND2extractor(nd2_file, file_directory, xml_file=None, xml_dir=None):
#
#     start_t = datetime.now()
#
#
#     os.chdir(file_directory)
#
#     # declare global variables
#     global main_dir
#     global nd2_file_name
#     global nd2_f
#     global file_dir
#     global pos_dict
#     global pos_offset
#     global lane_dict
#
#     pos_dict = None
#     pos_offset = None
#     file_dir = file_directory
#     nd2_f = nd2_file
#
#     # get position name if xml is available
#     if xml_file:
#         if not xml_dir:
#             xml_dir = file_directory
#         pos_dict, lane_dict, pos_offset = pos_info(xml_file, xml_dir)
#     # otherwise get lane info from y_um
#     else:
#         lane_dict, pos_offset = lane_info(nd2_file)
#     os.chdir(file_directory)
#
#     # switch to another ND2reader for faster iterations
#     nd2_file_name = nd2_file[:-4]
#     nd2 = nd2reader.Nd2(nd2_file)
#
#     main_dir = file_directory+"/"+nd2_file_name
#     if not os.path.exists(main_dir):
#         os.makedirs(main_dir, 0755)
#
#     # parallelize extraction
#     cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=cores)
#     poses = nd2.fields_of_view
#     pool.map(tiff_extractor, poses)
#
#
#     time_elapsed = datetime.now()-start_t
#     print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))
#
#
# # function to extract lane information
# def lane_info(nd2_file):
#         # dict for lane info
#     nd2_new = ND2_Reader(nd2_file)
#     nd2_new.iter_axes = 'm'
#     lane_dict = {}
#     lane_dict[0] = 1
#     pos_offset = {}
#     cur_lane = 1
#     pos_min  = 0
#     pos_offset[cur_lane] = pos_min - 1
#     y_prev = nd2_new[0].metadata['y_um']
#     pos_num = len(nd2_new)
#     for i in range(1, pos_num):
#         f = nd2_new[i]
#         y_now = f.metadata['y_um']
#         if abs(y_now - y_prev) > 200:  # a new lane
#             cur_lane += 1
#             pos_min = i - 1
#             pos_offset[cur_lane] = pos_min
#         lane_dict[i] = cur_lane
#         y_prev = y_now
#     nd2_new.close()
#     return lane_dict, pos_offset
#
#
# def pos_info(xml_file, xml_dir):
#     cur_dir = os.getcwd()
#     os.chdir(xml_dir)
#     tree = ET.ElementTree(file=xml_file)
#     root = tree.getroot()[0]
#     pos_dict = {}
#     lane_dict = {}
#     pos_offset = {}
#     lane_count = 0
#     lane_name_prev = None
#     dummy_count = 0
#     for i in root:
#         if i.tag.startswith('Point'):
#             ind = int(i.tag[5:])
#             pos_name = i[1].attrib['value']
#             if len(pos_name) < 1:
#                 pos_name = "dummy_" + str(dummy_count)
#                 dummy_count += 1
#                 lane_name_cur = "dummy"
#             else:
#                 lane_name_cur = re.match(r'\w',pos_name).group()
#             if lane_name_cur != lane_name_prev:
#                 lane_name_prev = lane_name_cur
#                 lane_count +=1
#                 pos_offset[lane_count] = ind - 1
#             lane_dict[ind] = lane_count
#             pos_dict[ind] = pos_name
#     os.chdir(cur_dir)
#
#     return pos_dict, lane_dict, pos_offset
#
#
# def tiff_extractor(pos):
#     nd2 = nd2reader.Nd2(nd2_f)
#     if pos_dict:
#         new_dir = main_dir+"/Lane_" +str(lane_dict[pos])+ "/"+pos_dict[pos] + "/"
#     else:
#         lane_ind = lane_dict[pos]
#         pos_off  = pos_offset[lane_ind]
#         new_dir = main_dir + "/Lane_" + str(lane_ind) + "/pos_" + str(pos - pos_off) + "/"
#
#     # create a folder for each position
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir, 0755)
#     os.chdir(new_dir)
#
#     if pos_dict:
#         meta_name = nd2_file_name + "_" + pos_dict[pos] + "_t"
#     else:
#         meta_name = nd2_file_name+"_pos_" + str(pos - pos_off) + "_t"
#
#     for image in nd2.select(fields_of_view=pos):
#         channel = image._channel
#         channel = str(channel.encode('ascii', 'ignore'))
#         time_point = image.frame_number
#         if time_point < 10:
#             time_point = "0"+str(time_point)
#             time_point = str(time_point)
#         tiff_name = meta_name+str(time_point) + "_c_" + channel + ".tiff"
#
#         # save file in 16-bit
#         # thanks to http://shortrecipes.blogspot.com/2009/01/python-python-imaging-library-16-bit.html
#         image = image.base.astype(np.uint16)
#         out = PIL.Image.frombytes("I;16", (image.shape[1], image.shape[0]), image.tobytes())
#         out.save(tiff_name)
#
#     os.chdir(file_dir)
