# ND2 extractor
# author: Suyang Wan, Paulsson Lab, Harvard
# product manager: Emanuele Leoncini, Somenath Bakshi
# Special thanks for technical support: Carlos Sanchez, Sadik Yidik
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



#
# # Todo: create a GUI
import matplotlib.pyplot as pl
import glob  # pathname pattern
from PIL import Image
import nd2reader
import os
import cv2
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
    def __init__(self, nd2_file, file_directory, xml_file=None, xml_dir=None, output_path=None, frame_start=None, frame_end=None,
                 lanes_to_extract=None,channels_to_extract=None):
        os.chdir(file_directory)
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
        self.single_pos = False
        self.channels = None
        self.frames = None
        self.lanes = None
        self.poses_to_extract = None



        self.frame_start = frame_start
        self.frame_end = frame_end
        self.lanes_to_extract = lanes_to_extract   # intermediate variables
        self.channels_to_extract = channels_to_extract
        self.nd2 = nd2reader.Nd2(self.nd2_f)    # for extraction iter
        self.nd2_new = ND2_Reader(self.nd2_file)  # for lane info iter

    def channel_info(self):
        self.channels = self.nd2.channels


    def lane_info(self):   # condition infos too
        lane_dict = {}
        lane_dict[0] = 1
        pos_offset = {}
        cur_lane = 1
        pos_min = 0
        pos_offset[cur_lane] = pos_min - 1
        if 'm' in self.nd2_new.axes:
            self.nd2_new.iter_axes = 'm'
            y_prev = self.nd2_new[0].metadata['y_um']
            self.lanes = len(self.nd2_new)
            for i in range(1, self.lanes):
                f = self.nd2_new[i]
                y_now = f.metadata['y_um']
                if abs(y_now - y_prev) > 200:  # a new lane
                    cur_lane += 1
                    pos_min = i - 1
                    pos_offset[cur_lane] = pos_min
                lane_dict[i] = cur_lane
                y_prev = y_now
            self.lanes = cur_lane
            self.nd2_new.close()
        else:
            self.lanes = 1
            self.single_pos = True      # TODO: maybe unnecessary
        self.lane_dict = lane_dict
        self.pos_offset = pos_offset


        ## TODO

    #
    # def pos_info(self):
    #     cur_dir = os.getcwd()
    #     os.chdir(self.xml_dir)
    #     tree = ET.ElementTree(file=self.xml_file)
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
    #                 lane_name_cur = re.match(r'\w', pos_name).group()
    #             if lane_name_cur != lane_name_prev:
    #                 lane_name_prev = lane_name_cur
    #                 lane_count += 1
    #                 pos_offset[lane_count] = ind - 1
    #             lane_dict[ind] = lane_count
    #             pos_dict[ind] = pos_name
    #     os.chdir(cur_dir)
    #     self.pos_dict = pos_dict
    #     self.lane_dict = lane_dict
    #     self.pos_offset = pos_offset


    def select_cond(self):
        # channels
        if self.channels_to_extract is None:
            self.channels_to_extract = [str(x) for x in self.channels]
        # lanes
        if self.lanes_to_extract is None:
            self.poses_to_extract = list(range(self.lanes))
        else:
            self.poses_to_extract = []
            for lane in self.lanes_to_extract:
                self.poses_to_extract += list(self.pos_dict[lane])
        # frames
        self.nd2_new.iter_axes = 't'
        self.frames = len(self.nd2_new)
        if self.frame_start is None:
            self.frame_start = 0
        if self.frame_end is None:
            self.frame_end = self.frames - 1


    def tiff_extractor(self, pos):

        if self.pos_dict:
            new_dir = self.main_dir + "/Lane_" + str(self.lane_dict[pos]).im(2) + "/" + self.pos_dict[pos] + "/"
        else:
            lane_ind = self.lane_dict[pos]
            pos_off = self.pos_offset[lane_ind]
            new_dir = self.main_dir + "/Lane_" + str(lane_ind).zfill(2) + "/pos_" + str(pos - pos_off).zfill(3) + "/"

        # create a folder for each position
        try:
            os.makedirs(new_dir)
        except OSError:
            pass
        os.chdir(new_dir)

        if self.pos_dict:
            meta_name = self.nd2_file_name + "_" + self.pos_dict[pos] + "_Time_"
        else:
            meta_name = self.nd2_file_name + "_pos_" + str(pos - pos_off).zfill(3) + "_Time_"

        for image in self.nd2.select(fields_of_view=pos, channels=self.channels_to_extract,start=self.frame_start,stop=self.frame_end):
            channel = image._channel

            channel = channel.encode('ascii', 'ignore')
            channel = str(channel.decode("utf-8"))
            # channel = str(channel.encode('ascii', 'ignore'))
            # experimental, may not work

            time_point = image.frame_number
            tiff_name = meta_name + str(time_point).zfill(4) + "_c_" + channel + ".tiff"

            # save file in 16-bit
            # thanks to http://shortrecipes.blogspot.com/2009/01/python-python-imaging-library-16-bit.html
            image = image.base.astype(np.uint16)
            out = PIL.Image.frombytes("I;16", (image.shape[1], image.shape[0]), image.tobytes())
            out.save(tiff_name)

        os.chdir(self.file_dir)

    def run_extraction(self,):
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
        # os.chdir(self.input_path)
        self.channel_info()


        # switch to another ND2reader for faster iterations
        # nd2 = nd2reader.Nd2(self.nd2_file)

        main_dir = self.input_path + "/" + self.nd2_file_name
        try:
            os.makedirs(main_dir)
        except OSError:
            pass

        # parallelize extraction
        # poses = nd2.fields_of_view
        self.select_cond()
        poses = self.poses_to_extract
        cores = pathos.multiprocessing.cpu_count()
        print(poses, cores)
        pool = pathos.multiprocessing.Pool(cores)
        pool.map(self.tiff_extractor, poses)

        time_elapsed = datetime.now() - start_t
        print('Time elapsed for extraction (hh:mm:ss.ms) {}'.format(time_elapsed))








#
# file_directory = r"/Volumes/SysBio/PAULSSON LAB/Carlos/Data_Ti3/Burdenless dyes/12232018/"
# nd2_file = "Barcodes_HADA_RADA_WT.nd2"
# #
# # EXTRACTOR
# new_extractor = ND2_extractor(nd2_file, file_directory)
# new_extractor.run_extraction()

# extractor par
file_directory = r"/Volumes/SysBio/PAULSSON LAB/Carlos/Data_Ti3/Burdenless dyes/12232018"
nd2_file = "HADA_RADA_WT.nd2"
#
# EXTRACTOR
new_extractor = ND2_extractor(nd2_file, file_directory)
new_extractor.run_extraction()