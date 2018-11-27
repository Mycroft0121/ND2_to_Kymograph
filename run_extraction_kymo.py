import ND2_extractor

file_directory = r"/home/paulssonlab/Desktop/Paulsson_lab/PAULSSON LAB/Somenath/DATA_Ti4/20181021"
nd2_file = "test_3LaneSnake.nd2"



frame_start=None
frame_end=None
lanes_to_extract=None
channels_to_extract=None

new_extractor = ND2_extractor.ND2_extractor(nd2_file, file_directory, frame_start, frame_end,
                 lanes_to_extract,channels_to_extract)
new_extractor.run_extraction()
