######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the test mode means the code will run on a small dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'SUN_256bit_sigmoid_kl10k_w1.0_w0.05_g4k_sparse_iou0.01'  # dissect results folder's name TODO
CUSTOM = True                               # custom model
CODE_LEN = 256                              # qss TODO
QUANTILE = 0.005                            # the threshold used for activation
# SEG_THRESHOLD = 0.04                      # the threshold used for visualization
SCORE_THRESHOLD = 0.01                      # the threshold used for IoU score (in HTML file), qss
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part", "texture", "color", "material", "scene"]  # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color" TODO
OUTPUT_FOLDER = "result/"+MODEL             # result will be stored in this folder


########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if 'alexnet' not in MODEL or 'SUN' in MODEL:  # qss TODO
    DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_227'
    IMG_SIZE = 227
# TODO
NUM_CLASSES = 397
FEATURE_NAMES = ['conv_last']
MODEL_FILE = '/home/ouc/data1/qiaoshishi/python_codes/IBC_release/outputs/' + \
             'SUN_256bit_sigmoid_kl10k_w1.0_w0.05_g4k_sparse/checkpoints/checkpoint_00080000.pt'  # TODO
MODEL_PARALLEL = False  # false for single gpu trained model

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 8
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4  # prefetched batches
    INDEX_FILE = 'index.csv'
