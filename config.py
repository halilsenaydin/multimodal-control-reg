# Paths
PARQUET_DIR = "dataset"
BEV_DIR = "dataset_bev"
CKPT_DIR = "checkpoints"

# Dataset
SEQ_LEN = 8
STRIDE = 2
VAL_RATIO = 0.15
SEED = 42

# BEV channels: 0=height (Z), 1=intensity, 2=occupancy
BEV_MEAN = [0.041568, 0.033508389236809565, 0.036843431994471126]
BEV_STD  = [0.51043, 0.17175369593296205, 0.18837726379088296]

# DataLoader
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH = 2

# Backbone
BACKBONE = "efficientnet_b0_features_only"
PRETRAINED = True
FREEZE_BACKBONE = False

# Image stream
USE_IMAGE = True
IMG_DIR = "dataset_img_multi"
IMG_CAMERAS = ["image_front", "image_front_left", "image_front_right", "image_rear"]
IMG_BACKBONE = "efficientnet_b1"
IMG_HIDDEN = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ConvLSTM
CONVLSTM_HIDDEN = 256
CONVLSTM_KERNEL = 3
CONVLSTM_LAYERS = 1

# Ego MLP
EGO_HIDDEN = 128

# Training
LR = 3e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 30
PATIENCE = 8
LR_SCHEDULER = "cosine"
AMP = True
GRAD_ACCUM = 4
PROJ_CHANNELS = 256
GRAD_CLIP = 1.0

# Targets
TARGETS = ["brake", "throttle", "steer"]

# LiDAR
LIDAR_MAX_RANGE: float = 80.0

# Training diagnostics: 1=every epoch, N=every N epochs, 0=disabled
EVAL_TRAIN_LOSS_EVERY: int = 1
