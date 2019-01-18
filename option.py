import sys, os, logging, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="NAS Super-resolution")

# Directory
parser.add_argument("--data_root", type=str, default="data")
parser.add_argument("--checkpoint_root", type=str, default="checkpoint")
parser.add_argument("--result_root", type=str, default="result")
parser.add_argument("--log_root", type=str, default="log")
parser.add_argument("--data_name", type=str, help="dataset name (e.g., news)")

# Model (parameters are fixed for each quality level)
parser.add_argument("--quality", type=str, choices=("low", "medium", "high", "ultra"))

# Training
parser.add_argument("--num_batch", type=int, default=64)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--num_update_per_epoch", type=int, default=1000)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--loss_type", type=str, default="l1", choices=("l2", "l1"))
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_epoch", type=int, default=100)
parser.add_argument("--lr_decay_rate", type=float, default=0.5)
parser.add_argument("--num_valid_image", type=int, default=10, help="number of images used for validation")

#Testing
parser.add_argument("--test_num_epoch", type=int, default=100)
parser.add_argument("--test_num_batch", type=int, default=1)

# Dataset
parser.add_argument("--patch_size", type=int, default=48)
parser.add_argument("--fps", type=float, default=0.1, help="data (or image) sampling rate from an original video")
parser.add_argument("--interpolation", default="bicubic", type=str, help="interpolation method to prepare dataset (used by ffmpeg)")
parser.add_argument('--dash_hr', default=1080, type=int, help='highest resolution of DASH videos')
parser.add_argument('--dash_lr', default=[240, 360, 480, 720], nargs='+', type=int, help='low resolutions of DASH videos')

# Resource
parser.add_argument("--num_thread", type=int, default=4, help="number of threads used for loading data (used by DataLoader)")
parser.add_argument("--use_cuda", action="store_true", help="use GPU(s) for training")
parser.add_argument("--load_on_memory", action="store_true", help="load dataset on memory")

opt = parser.parse_args()
logging.basicConfig(level=logging.INFO) #Logging level

# Create a directory
opt.checkpoint_dir = os.path.join(opt.checkpoint_root, opt.data_name, opt.quality)
opt.result_dir= os.path.join(opt.result_root, opt.data_name, opt.quality)
opt.data_dir= os.path.join(opt.data_root, opt.data_name)

os.makedirs(opt.checkpoint_dir, exist_ok=True)
os.makedirs(opt.result_dir, exist_ok=True)
os.makedirs(opt.data_dir, exist_ok=True)
