import argparse
import fc_utils
import shutil
import os
from PIL import Image
from scipy.io import loadmat
import matplotlib.image as mpimg
from scipy import fftpack
import numpy as np
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test for face recognition."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        # required=True,
        default='/storage4tb/PycharmProjects/Datasets/fc_captures_output/test',
        help="Path to input dataset."
    )

    parser.add_argument(
        "--out_path",
        type=str,
        # required=True,
        default='/storage4tb/PycharmProjects/Datasets/fc_captures_lensless/test',
        help="Path to output directory."
    )

    parser.add_argument(
        "--noise_locs",
        type=str,
        default='../data/noise_locations/noise_10_pixels_per_block.npy',
        help="Path to noise locations .npy file."
    )

    parser.add_argument(
        "--calib_file",
        type=str,
        default='../data/flatcam_calibdata.mat',
        help="Path to the calibation file (flatcam_calibdata.mat) required for processing raw sensor measurements."
    )

    parser.add_argument(
        "--meas_size",
        type=int,
        default=128,
        help="Measurement size to use. Default: 128 -> 128x128"
    )

    parser.add_argument(
        "--num_of_cores",
        type=int,
        default=-1,
        help="Number of cores to use to process the data. Default: -1 -> Uses all cores."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    out_path = args.out_path
    calib_file = args.calib_file
    meas_size = (args.meas_size, args.meas_size) # (256,256)
    num_of_cores = os.cpu_count() if args.num_of_cores == -1 else args.num_of_cores
    positions = np.load(args.noise_locs) # ../pos_10_6432.npy

    dirs = os.listdir(data_path)
    two_split = np.array_split(dirs, num_of_cores)
    pths = []
    for array in two_split:
        pths.append(list(array))

    Parallel(n_jobs=num_of_cores, prefer="threads")(delayed(fc_utils.process_raw_fc_meas)(x, data_path, calib_file, out_path, meas_size, positions) for x in pths)


