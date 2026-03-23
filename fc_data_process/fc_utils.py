import numpy as np
import os
from scipy.ndimage import rotate as imrotate
from PIL import Image
from scipy import fftpack
from scipy.io import loadmat
import matplotlib.image as mpimg
from skimage.feature import local_binary_pattern

def fc2bayer( im, calib ):
    # split up different color channels
    r = im[1::2, 1::2]
    gb = im[0::2, 1::2]
    gr = im[1::2, 0::2]
    b = im[0::2, 0::2]
    Y = np.dstack([r, gb, gr, b])
    # rotate capture
    Y = imrotate(Y, calib['angle'], reshape=False)
    # crop usable sensor measurements
    csize = calib['cSize']
    start_row = int((Y.shape[0] - csize[0])/2)
    end_row = int(start_row + csize[0]) # omit -1 because Python indexing does not include end index
    start_col = int((Y.shape[1] - csize[1])/2)
    end_col = int(start_col + csize[1])
    Y = Y[start_row:end_row, start_col:end_col, :]
    return Y


def make_separable( Y ):
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['cSize'] = np.squeeze(calib['cSize'])
    calib['angle'] = np.squeeze(calib['angle'])


def noise_add_ymdct(positions, ymdct):
    new_image_array = np.zeros_like(ymdct)
    new_image_array[:] = ymdct[:]
    for pos in positions:
        noise = np.random.uniform(low=np.min(new_image_array), high=np.max(new_image_array), size=1)
        new_image_array[pos[0], pos[1], pos[2]*3:pos[2]*3+2] = noise
    return new_image_array


def process_raw_fc_meas(dirs, data_path, calib_file, out_path, meas_size, positions):
    total = 0
    for dir in dirs:
        temp = len(os.listdir(os.path.join(data_path, dir)))
        total = total + temp
    cnt = 0
    for folder in dirs:
        files = sorted(os.listdir(os.path.join(data_path, folder)))
        for file in files:
            filename = os.path.splitext(file)[0]
            calib = loadmat(calib_file)  # load calibration data
            clean_calib(calib)
            meas = mpimg.imread(os.path.join(data_path, folder, file))  # load flatcam measurement
            Y = fc2bayer(meas, calib)
            # Y = flatcam.make_separable(Y) # let rows and columns have 0-mean
            r = Y[:, :, 0]
            g = np.mean((Y[:, :, 1], Y[:, :, 2]), axis=0)
            b = Y[:, :, 3]
            Y_rgb = np.dstack([r, g, b]) * 255

            y_im_meas = Image.fromarray(Y_rgb.astype(np.uint8))

            y_im_meas = y_im_meas.resize(meas_size)

            y_DCT = fftpack.dct(fftpack.dct(np.array(y_im_meas, dtype=np.float32), axis=0, norm='ortho'), axis=1,
                                norm='ortho')
            y_im_DCT = Image.fromarray(y_DCT.astype(np.uint8))

            y_DCT_1 = y_DCT[0:(meas_size[1] // 2), 0:(meas_size[1] // 2), :].copy()
            y_DCT_2 = y_DCT[(meas_size[1] // 2):meas_size[1], 0:(meas_size[1] // 2), :].copy()
            y_DCT_3 = y_DCT[0:(meas_size[1] // 2), (meas_size[1] // 2):meas_size[1], :].copy()
            y_DCT_4 = y_DCT[(meas_size[1] // 2):meas_size[1], (meas_size[1] // 2):meas_size[1], :].copy()

            y_im_meas_32 = Image.fromarray(Y_rgb.astype(np.uint8)).resize((meas_size[1] // 2,meas_size[1] // 2))
            y_DCT_5 = fftpack.dct(fftpack.dct(np.array(y_im_meas_32, dtype=np.float32), axis=0, norm='ortho'), axis=1,
                                norm='ortho')

            n_arr = np.concatenate((y_DCT_1, y_DCT_2, y_DCT_3, y_DCT_4, y_DCT_5), axis=2).transpose(2, 0, 1)
            n_arr_noise = noise_add_ymdct(positions, np.concatenate((y_DCT_1, y_DCT_2, y_DCT_3, y_DCT_4, y_DCT_5), axis=2)).transpose(2, 0, 1)

            if not os.path.isdir(os.path.join(out_path, 'dct_vis', folder)):
                os.makedirs(os.path.join(out_path, 'dct_vis', folder))
            if not os.path.isdir(os.path.join(out_path, 'meas_vis', folder)):
                os.makedirs(os.path.join(out_path, 'meas_vis', folder))
            if not os.path.isdir(os.path.join(out_path, 'ymdct_npy', folder)):
                os.makedirs(os.path.join(out_path, 'ymdct_npy', folder))
            if not os.path.isdir(os.path.join(out_path, 'ymdct_noisy_npy', folder)):
                os.makedirs(os.path.join(out_path, 'ymdct_noisy_npy', folder))

            y_im_DCT.save(os.path.join(out_path, 'dct_vis', folder, filename + '.jpg'))
            y_im_meas.save(os.path.join(out_path, 'meas_vis', folder, filename + '.jpg'))
            np.save(os.path.join(out_path, 'ymdct_npy', folder, filename + '.npy'), n_arr)
            np.save(os.path.join(out_path, 'ymdct_noisy_npy', folder, filename + '.npy'), n_arr_noise)

            cnt += 1
            if cnt % 100 == 0:
                print('Processed class:', cnt, '/', total)

