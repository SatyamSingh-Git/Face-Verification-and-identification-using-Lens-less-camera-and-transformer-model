import argparse
import shutil
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test for face recognition"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input dataset"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output directory (train/test split)"
    )

    parser.add_argument(
        "--resize_hw",
        type=str,
        default=None,
        help="Image size for resizing (e.g: 128 -> 128x128). Use 'None' to disable resizing. Default: None"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    out_path = args.out_path
    resize_hw = None if args.resize_hw.lower() == "none" else int(args.resize_hw)

    # Fixed test image names - 1st image from each variation is used as test image
    test_imgs = ['001', '011', '021', '031', '041', '051', '061', '071', '081', '091', '101', '111', '121', '131', '141', '151', '161', '171', '201', '211',
            '221', '227', '233', '239', '245', '251', '257', '263', '269']
    rngs = [[181, 190], [191, 200]] # Remove 'profile left' and 'profile right' views
    f_names = []
    for r in rngs:
        f_names += ["{0:03}".format(i) for i in range(r[0], r[1] + 1)]

    cnt = 0
    for folder in os.listdir(data_path):
        if not os.path.exists(os.path.join(out_path, 'train', folder)):
            os.makedirs(os.path.join(out_path, 'train', folder))
        if not os.path.exists(os.path.join(out_path, 'test', folder)):
            os.makedirs(os.path.join(out_path, 'test', folder))

        files = sorted(os.listdir(os.path.join(data_path, folder)))
        for file in files:
            f_name, f_ext = os.path.splitext(file)
            if f_name in f_names:
                continue
            
            im=Image.open(os.path.join(data_path, folder, file))
            if f_name in test_imgs:
                if resize_hw is not None:
                    im=im.resize((resize_hw,resize_hw))
                    im.save(os.path.join(out_path, 'test', folder, file))
                else:
                    shutil.copyfile(os.path.join(data_path, folder, file), os.path.join(out_path, 'test', folder, file))
            else:
                if resize_hw is not None:
                    im = im.resize((resize_hw, resize_hw))
                    im.save(os.path.join(out_path, 'train', folder, file))
                else:
                    shutil.copyfile(os.path.join(data_path, folder, file), os.path.join(out_path, 'train', folder, file))
        cnt += 1
        print('Processed class:', cnt, '/', len(os.listdir(data_path)))

