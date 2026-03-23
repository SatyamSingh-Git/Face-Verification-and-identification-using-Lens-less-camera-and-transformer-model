import argparse
import os
import shutil
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to unsplitted input dataset"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output directory (train/test split)"
    )

    parser.add_argument(
        "--img_size",
        type=str,
        default="128",
        help="Image size for resizing (default: 128 -> 128x128). Use 'None' to disable resizing."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path
    out_path = args.out_path

    # Handle img_size
    img_size = None if args.img_size.lower() == "none" else int(args.img_size)

    # Fixed test image names - 1st image from each variation is used as test image
    test_imgs = [
        '001', '011', '021', '031', '041', '051', '061', '071', '081', '091',
        '101', '111', '121', '131', '141', '151', '161', '171', '181', '191',
        '201', '211', '221', '227', '233', '239', '245', '251', '257', '263', '269'
    ]

    classes = sorted(os.listdir(data_path))
    cnt = 0

    for folder in classes:
        train_dir = os.path.join(out_path, "train", folder)
        test_dir = os.path.join(out_path, "test", folder)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = sorted(os.listdir(os.path.join(data_path, folder)))

        for file in files:
            f_name, _ = os.path.splitext(file)
            src_path = os.path.join(data_path, folder, file)

            if f_name in test_imgs:
                dst_path = os.path.join(test_dir, file)
            else:
                dst_path = os.path.join(train_dir, file)

            # Resize or copy
            if img_size is None:
                shutil.copyfile(src_path, dst_path)
            else:
                with Image.open(src_path) as im:
                    im = im.resize((img_size, img_size))
                    im.save(dst_path)

        cnt += 1
        print(f"Processed class: {cnt}/{len(classes)}")


if __name__ == "__main__":
    main()
