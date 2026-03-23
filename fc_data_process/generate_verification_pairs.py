import argparse
import os
import random
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate verification pairs for testing."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input data."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output text file."
    )

    parser.add_argument(
        "--num_of_pairs",
        type=int,
        required=True,
        help="Number of pairs to generate."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    f = open(f'{args.output_file}', 'w')
    num_of_pairs = args.num_of_pairs

    num_of_classes = len(os.listdir(data_path))
    num_of_samp_per_sub = int(num_of_pairs/num_of_classes)
    print(num_of_samp_per_sub)
    for i, subject in enumerate(os.listdir(data_path)):
        query_files = glob.glob(os.path.join(data_path, subject, '*'))
        temp_subjects = os.listdir(data_path)
        del temp_subjects[i]
        temp_files = []
        for temp_subject in temp_subjects:
            # temp_files += os.listdir(os.path.join(data_path, temp_subject))
            temp_files += glob.glob(os.path.join(data_path, temp_subject, '*'))
        random.shuffle(temp_files)
        for i in range(1, 3):
            for k, sample in enumerate(query_files):
                f.write(query_files[k].split('/')[-2] + '/' + query_files[k].split('/')[-1].split('.')[0] + ' ' + temp_files[i*k].split('/')[-2] + '/' + temp_files[i*k].split('/')[-1].split('.')[0] + ' 0' + '\n')
        for i in range(1, 3):
            query_files_temp = query_files.copy()
            random.shuffle(query_files_temp)
            for k, sample in enumerate(query_files):
                f.write(query_files[k].split('/')[-2] + '/' + query_files[k].split('/')[-1].split('.')[0] + ' ' + query_files_temp[k].split('/')[-2] + '/' + query_files_temp[k].split('/')[-1].split('.')[0] + ' 1' + '\n')
    f.close()




