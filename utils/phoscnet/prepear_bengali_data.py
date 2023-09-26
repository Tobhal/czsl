import argparse
import os
import shutil

from os.path import join as ospj
from dbe import dbe


from split_to_val import split_to_val

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some folders and a label file.")

    # Argument for input folder
    parser.add_argument("--input_folder", type=str, help="Input folder for bengali fold data")

    # Argument for output folder
    parser.add_argument("--output_folder", type=str, help="Output directory for bengali fold data")

    # Argument for labels (text file)
    parser.add_argument("--labels", type=str, help="Path to text file describing the bengali data")

    parser.add_argument('--augmentations', type=int, help='The number of times to augment a single training image')

    # Argument for what folds to prosess
    parser.add_argument('--folds', type=int, nargs='+', help='What folds to prosess', required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # dbe(args)

    for fold in args.folds:
        fold_str = f'Fold{fold}'

        input_fold_folder = ospj(args.input_folder, f'{fold_str}_n')
        output_fold_folder = ospj(args.output_folder, f'{fold_str}_superaug')

        input_folder_test = ospj(input_fold_folder, 'test')
        input_folder_train = ospj(input_fold_folder, 'train')
        input_folder_val = ospj(input_fold_folder, 'val')

        output_folder_test = ospj()
        output_folder_
        output_folder_

        fold_label_train = ospj(input_folder_train, f'Train_Labels_{fold_str}.txt')

        # Split train into train and validation
        split_to_val(input_folder_train, input_folder_val, fold_label_train, fold)

        # TODO: Not done...





        # shutil.rmtree(temp_prosessing_folder)
        


if __name__ == '__main__':
    main()

"""
python utils/phoscnet/prepear_bengali_data.py --input_folder ../../DATA_ROOT/BengaliWords/BengaliWords_CroppedVersion_Folds --output_folder ../../DATA_ROOT/BengaliWords/BengaliWords_CroppedVersion_Folds --labels ../../DATA_ROOT/BengaliWords/BengaliWords_CroppedVersion_Folds --augmentations 40 --folds 0 1 2 3 4
"""