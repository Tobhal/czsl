import os
import shutil
import random

from dbe import dbe
from pathlib import Path
from typing import IO


def split_to_val(input_dir: Path, dest_dir: Path, classes_file: IO[str], fold: int, split_percentage=0.2):
    if not os.path.exists(dest_dir):
        print('Dir created')
        os.makedirs(dest_dir)

    dirs = [dir for dir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, dir))]

    selected = random.sample(dirs, int(len(dirs)*split_percentage))
    
    # Read the original text file
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()

    # Prepare a list for the moved classes
    moved_classes = []
    remaining_classes = []

    for i, class_ in enumerate(classes):
        if f"{i}" in selected:
            moved_classes.append(class_)
        else:
            remaining_classes.append(class_)

    for dir in selected:
        shutil.move(os.path.join(input_dir, dir), os.path.join(dest_dir, dir))

    # Write the moved classes into a new text file
    with open(os.path.join(dest_dir, f'Val_Labels_Fold{fold}.txt'), 'w') as f:
        f.write('\n'.join(moved_classes))

    # Rewrite the original file with remaining classes
    with open(classes_file, 'w') as f:
        f.write('\n'.join(remaining_classes))

    print(moved_classes)


if __name__ == '__main__':
    fold = 4

    # usage
    base_dir = f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_n'
    input_dir = os.path.join(base_dir, 'train')
    dest_dir = os.path.join(base_dir, 'val')
    classes_file = f'Train_Labels_Fold{fold}.txt'

    split_to_val(input_dir, dest_dir, classes_file, 0.2)