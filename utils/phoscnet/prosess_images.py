from dbe import dbe

import os
import shutil
import csv


def copy_files(src_path, dst_directory, new_file_name):
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
        
    dst_path = os.path.join(dst_directory, new_file_name)
    shutil.copy2(src_path, dst_path)

def process_images(input_directory, output_directory, labels):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    csv_data = []
    total_images = 0

    d = 0
    fl = 0

    # Get a list of all directory indices (as integers) present in the input_directory
    dir_indices = [int(dir_name) for dir_name in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, dir_name))]
    dir_indices.sort()
    
    # initialize all labels
    labels_dict = {}
    with open(os.path.join(input_directory, labels), 'r') as f:
        for i, label in enumerate(f):
            labels_dict[str(dir_indices[i])] = label.strip()

    
    for root, dirs, files in os.walk(input_directory):
        d += 1
        
        for file_name in files:
            fl += 1
            
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                total_images += 1
                new_image_path = os.path.join(root, file_name)
                dir_index = os.path.basename(root)
                
                # Append the directory name to the file name to ensure uniqueness
                new_file_name = dir_index + "_" + file_name

                # Copy file with a new name
                copy_files(new_image_path, output_directory, new_file_name)

                if dir_index in labels_dict:
                    # Add new file name to the CSV data
                    csv_data.append((new_file_name, labels_dict[dir_index]))


    with open(os.path.join(output_directory, 'data.csv'), 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(('Image', 'Word'))
        
        # Sorting csv_data based on index and filename
        csv_data.sort(key=lambda x: (int(x[0].split('_')[0]), x[0].split('_')[1]))
        
        writer.writerows(csv_data)

    print(f"{input_directory} -> {output_directory}")
    print("Number of directories:", d)
    print("Number of files:", fl)
    print("Labels dict:", len(labels_dict))
    print("Number of images: ", total_images)
    print("Number of CSV data rows: ", len(csv_data))
    print("Process completed!\n")

if __name__ == '__main__':
    fold = 4

    folders = [
        # Input                                                           output                                                            labels
        (f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_n/train',   f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_new/train',  f'Train_Labels_Fold{fold}.txt'),
        (f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_n/test',    f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_new/test',   f'Test_Labels_Fold{fold}.txt'),
        (f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_n/val',     f'data/BengaliWords_CroppedVersion_Folds/fold_{fold}_new/val',    f'Val_Labels_Fold{fold}.txt'),
    ]
    
    for input, output, labels in folders:
        process_images(input, output, labels)