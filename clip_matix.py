import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

import logging

from PIL import Image

import clip
import os
from os.path import join as ospj

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils.dbe import dbe
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description
from modules.utils.utils import split_string_into_chunks

from modules import models, residualmodels

import numpy as np
import pandas as pd

split = 'Fold0_use_50'
use_augmented = False

# save_path = ospj('models', 'fine-tuned_clip', split)
save_path = ospj('models', 'trained_clip', split)
model_save_path = ospj(save_path, 'best.pt')
matrix_save_path = ospj(save_path, 'matrix.csv')
root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
image_loader_path = ospj(root_dir, split)

# Define phosc model
phosc_model = create_model(
    model_name='ResNet18Phosc',
    phos_size=195,
    phoc_size=1200,
    phos_layers=1,
    phoc_layers=1,
    dropout=0.5
).to(device)

# Sett phos and phoc language
set_phos_version('ben')
set_phoc_version('ben')

# Assuming you have the necessary imports and initializations done (like dset, phosc_model, etc.)
testset = dset.CompositionDataset(
    root=root_dir,
    phase='train',
    split=split,
    # phase='test'
    # split='fold_0_new',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features=False,
    train_only=True,
    open_world=True,
    augmented=use_augmented,
    add_original_data=True,
    # phosc_model=phosc_model,
    # clip_model=clip_model
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# Load original and fine-tuned CLIP models
original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
original_clip_model.float()

# Load fine-tuned clip model
fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_clip_model.float()

state_dict = torch.load(model_save_path, map_location=device)
fine_tuned_clip_model.load_state_dict(state_dict)

# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

loader = ImageLoader(image_loader_path)


def save_matrix_as_csv(matrix, model_save_path, csv_filename="matrix.csv"):
    # Extract the directory from the model save path
    directory = os.path.dirname(model_save_path)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full path for the CSV file
    csv_path = ospj(directory, csv_filename)

    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()

    # Convert the matrix to a DataFrame and save as CSV
    df = pd.DataFrame(matrix)
    df.to_csv(csv_path, index=False, header=False)

    print(f"Matrix saved as CSV at: {csv_path}")


def process_text_chunks(text_chunks, model, device):
    """Process each text chunk with the model."""
    batch_features = []
    for chunk in text_chunks:
        # Ensure the text chunk is within the context length limit
        tokens = clip.tokenize([chunk]).to(device)

        batch_features.append(tokens)

    return batch_features


def gen_word_objs_embeddings(obj, clip_model):
    shape_description = gen_shape_description(obj)

    shape_description = split_string_into_chunks(shape_description, 75)

    text = clip.tokenize(shape_description).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    # Check if the tensor's shape is less than [94, 512]
    if text_features.shape[0] < 94:
        # Calculate the number of rows to pad
        pad_rows = 94 - text_features.shape[0]

        # Pad the tensor
        text_features = F.pad(text_features, (0, 0, pad_rows, 0))
        
    return text_features


def calculate_cos_angle_matrix(vectors):
    n = len(vectors)
    cos_angle_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Convert vectors to PyTorch tensors if they aren't already
            vec_i = torch.tensor(vectors[i]).flatten()
            vec_j = torch.tensor(vectors[j]).flatten()

            # Calculate the dot product of the two vectors
            try:
                dot_product = torch.matmul(vec_i, vec_j)
            except RuntimeError as e:
                dbe(vec_i.shape, vec_j.shape, e)

            # Calculate the magnitudes of the vectors
            magnitude_i = torch.norm(vec_i)
            magnitude_j = torch.norm(vec_j)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_i * magnitude_j)

            # Ensure the cosine value is within the valid range [-1, 1]
            # cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix


def evaluate_model(model, dataloader, device, preprocess=clip_preprocess):
    model.eval()
    similarities = []
    batch_features_all = []

    description_shapes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            _, _, _, _, _, _, _, _, image_names, _, descriptions = batch

            # Precompute embeddings for all descriptions in the batch
            cash_descriptions = [gen_word_objs_embeddings(description, model) for description in
                                 tqdm(descriptions, position=1, desc="Descriptions Progress", leave=False)]
            
            for i in tqdm(range(len(image_names)), position=1, desc="Image Names Progress", leave=False):
                anchor_img_name = image_names[i]
                anchor_desc = descriptions[i]

                # Load and preprocess the anchor image
                anchor_img = loader(anchor_img_name)
                anchor_img = preprocess(anchor_img).unsqueeze(0).to(device)

                # Encode image and text using the model
                anchor_image_features = model.encode_image(anchor_img)
                anchor_text_features = cash_descriptions[i]

                # Add anchor text features to the batch features list
                batch_features_all.append(anchor_image_features)

                # Calculate cosine similarity between image and text features of the anchor
                similarity = torch.nn.functional.cosine_similarity(anchor_image_features,
                                                                   anchor_text_features).mean().item()
                similarities.append(similarity)

    return similarities, batch_features_all


def evaluate_text_embedings(model, dataloader, device, preprocess=clip_preprocess):
    model.eval()
    similarities = []
    batch_features_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            *_, image_names, _, descriptions = batch

            # Precompute embeddings for all descriptions in the batch
            batch_features_all.append([gen_word_objs_embeddings(description, model) for description in tqdm(descriptions, position=1, desc="Descriptions Progress", leave=False)])

    return batch_features_all


if __name__ == '__main__':
    # similarities, batch_features_all = evaluate_model(original_clip_model, test_loader, device, original_clip_preprocess)
    # similarities, batch_features_all = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)

    # batch_features_all = evaluate_text_embedings(original_clip_model, test_loader, device, original_clip_preprocess)
    batch_features_all = evaluate_text_embedings(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)

    # Flatten the nested list to a flat list of tensors
    flat_list_of_tensors = [item for sublist in batch_features_all for item in sublist]

    matrix = calculate_cos_angle_matrix(flat_list_of_tensors)

    # Find the minimum and maximum values in the matrix
    min_value = torch.min(matrix).item()
    max_value = torch.max(matrix).item()

    print(f"Minimum value in matrix: {min_value}")
    print(f"Maximum value in matrix: {max_value}")

    save_matrix_as_csv(matrix, matrix_save_path)
