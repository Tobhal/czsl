import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import logging

from PIL import Image

import clip
import os
from os.path import join as ospj

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER

from utils.dbe import dbe
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description

from modules import models, residualmodels

import numpy as np

from utils.dbe import dbe

device = "cuda" if torch.cuda.is_available() else "cpu"

split = 'Fold0_use_50'
use_augmented = False

model_save_path = ospj('models', 'fine-tuned_clip', split, 'model.pth')
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

def gen_word_objs_embeddings(obj, clip_model):
    shape_description = gen_shape_description(obj)
    text = clip.tokenize(shape_description).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    return text_features

def calculate_cos_angle_matrix(vector_1, vector_2):
    # Ensure the vectors are PyTorch tensors and flatten them if they are 2D
    vector_1 = torch.tensor(vector_1).flatten()
    vector_2 = torch.tensor(vector_2).flatten()

    vectors = [vector_1, vector_2]
    n = len(vectors)
    cos_angle_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Calculate the dot product of the two vectors
            dot_product = torch.matmul(vectors[i], vectors[j])

            # Calculate the magnitudes of the vectors
            magnitude_a = torch.norm(vectors[i])
            magnitude_b = torch.norm(vectors[j])

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_a * magnitude_b)

            # Ensure the cosine value is within the valid range [-1, 1]
            cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix

def evaluate_model(model, dataloader, device, preprocess=clip_preprocess):
    model.eval()
    batch_similarities_same_class = []
    batch_similarities_different_class = []

    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            _, _, _, _, _, _, _, _, image_names, _, descriptions = batch

            cash_descriptions = [gen_word_objs_embeddings(description, model) for description in tqdm(descriptions, position=1, desc="Descriptions Progress", leave=False)]

            same_class_similarity = []
            different_class_similarity = []
            for i in tqdm(range(len(image_names)), position=1, desc="Image Names Progress", leave=False):
                anchor_img_name = image_names[i]
                anchor_desc = descriptions[i]

                # Load and preprocess the anchor image
                anchor_img = loader(anchor_img_name)
                anchor_img = preprocess(anchor_img).unsqueeze(0).to(device)

                anchor_image_features = model.encode_image(anchor_img)
                anchor_text_features = cash_descriptions[i]

                for j in tqdm(range(len(image_names)), position=2, desc="Comparison Progress", leave=False):
                    if i != j:
                        negative_img_name = image_names[j]
                        negative_desc = descriptions[j]

                        negative_img = loader(negative_img_name)
                        negative_img = preprocess(negative_img).unsqueeze(0).to(device)

                        negative_image_features = model.encode_image(negative_img)
                        negative_text_features = cash_descriptions[j]

                        # Calculate cosine similarity
                        # similarity = torch.nn.functional.cosine_similarity(anchor_text_features, negative_text_features).mean().item()

                        similarity = torch.nn.functional.cosine_similarity(anchor_image_features, negative_image_features).mean().item()

                        # Check if descriptions are of the same class
                        if anchor_desc == negative_desc:
                            same_class_similarity.append(similarity)
                        else:
                            different_class_similarity.append(similarity)

            # Average similarity for the batch for same and different classes
            if same_class_similarity:
                batch_similarities_same_class.append(np.mean(same_class_similarity))
                
            if different_class_similarity:
                batch_similarities_different_class.append(np.mean(different_class_similarity))

    return batch_similarities_same_class, batch_similarities_different_class


def summarize_results(original_same_class, original_different_class, fine_tuned_same_class, fine_tuned_different_class):
    # Compute average similarities
    avg_similarity_original_same = np.mean(original_same_class)
    avg_similarity_original_different = np.mean(original_different_class)
    avg_similarity_fine_tuned_same = np.mean(fine_tuned_same_class)
    avg_similarity_fine_tuned_different = np.mean(fine_tuned_different_class)

    # Determine which model performs better for same-class pairs
    better_model_same_class = "fine-tuned" if avg_similarity_fine_tuned_same > avg_similarity_original_same else "original"

    # Determine which model performs better for different-class pairs
    better_model_different_class = "fine-tuned" if avg_similarity_fine_tuned_different < avg_similarity_original_different else "original"

    print(f"Average similarity for same-class pairs (original model): {avg_similarity_original_same:.4f}")
    print(f"Average similarity for same-class pairs (fine-tuned model): {avg_similarity_fine_tuned_same:.4f}")
    print(f"The {better_model_same_class} model performs better for same-class pairs based on average similarity.")

    print(f"Average similarity for different-class pairs (original model): {avg_similarity_original_different:.4f}")
    print(f"Average similarity for different-class pairs (fine-tuned model): {avg_similarity_fine_tuned_different:.4f}")
    print(f"The {better_model_different_class} model performs better for different-class pairs based on average similarity.")


# Evaluate both models
fine_tuned_distances_same, fine_tuned_distances_same_diffrent = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)
original_distances_same, original_distances_diffrent = evaluate_model(original_clip_model, test_loader, device, fine_tuned_clip_preprocess)

# Compare and summarize results
summarize_results(original_distances_same, original_distances_diffrent, fine_tuned_distances_same, fine_tuned_distances_same_diffrent)
