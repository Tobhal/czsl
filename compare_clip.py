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
original_clip_model, _ = clip.load("ViT-B/32", device=device)

# Load fine-tuned clip model
fine_tuned_clip_model, _ = clip.load("ViT-B/32", device=device)

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

def evaluate_model(model, dataloader, device):
    model.eval()
    batch_similarities_same_class = []
    batch_similarities_different_class = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Unpacking the batch data
            # _, _, _, _, image_names, _, descriptions = batch
            _, _, _, _, _, _, _, _, image_names, _, descriptions = batch

            same_class_similarity = []
            different_class_similarity = []
            for i in range(len(image_names)):
                anchor_img_name = image_names[i]
                anchor_desc = descriptions[i]

                # Load and preprocess the anchor image
                anchor_img = loader(anchor_img_name)
                anchor_img = clip_preprocess(anchor_img).unsqueeze(0).to(device)

                anchor_image_features = model.encode_image(anchor_img)
                anchor_text_features = gen_word_objs_embeddings(anchor_desc, model)

                for j in range(len(image_names)):
                    if i != j:
                        negative_desc = descriptions[j]
                        negative_text_features = gen_word_objs_embeddings(negative_desc, model)

                        # Calculate cosine similarity
                        similarity = torch.nn.functional.cosine_similarity(anchor_text_features, negative_text_features).mean().item()

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
original_distances_same, original_distances_diffrent = evaluate_model(original_clip_model, test_loader, device)
fine_tuned_distances_same, fine_tuned_distances_same_diffrent = evaluate_model(fine_tuned_clip_model, test_loader, device)

# Compare and summarize results
summarize_results(original_distances_same, original_distances_diffrent, fine_tuned_distances_same, fine_tuned_distances_same_diffrent)
