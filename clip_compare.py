import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import logging

from PIL import Image

import torch.nn.functional as F

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

from modules import models, residualmodels
from modules.utils.utils import get_phosc_description

import numpy as np

from utils.dbe import dbe
from utils.utils import text_features_from_description

from torchvision import transforms
from typing import List, Tuple

from modules.utils.utils import split_string_into_chunks

# align
from transformers import AlignProcessor, AlignModel
from enum import Enum

split = 'Fold0_use_50'
use_augmented = False

# align model
align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")

# czsl/models/fine-tuned_clip/Fold0_use/simple/18/best.pt
finetuned_model_save_path = ospj('models', 'trained_clip', split, 'bengali_words', '1', 'best.pt')
# trained_model_save_path = ospj('models', 'trained_clip', split, 'super_aug', '2', 'best.pt')
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
    num_workers=0,
)

# Load original and fine-tuned CLIP models
original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
original_clip_model.float()
"""
# Load fine-tuned clip model
fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_clip_model.float()

fine_tuned_state_dict = torch.load(finetuned_model_save_path, map_location=device)
fine_tuned_clip_model.load_state_dict(fine_tuned_state_dict)
"""
# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

align_preprocess = Compose([
    ToTensor(),
])

loader = ImageLoader(image_loader_path)


class ModelType(Enum):
    CLIP = "CLIP"
    ALIGN = "ALIGN"


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


def clip_preprocess_and_encode(image_names, words, clip_model, transform, loader, device):
    # Preprocess and encode images
    images = [transform(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
    images = torch.cat(images, dim=0)
    images_enc = clip_model.encode_image(images)

    # Precompute embeddings for all descriptions in the batch
    text_features = torch.stack([text_features_from_description(word, clip_model) for word in tqdm(words, position=1, desc='Generating Embeddings', leave=False)]).squeeze(1)

    # Normalize image embeddings after encoding
    ims = [F.normalize(image, dim=0) for image in images_enc]
    ims = torch.stack(ims)

    # Normalize text embeddings after encoding
    txt = [F.normalize(txt, dim=0) for txt in text_features]
    txt = torch.stack(txt)

    # Compute similarity scores between images and texts
    image_logits = ims @ txt.t() * clip_model.logit_scale.exp()
    ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)

    # Compute cosine similarity between image and text embeddings
    similarity_matrix = torch.nn.functional.cosine_similarity(image_logits, ground_truth, dim=0)

    return similarity_matrix


def align_preprocess_and_encode(image_names, words, align_model, transform, loader, device):
    images = [loader(img_name) for img_name in image_names]
    probs = []

    for image, word in zip(images, words):
        description = get_phosc_description(word)

        inputs = align_processor(text=description, images=image, return_tensors="pt")
        outputs = align_model(**inputs)

        logits = outputs.logits_per_image

        prob = logits.softmax(dim=1)
        # dbe(prob[0][0])
        probs.append(prob[0][0])
    
    return probs


def evaluate_model(clip_model, dataloader, device, loader, model_type: ModelType):
    clip_model.eval()
    batch_similarities_same_class = []
    batch_similarities_different_class = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            _, _, _, _, _, _, _, _, image_names, _, words = batch



            if model_type == ModelType.CLIP:
                batch_similarities_same_class.append(clip_preprocess_and_encode(image_names, words, clip_model, transform, loader, device))
            else:
                batch_similarities_same_class.append(align_preprocess_and_encode(image_names, words, clip_model, transform, loader, device))

    # Flatten the list of tensors into a single list of numbers
    flat_similarities = [item for sublist in batch_similarities_same_class for item in sublist]
    # flat_similarities = [item for sublist in batch_similarities_same_class for item in sublist.tolist()]

    # Compute average similarities for same and different classes
    avg_same_class_similarity = np.mean(flat_similarities) if flat_similarities else 0
    avg_different_class_similarity = np.mean(batch_similarities_different_class) if batch_similarities_different_class else 0

    return avg_same_class_similarity, avg_different_class_similarity



def summarize_results(original_same_class, original_different_class, fine_tuned_same_class, fine_tuned_different_class):
    # Compute average similarities
    avg_similarity_original_same = np.mean(original_same_class)
    # avg_similarity_original_different = np.mean(original_different_class)
    avg_similarity_fine_tuned_same = np.mean(fine_tuned_same_class)
    # avg_similarity_fine_tuned_different = np.mean(fine_tuned_different_class)

    # Determine which model performs better for same-class pairs
    # better_model_same_class = "fine-tuned" if avg_similarity_fine_tuned_same > avg_similarity_original_same else "original"

    # Determine which model performs better for different-class pairs
    # better_model_different_class = "fine-tuned" if avg_similarity_fine_tuned_different < avg_similarity_original_different else "original"

    print(f"Average similarity for same-class pairs (original model): {avg_similarity_original_same:.4f}")
    print(f"Average similarity for same-class pairs (fine-tuned model): {avg_similarity_fine_tuned_same:.4f}")
    # print(f"The {better_model_same_class} model performs better for same-class pairs based on average similarity.")

    # print(f"Average similarity for different-class pairs (original model): {avg_similarity_original_different:.4f}")
    # print(f"Average similarity for different-class pairs (fine-tuned model): {avg_similarity_fine_tuned_different:.4f}")
    # print(f"The {better_model_different_class} model performs better for different-class pairs based on average similarity.")


# Evaluate both models

# original_distances_same, original_distances_diffrent = evaluate_model(original_clip_model, test_loader, device, fine_tuned_clip_model, loader, ModelType.CLIP)
# fine_tuned_distances_same, fine_tuned_distances_same_diffrent = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_model, loader)
align_distances_same, align_distances_diffrent = evaluate_model(align_model, test_loader, device, loader, ModelType.ALIGN)

# Compare and summarize results
# summarize_results(original_distances_same, original_distances_diffrent, fine_tuned_distances_same, fine_tuned_distances_same_diffrent)
summarize_results(align_distances_same, align_distances_diffrent, align_distances_same, align_distances_diffrent)
