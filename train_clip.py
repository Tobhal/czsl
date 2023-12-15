import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import logging

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

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize logging
def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info("Initialized logging")


def gen_word_objs_embeddings(obj, model):
    shape_description = gen_shape_description(obj)
    text = clip.tokenize(shape_description).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features


# Function to check if the model save path's directory exists
def verify_model_save_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        print(f"Model save directory does not exist, creating: {directory}")
        os.makedirs(directory)


def custom_loss(image_features, text_features):
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def normalize_features(features):
    return features / features.norm(dim=1, keepdim=True)


def custom_loss_same_class(anchor_image_features, positive_text_features, negative_text_features):
    # Assuming anchor_image_features and positive_text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(
        normalize_features(anchor_image_features),
        normalize_features(positive_text_features)
    )
    # Maximize similarity (minimize distance) for same-class pairs: lower loss for higher similarity
    loss = -torch.mean(similarity)
    return loss


def custom_loss_different_class(anchor_image_features, positive_text_features, negative_text_features):
    # Assuming anchor_image_features and negative_text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(
        normalize_features(anchor_image_features),
        normalize_features(positive_text_features)
    )
    # Penalize high similarity for different classes
    loss = 1 - torch.mean(similarity)
    return loss


def custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features, margin=1.0):
    # Calculate triplet loss
    loss = torch.nn.functional.triplet_margin_loss(
        anchor_image_features,
        positive_text_features,
        negative_text_features,
        margin=margin
    )
    return loss


def train_one_epoch(epoch, train_loader, clip_model, clip_preprocess, loader, optimizer):
    global device

    clip_model.train()
    running_loss = 0

    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    for batch in train_bar:
        optimizer.zero_grad()

        # Unpacking the batch data
        *_, image_names, _, descriptions = batch

        cash_description = [gen_word_objs_embeddings(description, clip_model) for description in descriptions]

        accumulated_loss = torch.zeros(1, device=device)

        for i in range(len(image_names)):
            anchor_img_name = image_names[i]

            # Load and preprocess the anchor image
            anchor_img = loader(anchor_img_name)
            anchor_img = clip_preprocess(anchor_img).unsqueeze(0).to(device)

            anchor_image_features = clip_model.encode_image(anchor_img)
            anchor_text_features = cash_description[i]

            for j in range(len(image_names)):
                if i != j:
                    negative_text_features = cash_description[j]

                    loss = custom_triplet_loss(anchor_image_features, anchor_text_features, negative_text_features)
                    accumulated_loss += loss

        # Normalize loss by the number of comparisons
        normalized_loss = accumulated_loss.mean()
        normalized_loss.backward()
        optimizer.step()

        running_loss += normalized_loss.item()
        train_bar.set_description(f"Training Epoch {epoch + 1} Loss: {normalized_loss.item():.4f}")

    # Logging training loss
    train_loss = running_loss / len(train_loader)
    logging.info(f"Epoch {epoch}, Training Loss: {train_loss}")

    return train_loss


def validate_one_epoch(epoch, val_loader, clip_model, clip_preprocess, loader, scheduler):
    clip_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        for batch in val_bar:
            accumulated_loss = torch.zeros(1, device=device)

            # Unpacking the batch data
            *_, image_names, _, descriptions = batch

            for i in range(len(image_names)):
                anchor_img_name = image_names[i]
                anchor_desc = descriptions[i]

                # Load and preprocess the anchor image
                anchor_img = loader(anchor_img_name)
                anchor_img = clip_preprocess(anchor_img).unsqueeze(0).to(device)

                anchor_image_features = clip_model.encode_image(anchor_img)
                anchor_text_features = gen_word_objs_embeddings(anchor_desc, clip_model)

                for j in range(len(image_names)):
                    if i != j:
                        negative_desc = descriptions[j]
                        negative_text_features = gen_word_objs_embeddings(negative_desc, clip_model)

                        loss = custom_triplet_loss(anchor_image_features, anchor_text_features, negative_text_features)

                        accumulated_loss += loss

            # Normalize loss by the number of comparisons
            normalized_loss = accumulated_loss.mean()
            val_loss += normalized_loss.item()
            val_bar.set_description(f"Validation Epoch {epoch + 1} Loss: {normalized_loss.item():.4f}")

        val_loss /= len(val_loader)

    # Logging validation loss
    logging.info(f"Epoch {epoch}, Validation Loss: {val_loss}")

    scheduler.step(val_loss)

    return val_loss


def main(): 
    global clip_model, clip_preprocess, model_save_path
    # Datasett variables
    split = 'Fold0_use_50'
    use_augmented = False

    # Loader variables
    batch_size = 64

    # Define patchs
    root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
    image_loader_path = ospj(root_dir, split)

    root_model_path = ospj('models', 'fine-tined_clip', split)
    log_file_path = ospj(root_model_path, 'training_log.log')
    model_save_path = ospj(root_model_path, 'model.pth')

    verify_model_save_path(model_save_path)

    setup_logging(log_file_path)

    # Set up models
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

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

    # Setup datasett
    trainset = dset.CompositionDataset(
        root=root_dir,
        phase='train',
        split=split,
        model='resnet18',
        num_negs=1,
        pair_dropout=0.5,
        update_features = False,
        train_only=True,
        open_world=True,
        augmented=use_augmented,
        phosc_model=phosc_model,
    )

    validationset = dset.CompositionDataset(
        root=root_dir,
        phase='val',
        split=split,
        model='resnet18',
        num_negs=1,
        pair_dropout=0.5,
        update_features = False,
        train_only=False,
        open_world=True,
        augmented=use_augmented,
        phosc_model=phosc_model,
    )


    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        validationset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Define image loader
    loader = ImageLoader(image_loader_path)

    # Define early stopping parameters
    best_loss = float('inf')
    patience_counter = 0
    patience_threshold = 20  # Number of epochs to wait before stopping

    # Fine-tuning
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    for epoch in range(100):  # Set a maximum number of epochs
        train_loss = train_one_epoch(
            epoch,
            train_loader,
            clip_model,
            clip_preprocess,
            loader,
            optimizer
        )

        val_loss = validate_one_epoch(
            epoch,
            val_loader,
            clip_model,
            clip_preprocess,
            loader,
            scheduler
        )

        l = val_loss

        # Inside your training loop
        if l < best_loss:
            best_loss = l
            patience_counter = 0
            print(f'{epoch}: New best loss: {best_loss}')

            # Save the model
            torch.save(clip_model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                print("Early stopping triggered")
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Ctrl-C exit')