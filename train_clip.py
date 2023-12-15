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

# Initialize logging
def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Initialized logging")

model_save_path = ospj('models', 'fine-tuned_clip', 'model.pth')
log_file_path = ospj('models', 'fine-tuned_clip', 'training_log.log')
setup_logging(log_file_path)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

clip_model.float()

model_save_path = ospj('models', 'fine-tuned_clip', 'model.pth')

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

# Get dataset
trainset = dset.CompositionDataset(
    root=ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds"),
    phase='train',
    split='Fold0_use',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features = False,
    train_only=True,
    open_world=True,
    augmented=True,
    phosc_model=phosc_model,
)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

validationset = dset.CompositionDataset(
    root=ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds"),
    phase='val',
    split='Fold0_use',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features = False,
    train_only=True,    # TODO: change to `False`
    open_world=True,
    augmented=True,
    phosc_model=phosc_model,
)

val_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# Early stopping parameters
patience = 5
best_loss = float('inf')
counter = 0


def custom_loss(image_features, text_features):
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def gen_word_objs_embeddings(obj):
    shape_description = gen_shape_description(obj)
    text = clip.tokenize(shape_description).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)


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


def custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features):
    # Use torch.nn.functional.triplet_margin_loss
    loss = torch.nn.functional.triplet_margin_loss(anchor_image_features, positive_text_features, negative_text_features, margin=1.0)
    return loss


def has_nan(tensor):
    if torch.isnan(tensor).any():
        print("NaN value found in tensor. Exiting...")
        exit()


def custom_loss_same_class(anchor_image_features, positive_text_features, negative_text_features):
    # Assuming anchor_image_features and positive_text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(anchor_image_features, positive_text_features)
    # Maximize similarity (minimize distance) for same-class pairs: lower loss for higher similarity
    loss = -torch.mean(similarity)
    return loss


def custom_loss_different_class(anchor_image_features, positive_text_features, negative_text_features):
    # Assuming anchor_image_features and negative_text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(anchor_image_features, negative_text_features)
    # Penalize high similarity for different classes
    loss = 1 - torch.mean(similarity)
    return loss


def main(): 
    global patience, best_loss, counter

    # Fine-tuning (simple example)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    clip_model.train()

    loader = ImageLoader(ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds", 'Fold0_use'))

    # Fine-tuning (with early stopping)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    clip_model.train()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    model_save_path = ospj('models', 'fine-tuned_clip', 'model.pth')

    verify_model_save_path(model_save_path)

    for epoch in range(100):  # Set a maximum number of epochs
        clip_model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_bar:
            accumulated_loss = torch.zeros(1, device=device, requires_grad=True)

            # Unpacking the batch data
            _, _, _, _, _, _, _, _, image_names, _, descriptions = batch

            cash_description = [gen_word_objs_embeddings(description) for description in descriptions]

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

                        # Determine if descriptions are of the same class
                        is_same_class = (descriptions[i] == descriptions[j])

                        # Calculate custom loss based on class
                        if is_same_class:
                            # Minimize distance for same class
                            loss = custom_loss_same_class(anchor_image_features, anchor_text_features, negative_text_features)
                        else:
                            # Maximize distance for different classes
                            loss = custom_loss_different_class(anchor_image_features, anchor_text_features, negative_text_features)

                        accumulated_loss = accumulated_loss + loss  # Out-of-place operation
            
            # Normalize loss by the number of comparisons
            normalized_loss = accumulated_loss / (len(image_names) * (len(image_names) - 1))

            optimizer.zero_grad()
            normalized_loss.backward()  # Backpropagate the normalized loss
            optimizer.step()
            running_loss += normalized_loss.item()

            train_bar.set_description(f"Training Epoch {epoch + 1} Loss: {normalized_loss.item():.4f}")

        # Logging training loss
        train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch}, Training Loss: {train_loss}")

        # Validation step
        clip_model.eval()
        with torch.no_grad():
            val_loss = 0.0

            val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            for batch in val_bar:
                accumulated_loss = torch.zeros(1, device=device)

                # Unpacking the batch data
                _, _, _, _, _, _, _, _, image_names, _, descriptions = batch
                
                cash_description = [gen_word_objs_embeddings(description) for description in descriptions]

                for i in range(len(image_names)):
                    anchor_img_name = image_names[i]
                    anchor_desc = descriptions[i]

                    # Load and preprocess the anchor image
                    anchor_img = loader(anchor_img_name)
                    anchor_img = clip_preprocess(anchor_img).unsqueeze(0).to(device)

                    anchor_image_features = clip_model.encode_image(anchor_img)
                    anchor_text_features = cash_description[i]

                    for j in range(len(image_names)):
                        if i != j:
                            negative_text_features = cash_description[j] 

                            # Determine if descriptions are of the same class
                            is_same_class = (descriptions[i] == descriptions[j])

                            # Calculate custom loss based on class
                            if is_same_class:
                                # Minimize distance for same class
                                loss = custom_loss_same_class(anchor_image_features, anchor_text_features, negative_text_features)
                            else:
                                # Maximize distance for different classes
                                loss = custom_loss_different_class(anchor_image_features, anchor_text_features, negative_text_features)

                            accumulated_loss += loss

                # Normalize loss by the number of comparisons
                normalized_loss = accumulated_loss.mean()
                val_loss += normalized_loss.item()
                val_bar.set_description(f"Validation Epoch {epoch + 1} Loss: {normalized_loss.item():.4f}")

            val_loss /= len(val_loader)
            # Logging validation loss
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss}")

            scheduler.step(val_loss)

            print(f"Epoch {epoch}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}, Patience: {counter}")
            logging.info(f"Epoch {epoch}, Best Loss: {best_loss}, Patience Counter: {counter}")

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                # Save the model
                torch.save(clip_model.state_dict(), model_save_path)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Early exit')
        print(f'Best loss: {best_loss}')

        logging.info(f'Best loss: {best_loss}')
        # Save the model
        torch.save(clip_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")