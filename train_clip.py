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
    split='fold_0_new',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features = False,
    train_only=True,
    open_world=True,
    augmented=False,
    phosc_model=phosc_model,
)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

validationset = dset.CompositionDataset(
    root=ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds"),
    phase='val',
    split='fold_0_new',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features = False,
    train_only=True,
    open_world=True,
    augmented=False,
    phosc_model=phosc_model,
)

val_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=1,
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
    global clip_model

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


def has_nan(tensor):
    if torch.isnan(tensor).any():
        print("NaN value found in tensor. Exiting...")
        exit()


def main(): 
    global patience, best_loss, counter

    # Fine-tuning (simple example)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    clip_model.train()

    loader = ImageLoader(ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds", 'fold_0_new'))

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
        for data in train_bar:
            _, _, _, _, _, _, _, _, image, attr, obj = data
            image = loader(image[0])
            image = clip_preprocess(image).unsqueeze(0).to(device)

            text_features = gen_word_objs_embeddings(obj[0])  # Get text features

            # Calculate loss
            image_features = clip_model.encode_image(image)

            has_nan(image_features)

            loss = custom_loss(image_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # dbe(running_loss, loss, calls_before_exit=20, print_when_not_exit=True)

            # Update progress bar description with the current loss
            train_bar.set_description(f"Training Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Logging training loss
        train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch}, Training Loss: {train_loss}")

        # Validation step
        clip_model.eval()
        with torch.no_grad():
            val_loss = 0.0

            val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            for data in val_bar:
                _, _, _, _, _, _, _, _, image, attr, obj = data
                image = loader(image[0])
                image = clip_preprocess(image).unsqueeze(0).to(device)

                text_features = gen_word_objs_embeddings(obj[0])  # Get text features

                image_features = clip_model.encode_image(image)

                has_nan(image_features)

                val_loss += custom_loss(image_features, text_features)

                # Update progress bar description with the current loss
                val_bar.set_description(f"Validation Epoch {epoch + 1} Loss: {val_loss / len(val_loader):.4f}")

            val_loss /= len(val_loader)
            # Logging validation loss
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss}")


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