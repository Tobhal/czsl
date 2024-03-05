import torch
import numpy as np
import os
import json

from utils.dbe import dbe

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path: str, patience=7, verbose=True, save_every=5, model_arguments: dict = {}):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_every = save_every
        self.model_arguments = model_arguments
        self.best_model_path = None
        
        # Initialize save_path with a new run-specific folder
        self.save_path = self.initialize_save_path(save_path)

        # Print the save path
        print(f"Saving model to {self.save_path}")
        
        # Convert model_arguments to dictionary (if it's not already) and save
        if isinstance(self.model_arguments, dict):
            args_dict = self.model_arguments
        else:
            args_dict = vars(self.model_arguments)  # Assuming model_arguments might be a Namespace from argparse

        args_save_path = os.path.join(self.save_path, f'model_args.json')

        with open(args_save_path, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)


    def initialize_save_path(self, base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            run_number = 1
        else:
            existing_runs = [int(folder) for folder in os.listdir(base_path) if folder.isdigit() and os.path.isdir(os.path.join(base_path, folder))]
            run_number = max(existing_runs) + 1 if existing_runs else 1

        run_save_path = os.path.join(base_path, str(run_number))
        os.makedirs(run_save_path, exist_ok=True)

        return run_save_path

    def __call__(self, val_loss, model, epoch) -> bool:
        score = -val_loss
        should_stop = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, 'best', checkpoint_type='best')
            self.best_model_path = os.path.join(self.save_path, 'best.pt')

        elif score < self.best_score:
            if self.patience is not None:
                self.counter += 1

                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

                if self.counter >= self.patience:
                    self.early_stop = True
                    should_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, 'best', checkpoint_type='best')
            self.counter = 0

        if self.save_every is not None:
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(val_loss, model, f'epoch {epoch + 1}', checkpoint_type='epoch')

        return should_stop


    def save_checkpoint(self, val_loss, model, name, checkpoint_type):
        '''Saves model and model arguments when validation loss decreases.'''
        if self.verbose:
            if checkpoint_type == 'best':
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model and arguments ...') 
            elif checkpoint_type == 'epoch':
                print(f'Saving model and arguments for epoch {name} ...')

        model_save_path = os.path.join(self.save_path, f'{name}.pt')

        # Save the model
        torch.save(model.state_dict(), model_save_path)

        self.val_loss_min = val_loss

