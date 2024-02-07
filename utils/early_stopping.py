import torch
import numpy as np
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path: str, patience=7, verbose=True, save_epoch=5):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_path = save_path
        self.save_epoch = save_epoch

    def __call__(self, val_loss, model, epoch) -> bool:
        score = -val_loss
        should_stop = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, 'best')
        elif score < self.best_score:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                should_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, 'best')
            self.counter = 0

        if self.save_epoch is not None:
            if (epoch + 1) % self.save_epoch == 0:
                self.save_checkpoint(val_loss, model, f'epoch{epoch + 1}')

        return should_stop


    def save_checkpoint(self, val_loss, model, name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Check if the directory exists, if not, create it
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        save_file_path = os.path.join(self.save_path, f'{name}.pt')

        # Save the model
        torch.save(model.state_dict(), save_file_path)

        # Save the model here if needed
        self.val_loss_min = val_loss

