from pathlib import Path

import numpy as np
import torch
from IPython.display import Audio, display


def get_all_files(folder):
    """
    Returns:
        list with PosixPath of all files from folder
    """
    return [*Path(folder).iterdir()]


def play_audio(waveform, sample_rate):
    display(Audio(waveform, rate=sample_rate))


def count_parameters(model):
    """
    Count trainable parameters.
    :param model: model to count parameters for
    :return: number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(epoch, model, optimizer, loss_val):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': loss_val},
               f'./epoch_{epoch}_loss{np.mean(loss_val):4f}.pth')


def load_model(model, path_to_weights, optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(path_to_weights)
    else:
        checkpoint = torch.load(path_to_weights, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    return epoch, loss
