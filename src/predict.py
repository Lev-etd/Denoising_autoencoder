import os
import sys

import soundfile as sf
import torch
import torch.optim as optim

from model import Autoencoder, Encoder, Decoder
from src.dataloaders import get_dataloader
from src.utils import load_model, get_all_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
print(nb_dir)

def main(config):
    learning_rate = config.get('learning_rate')
    test_path = config.get('test_path')
    weights_path = config.get('weights_path')

    model = Autoencoder(latent_dim=256, encoder_class=Encoder, decoder_class=Decoder)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch, loss = load_model(model, weights_path, optimizer)

    model.to(device)
    model.eval()

    test_dataloader = get_dataloader(data=get_all_files(test_path),
                                     batch_size=1)
    output = model(next(iter(test_dataloader))['transformed_waveforms'])

    print(output)

    sf.write('../result_file.wav', output.cpu().detach().squeeze(), 16000)


if __name__ == "__main__":
    config = {
        "n_epochs": 15,
        "learning_rate": 1e-3,
        "test_path": f"{nb_dir}/splitted_data/clean_testset_wav",
        "weights_path": f"{nb_dir}/weights/epoch_17_loss0.872458.pth"
    }
    main(config)
