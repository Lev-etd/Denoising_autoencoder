import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Autoencoder, Encoder, Decoder
from src.dataloaders import get_dataloader
from src.utils import save_model, count_parameters, get_all_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train(model, epoch, dataloader, criterion, optimizer):
    model.train()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        transformed_data = data["transformed_waveforms"].to(device)
        output = model(transformed_data).to(device)
        loss = criterion(transformed_data, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(
            f"Train Epoch: {epoch} [{(1 + batch_idx) * len(transformed_data)}/{len(dataloader.dataset)}] \tLoss: {np.mean(losses)}")
    return np.mean(losses)


def validation(model, epoch, dataloader, criterion, optimizer):
    loss_val = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            transformed_data = data["transformed_waveforms"].to(device)

            output = model(transformed_data).to(device)
            loss = criterion(transformed_data, output)

            loss_val.append(loss.item())
            print(
                f"Val Epoch: {epoch} [{(1 + batch_idx) * len(transformed_data)}/{len(dataloader.dataset)}] \tLoss: {np.mean(loss_val)}")

    save_model(epoch, model, optimizer, loss_val)
    return np.mean(loss_val)


def main(config):
    n_epoch = config.get('n_epochs')
    learning_rate = config.get('learning_rate')
    train_data = get_all_files(config.get("train_data"))
    val_data = get_all_files(config.get("val_data"))

    model = Autoencoder(latent_dim=256, encoder_class=Encoder, decoder_class=Decoder)
    model.to(device)

    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8,
                                          gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
    criterion = nn.MSELoss()

    train_dataloader = get_dataloader(train_data, batch_size=1407)
    val_dataloader = get_dataloader(val_data, batch_size=742)

    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train_loss = train(model, epoch, train_dataloader, criterion, optimizer)
            val_loss = validation(model, epoch, val_dataloader, criterion, optimizer)
            scheduler.step()
            pbar.update()
            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)


if __name__ == "__main__":
    config = {
        "n_epochs": 15,
        "learning_rate": 1e-3,
        "train_data": "splitted_data/clean_train",
        "val_data": "splitted_data/clean_val"
    }
    main(config)
