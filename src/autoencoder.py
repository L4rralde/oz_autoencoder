import os
import logging
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from src.utils import GIT_ROOT


class Autoencoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, x_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class BowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = torch.from_numpy(np.float32(X))
        self._y = torch.from_numpy(np.float32(y))

    def __len__(self) -> int:
        return self._y.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self._X[idx], self._y[idx]


def train(model: Autoencoder, epochs: int, device: str, train_dataloader: object, val_dataloader: object) -> dict:
    model_dir = f"{GIT_ROOT}/models"
    os.makedirs(model_dir, exist_ok=True)

    fileh = logging.FileHandler(f"{model_dir}/training.log", 'a')
    sys_out = logging.StreamHandler()
    logger = logging.getLogger(__name__)  # root logger
    logger.setLevel(logging.INFO)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(sys_out)
    logger.addHandler(fileh)

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    start = perf_counter()

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0.0
        model.train()
        num_train_elements = 0
        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss += loss.data.item() * batch_size
            num_train_elements += batch_size
        train_loss /= num_train_elements
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        num_val_elements = 0
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.to(device)
                x_hat = model(x)
                loss = loss_fn(x_hat, x)
                batch_size = x.size(0)
                val_loss += loss.data.item() * batch_size
                num_val_elements += batch_size
        val_loss /= num_val_elements
        val_losses.append(val_loss)

        logger.info(f"Epoch: {epoch}. Training loss:{train_loss: .3e}. Validation loss: {val_loss: .3e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{model_dir}/best"
            logger.info(f"Saving model at: {model_path}")
            torch.save(model.state_dict(), model_path)
    end = perf_counter()
    execution_time = end - start
    logger.info(f"Execution time: {execution_time: .4f}s")
    hist = {
        "training_losses": train_losses,
        "validation_losses": val_losses,
    }

    logger.info("Saving training history")
    for k, array in hist.items():
        np.save(f"{model_dir}/{k}.npy", array)
    return hist
