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
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim),
        )
        self._x_dim = x_dim
        self._z_dim = z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @classmethod
    def load_best(cls, x_dim: int, z_dim: int) -> "Autoencoder":
        model = cls(x_dim, z_dim)
        path = f"{GIT_ROOT}/models/z_dim_{z_dim}_best"
        status = model.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location=torch.device("cpu")
            )
        )
        print(status)
        return model


class BowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = torch.from_numpy(np.float32(X))
        self._y = torch.from_numpy(np.float32(y))

    def __len__(self) -> int:
        return self._y.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self._X[idx], self._y[idx]


def train(model: Autoencoder, epochs: int, device: str, dataloader: object) -> dict:
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
    best_loss = float('inf')

    start = perf_counter()

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0.0
        model.train()
        num_train_elements = 0
        for x, _ in dataloader:
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

        logger.info(f"Epoch: {epoch}. Training loss:{train_loss: .3e}.")
        if train_loss < best_loss:
            best_loss = train_loss
            model_path = f"{GIT_ROOT}/models/z_dim_{model._z_dim}_best"
            logger.info(f"Saving model at: {model_path}")
            torch.save(model.state_dict(), model_path)
    end = perf_counter()
    execution_time = end - start
    logger.info(f"Execution time: {execution_time: .4f}s")
    hist = {
        "training_losses": train_losses,
    }

    logger.info("Saving training history")
    for k, array in hist.items():
        np.save(f"{model_dir}/{k}.npy", array)
    return hist


def encode_and_test(model: Autoencoder, device: str, dataloader: object) -> np.ndarray:
    num_elements = 0
    test_loss = 0.0
    zs = []

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            zs.append(z.cpu().numpy())
            x_hat = model.decoder(z)
            loss = loss_fn(x_hat, x)
            batch_size = x.size(0)
            test_loss += loss.data.item() * batch_size
            num_elements += batch_size
    test_loss /= num_elements
    print(f"Loss: {test_loss}")
    return np.vstack(zs)
