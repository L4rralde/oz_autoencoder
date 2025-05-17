import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import Dataset, group_obs, sub_dataset
from src.autoencoder import Autoencoder, BowDataset, train


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def main() -> None:
    X, y, labels = Dataset.load_dataset(True)
    X, y = group_obs(X, y, labels, 100)
    dataset = BowDataset(X, y)

    tot_size = len(dataset)
    val_size = tot_size//10
    train_size = tot_size - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    M, N = X.shape
    autoencoder = Autoencoder(N, 2)
    autoencoder = autoencoder.to(device)

    hist = train(
        autoencoder,
        2000,
        device,
        train_dataloader,
        val_dataloader
    )


if __name__ == '__main__':
    main()
