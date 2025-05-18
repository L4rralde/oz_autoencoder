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

    dataloader = DataLoader(dataset, batch_size=32)

    M, N = X.shape
    autoencoder = Autoencoder(N, 2, n_classes=3 ,tag='oz')
    autoencoder = autoencoder.to(device)

    hist = train(
        autoencoder,
        5000,
        device,
        dataloader,
        alpha=1.0
    )


if __name__ == '__main__':
    main()
