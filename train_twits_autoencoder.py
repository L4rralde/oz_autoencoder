import torch
from torch.utils.data import DataLoader

from src.dataset import load_twits_dataset
from src.autoencoder import Autoencoder, BowDataset, train


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def main() -> None:
    X, y = load_twits_dataset()
    dataset = BowDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=256)
    M, N = X.shape
    autoencoder = Autoencoder(N, 2, n_classes=2, tag='twits')
    autoencoder = autoencoder.to(device)

    hist = train(
        autoencoder,
        1000,
        device,
        dataloader,
        alpha=0.2
    )

if __name__ == '__main__':
    main()
