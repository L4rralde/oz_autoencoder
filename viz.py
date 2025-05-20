import os

import torch
from torch.utils.data import DataLoader
from torchview import draw_graph

from src.dataset import Dataset, group_obs
from src.autoencoder import BowDataset, Autoencoder
from src.utils import GIT_ROOT


def main():
    IMG_DIR = f"{GIT_ROOT}/docs/imgs"
    os.makedirs(IMG_DIR, exist_ok=True)

    X, y, labels = Dataset.load_dataset(True)
    X, y = group_obs(X, y, labels, 100)
    dataset = BowDataset(X, y)

    test_dataloader = DataLoader(dataset, batch_size=32)
    for x, _ in test_dataloader:
        break
    z = torch.zeros_like(x)
    model = Autoencoder.load_best(X.shape[1], 2, 3, 'oz')
    model_graph = draw_graph(model, z)
    fpath = f"{IMG_DIR}/model"
    model_graph.visual_graph.render(fpath, format="jpeg")


if __name__ == '__main__':
    main()
