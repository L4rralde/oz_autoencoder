import numpy as np
import matplotlib.pyplot as plt


def plot_2D_project(project: np.ndarray, title: str, y: np.ndarray, labels: dict, ax: object = None) -> None:
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()
    for i, label in labels.items():
        idx = y == i
        ax.scatter(*project[idx].T, label=label, alpha=0.3)
    ax.legend()
    ax.set_title(title)


def plot_3D_project(project: np.ndarray, title: str, y: np.ndarray, labels: dict, ax: object=None) -> None:
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(projection='3d')
    for i, label in labels.items():
        idx = y == i
        ax.scatter(*project[idx].T, label=label, alpha=0.3)
    ax.legend()
    ax.set_title(title)


def axplot_project(
        project: np.ndarray,
        title: str,
        y: np.ndarray,
        labels: dict,
        ax: object,
        legend: bool = False
    ) -> None:
    for i, label in labels.items():
        idx = y == i
        ax.scatter(*project[idx].T, label=label, alpha=0.3)
    if legend:
        ax.legend()
    ax.set_title(title)
