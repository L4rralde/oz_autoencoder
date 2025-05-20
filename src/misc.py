import numpy as np
import matplotlib.pyplot as plt


def plot_2D_project(project: np.ndarray, title: str, y: np.ndarray, labels: dict) -> None:
    plt.figure(figsize=(6, 4))
    for i, label in labels.items():
        idx = y == i
        plt.scatter(*project[idx].T, label=label, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_3D_project(project: np.ndarray, title: str, y: np.ndarray, labels: dict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, label in labels.items():
        idx = y == i
        ax.scatter(*project[idx].T, label=label, alpha=0.3)
    ax.legend()
    ax.set_title(title)