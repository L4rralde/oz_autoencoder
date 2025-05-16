from sklearn.model_selection import KFold


def kfolding(dataset: list, folds: int) -> list:
    kf = KFold(n_splits=5)

    folds = [
        (
            [dataset[idx] for idx in train_idx],
            [dataset[idx] for idx in valid_idx]
        )
        for train_idx, valid_idx in kf.split(dataset)
    ]

    return folds
