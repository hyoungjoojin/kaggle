# %% [markdown]
"""
# Digit Recognizer
## Simple CNN with an MLP Classifier

- LB Score: 0.97978

"""

# %%
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"

NUM_EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

SEED = 503
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% [markdown]
"""
## PyTorch Dataset and DataLoader
"""


# %%
def columns_to_image(columns: List) -> np.ndarray:
    """Transform the list of 784 items into a 2D numpy array."""
    image = np.array(columns)
    image = image.reshape((28, 28))
    return image


class DigitDataset(data.Dataset):
    def __init__(self, images: List, labels: List) -> None:
        super(DigitDataset, self).__init__()
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple:
        image = columns_to_image(self.images[index].tolist())
        image = torch.from_numpy(image).float().unsqueeze(0)

        label = self.labels[index]
        return (image, label)

    def __len__(self) -> int:
        return len(self.images)


class DigitDataLoader(data.DataLoader):
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = BATCH_SIZE,
        *args,
        **kwargs,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        super(DigitDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    def __len__(self) -> int:
        assert self.batch_size is not None
        return int(len(self.dataset) / self.batch_size)  # type:ignore


# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_X = train_df.loc[:, train_df.columns != "label"].to_numpy()
train_y = train_df["label"].to_numpy()

train_X, val_X, train_y, val_y = train_test_split(
    train_X,
    train_y,
    test_size=0.2,
    shuffle=True,
)


train_dataset = DigitDataset(train_X, train_y)
val_dataset = DigitDataset(val_X, val_y)

train_dataloader = DigitDataLoader(train_dataset)
val_dataloader = DigitDataLoader(val_dataset)


# %% [markdown]
"""
## PyTorch Model
"""


# %%
class DigitClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DigitClassifier, self).__init__(*args, **kwargs)
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=245, out_features=100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)
        return x


# %% [markdown]
"""
## Metrics
"""


# %%
def accuracy(pred: torch.Tensor, real: torch.Tensor) -> Tuple[int, int]:
    """Get the accuracy of two tensors.

    Returns:
        Tuple: (number of right elements, number of wrong elements)

    """
    total_items = pred.shape[0]
    total_right = int((torch.argmax(pred, dim=1) == real).sum().item())
    return (total_right, total_items - total_right)


# %% [markdown]
"""
## Training Procedure
"""

# %%
model = DigitClassifier().to(DEVICE)
optimizer = torch.optim.Adam(lr=3e-4, params=model.parameters())
criterion = nn.CrossEntropyLoss().to(DEVICE)

best_validation_loss = 1000000
for i in range(NUM_EPOCHS):
    model.train()
    total_iter, total_loss, total_right, total_wrong = 0, 0.0, 0, 0
    for batch_idx, (image, label) in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
    ):
        image, label = image.to(DEVICE), label.to(DEVICE)

        prediction = model(image)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iter += 1
        total_loss += loss.item()

        right, wrong = accuracy(prediction, label)
        total_right += right
        total_wrong += wrong

    train_loss = total_loss / total_iter
    train_accuracy = total_right / (total_right + total_wrong)
    print(f"[TRAIN] Epoch {i}")
    print(f"\tLoss: {train_loss:4f}")
    print(f"\tAcc : {train_accuracy:4f}%")

    model.eval()
    total_iter, total_loss, total_right, total_wrong = 0, 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (image, label) in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
        ):
            image, label = image.to(DEVICE), label.to(DEVICE)

            prediction = model(image)
            loss = criterion(prediction, label)

            total_iter += 1
            total_loss += loss.item()

            right, wrong = accuracy(prediction, label)
            total_right += right
            total_wrong += wrong

    val_loss = total_loss / total_iter
    val_accuracy = total_right / (total_right + total_wrong)
    print(f"[VAL] Epoch {i}")
    print(f"\tLoss: {val_loss:4f}")
    print(f"\tAcc : {val_accuracy:4f}%")

    if val_loss >= best_validation_loss:
        print("Validation loss did not decrease. Early stopping.")
        break
    else:
        best_validation_loss = val_loss


# %% [markdown]
"""
## Generate Submission
"""


# %%
def generate_submission(
    model: nn.Module,
    test_X: pd.DataFrame,
    submission_file="./submission.csv",
):
    print("Generating submission file ...")
    model.eval()

    results = []
    for _, row in tqdm(test_X.iterrows(), total=len(test_X)):
        image = (
            torch.from_numpy(columns_to_image(row.tolist()))
            .float()
            .to(DEVICE)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )

        label = model(image).squeeze()
        results.append(torch.argmax(label).item())

    print(f"Saving submission file to {submission_file} ...")
    with open(submission_file, "w") as f:
        contents = "ImageId,Label\n" + "\n".join(
            [f"{index + 1},{item}" for index, item in enumerate(results)],
        )
        f.write(contents)


generate_submission(model, test_df)
