from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# %matplotlib inline
# plt.rcParams["figure.figsize"] = [9.6, 7.2]

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Pickle
import pickle

# Pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.model_zoo as model_zoo
import torchmetrics
from torchmetrics import Metric

# PyTorch Lightning
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

# Sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import warnings

# warnings.filterwarnings('ignore')

# Torchvision for CV
import torchvision
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
)

# Others
import os
from PIL import Image
from enum import Enum
from pathlib import Path
from collections import defaultdict

"""#Datasets"""


class Datasets(Enum):
    CRLeaves = 1
    PLantCLEF2017Trusted = 2


class Sampling(Enum):
    NUMPY = 1
    SKLEARN = 2
    NONE = 3


"""##DataModule"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        folder_counts = {}

        for folder in os.scandir(root_dir):
            if folder.is_dir():
                folder_counts[folder.name] = len(entry for entry in os.scandir(folder) if entry.is_file())

        self.folders = sorted(folder_counts, key=folder_counts.get)
        
        self.images = []
        self.labels = []
        self.class_counts = defaultdict(int)
        for i, folder in enumerate(self.folders):
            folder_path = os.path.join(root_dir, folder)
            for image_name in os.scandir(folder_path):
                if image_name.name.lower().endswith(("jpg", "jpeg", "png")) and image_name.is_file():
                    self.images.append(os.path.join(folder_path, image_name.name))
                    self.labels.append(i)
                    self.class_counts[i] += 1

        self.class_counts = list(self.class_counts.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class PlantDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        root_dir,
        batch_size,
        test_size=0.5,
        use_index=True,
        sampling=Sampling.NONE,
        train_transform=None,
        test_transform=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.use_index = use_index
        self.sampling = sampling

        self.train_folder = ImageFolderDataset(
            root_dir=root_dir, transform=train_transform
        )
        self.test_folder = ImageFolderDataset(
            root_dir=root_dir, transform=test_transform
        )
        self.class_counts = self.train_folder.class_counts
        self.idxPATH = "idxPATH/" + str(dataset) + ".pkl"

    def prepare_data(self):
        if self.use_index:
            with open(self.idxPATH, "rb") as file:
                data = pickle.load(file)
                self.train_indices, self.test_indices = (
                    data["train_indices"],
                    data["test_indices"],
                )
                # self.train_indices, self.test_indices = (data["train_idx"], data["val_idx"])
        else:
            self.train_indices, self.test_indices = train_test_split(
                range(len(self.train_folder)),
                test_size=self.test_size,
                stratify=self.train_folder.labels,
            )

            with open(self.idxPATH, "wb") as file:
                pickle.dump(
                    {
                        "train_indices": self.train_indices,
                        "test_indices": self.test_indices,
                    },
                    file,
                )

        self.train_dataset = Subset(self.train_folder, self.train_indices)
        self.test_dataset = Subset(self.test_folder, self.test_indices)
        train_labels = np.array(self.train_folder.labels)[self.train_indices]
        self.class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )

        if self.sampling != Sampling.NONE:
            if self.sampling == Sampling.NUMPY:
                class_counts = np.array(
                    [np.sum(train_labels == c) for c in np.unique(train_labels)]
                )
                self.class_weights = 1.0 / class_counts

            weights = [self.class_weights[label] for label in train_labels]
            self.train_sampler = WeightedRandomSampler(weights, len(self.train_dataset))

    def create_data_loaders(self):
        if self.sampling == Sampling.NONE:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=7,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=7,
                pin_memory=True,
                sampler=self.train_sampler,
                persistent_workers=True,
            )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader


class PlantDataModuleMinority(LightningDataModule):
    def __init__(
        self,
        dataset,
        root_dir,
        batch_size,
        train_transform=None,
        test_transform=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size

        

        self.train_folder = ImageFolderDataset(
            root_dir=root_dir, transform=train_transform
        )
        self.test_folder = ImageFolderDataset(
            root_dir=root_dir, transform=test_transform
        )
        self.class_counts = self.train_folder.class_counts
        self.idxPATH = "idxPATH/" + str(dataset) + ".pkl"

    def prepare_data(self):
        with open(self.idxPATH, "rb") as file:
            data = pickle.load(file)
            train_indices, test_indices = (
                data["train_indices"],
                data["test_indices"],
            )

            train_labels = np.array(self.train_folder.labels)[train_indices]
            train_indices = np.array(train_indices)[
                np.where(np.array(self.class_counts)[train_labels] < 10)[0]
            ]

            test_labels = np.array(self.test_folder.labels)[test_indices]
            test_indices = np.array(test_indices)[
                np.where(np.array(self.class_counts)[test_labels] < 10)[0]
            ]

            self.train_dataset = Subset(self.train_folder, train_indices)
            self.test_dataset = Subset(self.test_folder, test_indices)

    def create_data_loaders(self):

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader


class PlantDataModuleMajority(LightningDataModule):
    def __init__(
        self,
        dataset,
        root_dir,
        batch_size,
        train_transform=None,
        test_transform=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.train_folder = ImageFolderDataset(
            root_dir=root_dir, transform=train_transform
        )
        self.test_folder = ImageFolderDataset(
            root_dir=root_dir, transform=test_transform
        )
        self.class_counts = self.train_folder.class_counts
        self.idxPATH = "idxPATH/" + str(dataset) + ".pkl"

    def prepare_data(self):
        with open(self.idxPATH, "rb") as file:
            data = pickle.load(file)
            train_indices, test_indices = (
                data["train_indices"],
                data["test_indices"],
            )

            train_labels = np.array(self.train_folder.labels)[train_indices]
            train_indices = np.array(train_indices)[
                np.where(np.array(self.class_counts)[train_labels] >= 10)[0]
            ]

            test_labels = np.array(self.test_folder.labels)[test_indices]
            test_indices = np.array(test_indices)[
                np.where(np.array(self.class_counts)[test_labels] >= 10)[0]
            ]

            self.train_dataset = Subset(self.train_folder, train_indices)
            self.test_dataset = Subset(self.test_folder, test_indices)

    def create_data_loaders(self):

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader

##Resnet50


class Resnet50(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet50(x)


class Resnet50SelfSupervision(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(2048, out_features)
        )

    def forward(self, x):
        return self.backbone(x)


class Resnet101(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet101 = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet101.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet101(x)


class Resnet152(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet152 = resnet152(weights=ResNet152_Weights.DEFAULT)
        self.resnet152.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet152(x)


"""##LightningModule"""
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassConfusionMatrix,
)

from torcheval.metrics import ReciprocalRank

from torchmetrics.functional import accuracy


class ModeloSelfSupervision(pl.LightningModule):
    def __init__(self, model, criterion, batch_size, *args, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / 0.07
        return logits, labels

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = torch.cat(x, dim=0)
        features = self.model(x)
        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        labels, preds = labels.cpu(), preds.cpu()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train/general_acc",
            accuracy_score(labels, preds),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0003, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]


class ModeloBase(pl.LightningModule):
    def __init__(
        self,
        lr,
        model,
        criterion,
        class_count,
        id,
        species_count,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.criterion = criterion
        self.class_count = class_count
        self.species_count = species_count
        self.resultsPATH = "resultsPATH/" + id + ".pkl"
        with open(self.resultsPATH, "wb") as file:
            pickle.dump({"class_count": class_count}, file)

        metrics = MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(
                    num_classes=species_count, average="micro"
                ),
                "BalancedAccuracy": MulticlassAccuracy(num_classes=species_count),
                "CohenKappa": MulticlassCohenKappa(num_classes=species_count),
                "F1Score": MulticlassF1Score(num_classes=species_count),
                "MatthewsCorrCoef": MulticlassMatthewsCorrCoef(
                    num_classes=species_count
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.train_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.val_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.test_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.train_mrr = ReciprocalRank()
        self.val_mrr = ReciprocalRank()
        self.test_mrr = ReciprocalRank()

    def forward(self, x):
        return self.model(x)

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=30, verbose=True
        # )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, verbose=True
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.train_metrics.update(y_hat, y)
        self.train_cm.update(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.log(
            "train/MRR", self.train_mrr.compute().mean(), on_step=False, on_epoch=True
        )
        matrix = self.train_cm.compute()
        self.correlation(matrix, "train")

        self.train_metrics.reset()
        self.train_mrr.reset()
        self.train_cm.reset()

        train_labels, train_predictions = torch.cat(
            [x["train/labels"] for x in outputs], dim=0
        ), torch.cat([x["train/predictions"] for x in outputs], dim=0)
        self.save_results(train_labels, train_predictions, "train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.val_metrics.update(y_hat, y)
        self.val_cm.update(y_hat, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.log("val/MRR", self.val_mrr.compute().mean(), on_step=False, on_epoch=True)
        matrix = self.val_cm.compute()
        self.correlation(matrix, "val")

        self.val_metrics.reset()
        self.val_mrr.reset()
        self.val_cm.reset()

        val_labels, val_predictions = torch.cat(
            [x["val/labels"] for x in outputs], dim=0
        ), torch.cat([x["val/predictions"] for x in outputs], dim=0)
        self.save_results(val_labels, val_predictions, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.test_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.test_metrics.update(y_hat, y)
        self.test_cm.update(y_hat, y)

        self.log("test/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.log(
            "test/MRR", self.test_mrr.compute().mean(), on_step=False, on_epoch=True
        )
        matrix = self.test_cm.compute()
        self.correlation(matrix, "test")

        self.test_metrics.reset()
        self.test_mrr.reset()
        self.test_cm.reset()

        test_labels, test_predictions = torch.cat(
            [x["test/labels"] for x in outputs], dim=0
        ), torch.cat([x["test/predictions"] for x in outputs], dim=0)
        self.save_results(test_labels, test_predictions, "test")

    def correlation(self, matrix, phase):
        class_recall = matrix.diag() / matrix.sum(dim=1)
        class_recall = class_recall.masked_fill_(torch.isnan(class_recall), 0)
        minority_class_recall_mean = torch.mean(
            class_recall[torch.lt(torch.tensor(self.class_count), 10)]
        )

        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        plot = plt.figure()
        ax = plt.gca()
        ax.scatter(self.class_count, class_recall.cpu(), alpha=0.5)
        ax.set_xscale("log")
        ax.axvline(10)
        ax.set_ylim((0, 1))
        for axis in [ax.xaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.xlabel("Image Count")
        plt.ylabel("Accuracy")
        self.logger.experiment.log(
            {f"{phase}/Accuracy Correlation": [wandb.Image(plot)]}
        )

    def save_results(self, labels, predictions, phase):
        with open(self.resultsPATH, "rb") as file:
            data = pickle.load(file)

        with open(self.resultsPATH, "wb") as file:
            data[f"{self.current_epoch}-{phase}-labels"] = labels
            data[f"{self.current_epoch}-{phase}-predictions"] = predictions
            pickle.dump(data, file)
import psutil
class ModeloEnsamblado(pl.LightningModule):
    def __init__(
        self,
        models,
        class_count,
        id,
        species_count,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.class_count = class_count
        self.species_count = species_count
        self.resultsPATH = "resultsPATH/" + id + ".pkl"
        with open(self.resultsPATH, "wb") as file:
            pickle.dump({"class_count": class_count}, file)

        metrics = MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(
                    num_classes=species_count, average="micro"
                ),
                "BalancedAccuracy": MulticlassAccuracy(num_classes=species_count)
            }
        )
        self.test_metrics = metrics.clone(prefix="test/")
        mem_stats = psutil.virtual_memory()

        # Print the total, available, and used memory in bytes
        print("Total memory: ", mem_stats.total / (1024 ** 3), "GB")
        print("Available memory: ", mem_stats.available / (1024 ** 3), "GB")
        print("Used memory: ", mem_stats.used / (1024 ** 3), "GB")

        self.test_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.test_mrr = ReciprocalRank()

    def configure_optimizers(self):
        pass

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        logits = torch.stack(logits_list, dim=1).sum(dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.test_metrics.update(y_hat, y)
        self.test_cm.update(y_hat, y)

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.log(
            "test/MRR", self.test_mrr.compute().mean(), on_step=False, on_epoch=True
        )
        matrix = self.test_cm.compute()
        self.correlation(matrix, "test")

    def correlation(self, matrix, phase):
        
        class_recall = matrix.diag() / matrix.sum(dim=1)
        class_recall = class_recall.masked_fill_(torch.isnan(class_recall), 0)
        minority_class_recall_mean = torch.mean(
            class_recall[torch.lt(torch.tensor(self.class_count), 10)]
        )

        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        plot = plt.figure()
        ax = plt.gca()
        ax.scatter(self.class_count, class_recall.cpu(), alpha=0.5)
        ax.set_xscale("log")
        ax.axvline(10)
        ax.set_ylim((0, 1))
        for axis in [ax.xaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.xlabel("Image Count")
        plt.ylabel("Accuracy")
        self.logger.experiment.log(
            {f"{phase}/Accuracy Correlation": [wandb.Image(plot)]}
        )


train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(232),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(232),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ]
)        
# Training
# LR
# PlantCLEF 2017
# 0.2754228703338169     0.5 SGD
# 0.001584893192461114   0.5 Adam
# CR Leaves
"""
dataset = Datasets.PLantCLEF2017Trusted
batch_size = 64
test_size = 0.5
use_index = True
lr = 0.0022908676527677745 if dataset == Datasets.CRLeaves else 0.001584893192461114
lr = 0.003311311214825908 if dataset == Datasets.CRLeaves else 0.001584893192461114
lr = 0.0003
sampling = Sampling.NONE
if dataset == Datasets.CRLeaves:
    root_dir = "CRLeaves/"
elif dataset == Datasets.PLantCLEF2017Trusted:
    root_dir = "data/"
#datamodule = PlantDataModule(
#    dataset=dataset,
#    root_dir=root_dir,
#    batch_size=batch_size,
#    test_size=test_size,
#    use_index=use_index,
#    sampling=sampling,
#)

datamodule = PlantDataModuleMinority(
    dataset=dataset, root_dir=root_dir, batch_size=batch_size
)
datamodule.prepare_data()
datamodule.create_data_loaders()
class_weights = False
species_count = 254 if dataset == Datasets.CRLeaves else 10000
criterion = (
    torch.nn.CrossEntropyLoss(weight=datamodule.class_weights)
    if class_weights
    else torch.nn.CrossEntropyLoss()
)

module = Resnet50(out_features=species_count)
model = ModeloBase(
    lr=lr,
    model=module,
    criterion=criterion,
    class_count=datamodule.class_counts,
    id="id",
    species_count=species_count
)

trainer = pl.Trainer(accelerator="gpu", devices=-1, max_epochs=30, precision=16)

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)

# Results can be found in
print(lr_finder.results)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)


"""

# Data Module
dataset = Datasets.PLantCLEF2017Trusted
batch_size = 512 if dataset == Datasets.CRLeaves else 64  # 128
test_size = 0.5 if dataset == Datasets.CRLeaves else 0.5
use_index = False
sampling = Sampling.NONE
if dataset == Datasets.CRLeaves:
    root_dir = "CRLeaves/"
elif dataset == Datasets.PLantCLEF2017Trusted:
    root_dir = "data/"

datamodule = PlantDataModule(
    dataset=dataset,
    root_dir=root_dir,
    batch_size=batch_size,
    test_size=test_size,
    use_index=use_index,
    sampling=sampling,
    train_transform=train_transform,
    test_transform=test_transform
)

#datamodule = PlantDataModuleMajority(
#    dataset=dataset, root_dir=root_dir, batch_size=batch_size,train_transform=train_transform,test_transform=test_transform
#)

datamodule.prepare_data()
datamodule.create_data_loaders()
class_weights = False
species_count = 254 if dataset == Datasets.CRLeaves else 10000
criterion = (
    torch.nn.CrossEntropyLoss(
        weight=torch.tensor(datamodule.class_weights, dtype=torch.float)
    )
    if class_weights
    else torch.nn.CrossEntropyLoss()
)
id = None
if id is None:
    id = wandb.util.generate_id()
print(id)

lr = 0.0003


module = Resnet50(out_features=species_count)
# module = Resnet50SelfSupervision(out_features=species_count)
#module = ModeloSelfSupervision.load_from_checkpoint(
#    "/home/ruben/Documents/Thesis/ThesisTest/41d1xucn/checkpoints/epoch=29-step=6150.ckpt",
#    model=module,
#    criterion=torch.nn.CrossEntropyLoss(),
#    batch_size=batch_size,
#).model

model = ModeloBase(
    lr=lr,
    model=module,
    criterion=criterion,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
)

wandb_logger = WandbLogger(project="ThesisTest", id=id, resume="allow")

from pytorch_lightning.callbacks import LearningRateMonitor

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=30,
    precision=16,
    callbacks=[LearningRateMonitor(logging_interval="epoch")],
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
wandb.finish()


"""




dataset = Datasets.PLantCLEF2017Trusted
batch_size = 512
test_size = 0.5 if dataset == Datasets.CRLeaves else 0.5
use_index = True
sampling = Sampling.NONE
if dataset == Datasets.CRLeaves:
    root_dir = "CRLeaves/"
elif dataset == Datasets.PLantCLEF2017Trusted:
    root_dir = "data/"

datamodule = PlantDataModule(
    dataset=dataset,
    root_dir=root_dir,
    batch_size=batch_size,
    test_size=test_size,
    use_index=use_index,
    sampling=sampling,
    train_transform=train_transform,
    test_transform=test_transform
)

datamodule.prepare_data()
datamodule.create_data_loaders()
class_weights = False
species_count = 254 if dataset == Datasets.CRLeaves else 10000
criterion = (
    torch.nn.CrossEntropyLoss(
        weight=torch.tensor(datamodule.class_weights, dtype=torch.float)
    )
    if class_weights
    else torch.nn.CrossEntropyLoss()
)
id = None
if id is None:
    id = wandb.util.generate_id()
print(id)

lr = 0.0003



module1 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/Thesis/ThesisTest/8084omju/checkpoints/epoch=29-step=6150.ckpt",
    lr=lr,
    model=Resnet50(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
).model

module2 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/Thesis/ThesisTest/2ug8at72/checkpoints/epoch=29-step=53940.ckpt",
    lr=lr,
    model=Resnet50(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
).model

model = ModeloEnsamblado(
    models = [module1, module2],
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
)

wandb_logger = WandbLogger(project="ThesisTest", id=id, resume="allow")

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=0,
    num_sanity_val_steps=0,
    precision=16
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
wandb.finish()

"""