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

torch.set_float32_matmul_precision("high")

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
    convnext_base,
    ConvNeXt_Base_Weights,
    efficientnet_b4,
    EfficientNet_B4_Weights,
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
    def __init__(self, root_dir, transform=None, minority_index=None, binary=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.minority_index = minority_index
        self.binary = binary

        folder_counts = {}

        for folder in os.scandir(root_dir):
            if folder.is_dir():
                folder_counts[folder.name] = len(
                    [entry for entry in os.scandir(folder) if entry.is_file()]
                )

        self.folders = sorted(folder_counts, key=folder_counts.get)

        self.images = []
        self.labels = []
        self.class_counts = defaultdict(int)
        for i, folder in enumerate(self.folders):
            folder_path = os.path.join(root_dir, folder)
            for image_name in os.scandir(folder_path):
                if (
                    image_name.name.lower().endswith(("jpg", "jpeg", "png"))
                    and image_name.is_file()
                ):
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
        if self.minority_index:
            if self.binary:
                label = 0 if label < minority_index else 1
            else:
                label -= minority_index
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
        test_transform=None,
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
        minority_index,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.minority_index = minority_index

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
            train_indices = np.array(train_indices)[train_labels < self.minority_index]

            test_labels = np.array(self.test_folder.labels)[test_indices]
            test_indices = np.array(test_indices)[test_labels < self.minority_index]

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


class PlantDataModuleMajority(PlantDataModuleMinority):
    def __init__(
        self,
        dataset,
        root_dir,
        batch_size,
        minority_index,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__(
            dataset,
            root_dir,
            batch_size,
            minority_index,
            train_transform,
            test_transform,
        )
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.minority_index = minority_index

        self.train_folder = ImageFolderDataset(
            root_dir=root_dir, transform=train_transform, minority_index=minority_index
        )
        self.test_folder = ImageFolderDataset(
            root_dir=root_dir, transform=test_transform, minority_index=minority_index
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
            train_indices = np.array(train_indices)[train_labels >= self.minority_index]

            test_labels = np.array(self.test_folder.labels)[test_indices]
            test_indices = np.array(test_indices)[test_labels >= self.minority_index]

            self.train_dataset = Subset(self.train_folder, train_indices)
            self.test_dataset = Subset(self.test_folder, test_indices)


class PlantDataModuleBinary(PlantDataModule):
    def __init__(
        self,
        dataset,
        root_dir,
        batch_size,
        minority_index,
        test_size=0.5,
        use_index=True,
        sampling=Sampling.NONE,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__(
            dataset,
            root_dir,
            batch_size,
            test_size,
            use_index,
            sampling,
            train_transform,
            test_transform,
        )
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.use_index = use_index
        self.sampling = sampling

        self.train_folder = ImageFolderDataset(
            root_dir=root_dir,
            transform=train_transform,
            minority_index=minority_index,
            binary=True,
        )
        self.test_folder = ImageFolderDataset(
            root_dir=root_dir,
            transform=test_transform,
            minority_index=minority_index,
            binary=True,
        )
        self.class_counts = self.train_folder.class_counts
        self.idxPATH = "idxPATH/" + str(dataset) + ".pkl"


class GaussianBlur(object):
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3,
            3,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.blur_v = nn.Conv2d(
            3,
            3,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.k = kernel_size
        self.r = radias
        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)
        self.pil_to_tensor = torchvision.transforms.ToTensor()
        self.tensor_to_pil = torchvision.transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
        img = self.tensor_to_pil(img)
        return img


class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def simclr_transform(size, s=1):
    color_jitter = torchvision.transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    return data_transforms


class PlantDataModuleSelfSupervision(LightningDataModule):
    def __init__(self, root_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        transform = ContrastiveLearningViewGenerator(simclr_transform(224), 2)
        self.dataset = ImageFolderDataset(root_dir=root_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )


##Resnet50


class Resnet50(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet50(x)


class EfficientNetB4(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.efficientnet_b4 = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        self.efficientnet_b4.classifier[1] = torch.nn.Linear(
            in_features=1792, out_features=out_features
        )

    def forward(self, x):
        return self.efficientnet_b4(x)


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


class Convnext(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.convnext.classifier[2] = nn.Linear(1024, out_features, bias=True)

    def forward(self, x):
        return self.convnext(x)


"""##LightningModule"""
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassConfusionMatrix,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

from torcheval.metrics import ReciprocalRank

from torchmetrics.functional import accuracy


class ModeloSelfSupervision(pl.LightningModule):
    def __init__(self, lr, model, criterion, batch_size, *args, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr

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
            "train/Accuracy",
            accuracy_score(labels, preds),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, verbose=True
        )
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
        minority_index,
        majority=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.criterion = criterion
        self.class_count = class_count
        self.species_count = species_count
        self.majority = majority
        self.minority_index = minority_index
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
        minority_class_recall_mean = (
            0.0 if self.majority else torch.mean(class_recall[: self.minority_index])
        )

        majority_class_recall_mean = (
            # torch.mean(class_recall)
            torch.mean(class_recall[self.minority_index :])
            if self.majority
            else torch.mean(class_recall[self.minority_index :])
        )
        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{phase}/BalancedAccuracyMajority",
            majority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        class_count = self.class_count
        # class_count = (
        #    self.class_count[self.minority_index :]
        #    if self.majority
        #    else self.class_count[: self.minority_index]
        # )

        plot = plt.figure()
        ax = plt.gca()
        ax.scatter(class_count, class_recall.cpu(), alpha=0.5)
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


class ModeloBinario(pl.LightningModule):
    def __init__(
        self,
        lr,
        model,
        criterion,
        class_count,
        minority_index,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([4.8930467, 0.55690811]))
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([5.81, 0.53]))
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.10247254]))
        # self.criterion = nn.BCEWithLogitsLoss()
        self.class_count = class_count
        self.minority_index = minority_index

        metrics = MetricCollection(
            {
                "Accuracy": BinaryAccuracy(),
                "BalancedAccuracy": MulticlassAccuracy(num_classes=2),
                "F1Score": BinaryF1Score(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.train_cm = BinaryConfusionMatrix()
        self.val_cm = BinaryConfusionMatrix()
        self.test_cm = BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def configure_optimizers(self):
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
        # loss = self.loss(y_hat, y.unsqueeze(1).float())
        y_hat = y_hat.argmax(dim=-1)
        # y_hat = torch.round(torch.sigmoid(y_hat.squeeze()))

        self.train_metrics.update(y_hat, y)
        self.train_cm.update(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        matrix = self.train_cm.compute()
        self.correlation(matrix, "train")

        self.train_metrics.reset()
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # loss = self.loss(y_hat, y.unsqueeze(1).float())
        y_hat = y_hat.argmax(dim=-1)
        # y_hat = torch.round(torch.sigmoid(y_hat.squeeze()))
        self.val_metrics.update(y_hat, y)
        self.val_cm.update(y_hat, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        matrix = self.val_cm.compute()
        self.correlation(matrix, "val")

        self.val_metrics.reset()
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # loss = self.loss(y_hat, y.unsqueeze(1).float())
        y_hat = y_hat.argmax(dim=-1)
        # y_hat = torch.round(torch.sigmoid(y_hat.squeeze()))

        self.test_metrics.update(y_hat, y)
        self.test_cm.update(y_hat, y)

        self.log("test/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        matrix = self.test_cm.compute()
        self.correlation(matrix, "test")

        self.test_metrics.reset()
        self.test_cm.reset()

    def correlation(self, matrix, phase):
        class_recall = matrix.diag() / matrix.sum(dim=1)
        class_recall = class_recall.masked_fill_(torch.isnan(class_recall), 0)
        minority_class_recall_mean = class_recall[0]
        majority_class_recall_mean = class_recall[1]

        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{phase}/BalancedAccuracyMajority",
            majority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        minority = self.class_count[: self.minority_index]
        majority = self.class_count[self.minority_index :]
        class_count = [sum(minority) / len(minority), sum(majority) / len(majority)]

        plot = plt.figure()
        ax = plt.gca()
        ax.scatter(class_count, class_recall.cpu(), alpha=0.5)
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
                # "Accuracy": MulticlassAccuracy(
                #     num_classes=species_count, average="micro"
                # ),
                # "BalancedAccuracy": MulticlassAccuracy(num_classes=species_count),
                # "CohenKappa": MulticlassCohenKappa(num_classes=species_count),
                # "F1Score": MulticlassF1Score(num_classes=species_count),
                "MatthewsCorrCoef": MulticlassMatthewsCorrCoef(
                    num_classes=species_count
                ),
            }
        )
        self.test_metrics = metrics.clone(prefix="test/")

        self.test_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.test_mrr = ReciprocalRank()

    def configure_optimizers(self):
        pass

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        # logits = torch.stack(logits_list, dim=1).sum(dim=1)
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        # logits = torch.cat(logits_list, dim=1)
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
        majority_class_recall_mean = torch.mean(
            class_recall[torch.gt(torch.tensor(self.class_count), 10)]
        )

        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{phase}/BalancedAccuracyMajority",
            majority_class_recall_mean,
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


class ModeloEnsamblado2(pl.LightningModule):
    def __init__(
        self,
        binary_model,
        models,
        class_count,
        id,
        species_count,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.binary_model = binary_model
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
                "BalancedAccuracy": MulticlassAccuracy(num_classes=species_count),
                # "CohenKappa": MulticlassCohenKappa(num_classes=species_count),
                # "F1Score": MulticlassF1Score(num_classes=species_count),
                # "MatthewsCorrCoef": MulticlassMatthewsCorrCoef(
                #    num_classes=species_count
                # ),
            }
        )
        self.test_metrics = metrics.clone(prefix="test/")

        self.test_cm = MulticlassConfusionMatrix(num_classes=species_count)
        self.test_mrr = ReciprocalRank()

    def configure_optimizers(self):
        pass

    def forward(self, x):
        logits_binary = self.binary_model(x)

        y_hat = logits_binary.argmax(dim=-1)
        logits_list = torch.zeros(x.shape[0], self.species_count).to("cuda:0")

        # EXACT
        for i, item in enumerate(y_hat):
            xi = torch.unsqueeze(x[i], 0)
            if item == 1:
                logits = self.models[0](xi)
                full = torch.full((1, 254 - 18), -100).to("cuda:0")
                logits = torch.cat((logits, full), dim=1)
                logits_list[i] = logits
            else:
                full = torch.full((1, 18), -100).to("cuda:0")
                logits = self.models[1](xi)
                logits = torch.cat((full, logits), dim=1)
                logits_list[i] = logits
        return logits_list.to("cuda:0")
        """

        logits_list = torch.zeros(x.shape[0], self.species_count).to("cuda:0")
        y_hat = F.softmax(logits_binary, dim=0)

        for i, item in enumerate(y_hat):
            xi = torch.unsqueeze(x[i], 0)
            logits = [model(xi) for model in self.models]
            logits = [logits[0] * item[0], logits[1] * item[1]]
            logits = torch.stack(logits, dim=1).mean(dim=1)
            logits_list[i] = logits

        return logits_list

        for model in self.models:
            output = model(x)
            outputs.append(output)
        weighted_outputs = [
            output * weight for output, weight in zip(outputs, self.weights)
        ]
        sum_weighted_outputs = sum(weighted_outputs)
        """
        return sum_weighted_outputs

        logits_list = [model(x) for model in self.models]
        # logits = torch.stack(logits_list, dim=1).sum(dim=1)
        # logits = torch.stack(logits_list, dim=1).mean(dim=1)
        logits = torch.cat(logits_list, dim=1)
        print(logits.shape)
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
        majority_class_recall_mean = torch.mean(
            class_recall[torch.gt(torch.tensor(self.class_count), 10)]
        )

        self.log(
            f"{phase}/BalancedAccuracyMinority",
            minority_class_recall_mean,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{phase}/BalancedAccuracyMajority",
            majority_class_recall_mean,
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
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(232),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

"""
# Data Module
dataset = Datasets.PLantCLEF2017Trusted
batch_size = 128 if dataset == Datasets.CRLeaves else 64  # 64  # 64  # 64  # 128
test_size = 0.5
minority_index = 18 if dataset == Datasets.CRLeaves else 4196
species_count = 254 if dataset == Datasets.CRLeaves else 10000
majority = False


# species_count = species_count - minority_index if majority else minority_index

class_weights = False
lr = 0.0003
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
    test_transform=test_transform,
)

# datamodule = PlantDataModuleSelfSupervision(root_dir=root_dir, batch_size=batch_size)

# datamodule = PlantDataModuleBinary(
#    dataset=dataset,
#    root_dir=root_dir,
#    batch_size=batch_size,
#    minority_index=minority_index,
#    test_size=test_size,
#    use_index=use_index,
#    sampling=sampling,
#    train_transform=train_transform,
#    test_transform=test_transform,
# )

# datamodule = PlantDataModuleMinority(
#    dataset=dataset,
#    root_dir=root_dir,
#    batch_size=batch_size,
#    minority_index=minority_index,
#    train_transform=train_transform,
#    test_transform=test_transform,
# )

# datamodule = PlantDataModuleMajority(
#    dataset=dataset,
#    root_dir=root_dir,
#    batch_size=batch_size,
#    minority_index=minority_index,
#    train_transform=train_transform,
#    test_transform=test_transform,
# )

datamodule.prepare_data()
datamodule.create_data_loaders()
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


module = Resnet50(out_features=species_count)
# module = Resnet50SelfSupervision(out_features=species_count)
# module = ModeloSelfSupervision.load_from_checkpoint(
#    "/home/ruben/Documents/Thesis/ThesisTest/875qkfwi/checkpoints/epoch=49-step=100100.ckpt",
#    lr=lr,
#    model=module,
#    criterion=torch.nn.CrossEntropyLoss(),
#    batch_size=batch_size,
# ).model

model = ModeloBase(
    lr=lr,
    model=module,
    criterion=criterion,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
    minority_index=minority_index,
    majority=majority,
)

# model = ModeloSelfSupervision(
#    lr=lr, model=module, criterion=criterion, batch_size=batch_size
# )

# model = ModeloBinario(
#    lr=lr,
#    model=module,
#    criterion=criterion,
#    class_count=datamodule.class_counts,
#    minority_index=minority_index,
# )

# model = ModeloBase.load_from_checkpoint(
#    "/home/ruben/Documents/Thesis/ThesisTest/aud73acx/checkpoints/epoch=29-step=60090.ckpt",
#    lr=lr,
#    model=Resnet50(out_features=10000),
#    criterion=torch.nn.CrossEntropyLoss(),
#    batch_size=batch_size,
#    class_count=datamodule.class_counts,
#    id=id,
#    species_count=species_count,
#    minority_index=minority_index,
# )

"""
"""
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
"""
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
minority_index = 18 if dataset == Datasets.CRLeaves else 4196

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
    test_transform=test_transform,
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

# R50 49qa0ona
# CN uzwl0smq
# EFB4 zrm49pqq

module1 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/Thesis/ThesisTest/49qa0ona/checkpoints/epoch=29-step=60090.ckpt",
    lr=lr,
    model=Resnet50(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
    minority_index=minority_index,
).model

module2 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/Thesis/ThesisTest/uzwl0smq/checkpoints/epoch=29-step=120150.ckpt",
    lr=lr,
    model=Convnext(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
    minority_index=minority_index,
).model


module3 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/Thesis/ThesisTest/zrm49pqq/checkpoints/epoch=29-step=60090.ckpt",
    lr=lr,
    model=EfficientNetB4(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
    minority_index=minority_index,
).model

# module3 = ModeloBinario.load_from_checkpoint(
#    "/home/ruben/Documents/Thesis/ThesisTest/lrw481em/checkpoints/epoch=29-step=840.ckpt",
#    lr=lr,
#    model=Resnet50(out_features=2),
#    criterion=torch.nn.CrossEntropyLoss(),
#    class_count=datamodule.class_counts,
#    minority_index=minority_index,
# ).model

model = ModeloEnsamblado(
    models=[module1, module2, module3],
    class_count=datamodule.class_counts,
    id=id,
    species_count=species_count,
)
# model = ModeloEnsamblado2(
#    binary_model=module3,
#    models=[module1, module2],
#    class_count=datamodule.class_counts,
#    id=id,
#    species_count=species_count,
# )

wandb_logger = WandbLogger(project="ThesisTest", id=id, resume="allow")

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=0,
    num_sanity_val_steps=0,
    precision=16,
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
wandb.finish()
