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

# Scipy
from scipy.stats import pearsonr, spearmanr, kendalltau, boxcox

# Sklearn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
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

"""#Datasets"""


class Datasets(Enum):
    CRLeaves = 1
    PLantCLEF2017Trusted = 2


class Sampling(Enum):
    NUMPY = 1
    SKLEARN = 2
    NONE = 3


DATASET = Datasets.PLantCLEF2017Trusted

if DATASET == Datasets.CRLeaves:
    basePath = "CRLeaves/"
elif DATASET == Datasets.PLantCLEF2017Trusted:
    basePath = "data/"

speciesId = [id for id in os.listdir(basePath)]
print(len(speciesId))

a = []
images = {}
for id in speciesId:
    images[id] = sum(
        [
            1
            for fileName in os.listdir(basePath + id)
            if (fileName.endswith(".jpg") or fileName.endswith(".JPG"))
        ]
    )

images = {k: v for k, v in sorted(images.items(), key=lambda x: x[1])}
print(images)
classCount = list(images.values())
print("Total images: ", sum(images.values()))

df = pd.DataFrame()
df["Count"] = classCount
df["Class"] = range(len(speciesId))
print(df["Count"].describe())
plt.scatter(df["Class"], df["Count"])
plt.xlabel("Species")
plt.ylabel("Images")
plt.show()

"""##Dataset"""


class PlantsDataset(torch.utils.data.Dataset):
    def __init__(self, transform, speciesCount):
        self.transform = transform
        self.speciesCount = speciesCount
        self.basePath = basePath
        self.folders = [folder for folder in os.listdir(basePath)]
        self.images = {}
        self.targets = []
        imagesCount = {}
        for i, folder in enumerate(self.folders):
            self.images[folder] = [
                x
                for x in os.listdir(self.basePath + "/" + folder)
                if (x.endswith(".jpg") or x.endswith(".JPG"))
            ]
            imagesCount[folder] = len(self.images[folder])

        imagesCount = {k: v for k, v in sorted(imagesCount.items(), key=lambda x: x[1])}
        self.folders = list(imagesCount.keys())
        self.classCount = list(imagesCount.values())

        for i in range(len(self.folders)):
            self.targets += [i] * self.classCount[i]

    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.folders])

    def __getitem__(self, index):
        for i in range(self.speciesCount):
            index -= self.classCount[i]
            if index < 0:
                folder = self.folders[i]
                imageName = self.images[folder][index + self.classCount[i]]
                imagePath = os.path.join(
                    self.basePath + str(self.folders[i]), imageName
                )
                image = Image.open(imagePath).convert("RGB")
                image = self.transform(image)
                return image, i


"""##DataModule"""


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        test_size,
        species_count,
        class_weigths,
        sampling,
        data_transforms,
        use_index,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.species_count = species_count
        self.class_weigths = class_weigths
        self.sampling = sampling
        self.data_transforms = data_transforms
        self.use_index = use_index
        self.idxPATH = "idxPATH/" + str(self.dataset) + ".pkl"

        if self.data_transforms:
            self.transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(45),
                    # torchvision.transforms.ColorJitter(brightness=20, saturation=32),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.transform_val = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        self.train_dataset = PlantsDataset(self.transform_train, self.species_count)
        self.val_dataset = PlantsDataset(self.transform_val, self.species_count)

        if self.use_index:
            a_file = open(self.idxPATH, "rb")
            checkpoint = pickle.load(a_file)
            self.train_idx, self.val_idx = (
                checkpoint["train_idx"],
                checkpoint["val_idx"],
            )
            a_file.close()
        else:
            num_samples = self.train_dataset.__len__()
            self.train_idx, self.val_idx = train_test_split(
                np.arange(num_samples),
                test_size=self.test_size,
                stratify=self.train_dataset.targets,
            )
            a_file = open(self.idxPATH, "wb")
            pickle.dump({"train_idx": self.train_idx, "val_idx": self.val_idx}, a_file)
            a_file.close()

        self.train_sampler = torch.utils.data.SubsetRandomSampler(self.train_idx)
        self.val_sampler = torch.utils.data.SubsetRandomSampler(self.val_idx)

        if self.class_weigths:
            targets = np.take(self.train_dataset.targets, self.train_idx)
            unique = np.arange(len(self.train_dataset.folders))
            self.class_weights = torch.tensor(
                class_weight.compute_class_weight(
                    class_weight="balanced", classes=unique, y=targets
                ),
                dtype=torch.float,
            )

        elif self.sampling != Sampling.NONE:
            y_train = [self.train_dataset.targets[i] for i in self.train_idx]
            y_valid = [self.train_dataset.targets[i] for i in self.val_idx]
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset, self.train_idx
            )
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )
            if self.sampling == Sampling.NUMPY:
                weight = 1.0 / class_sample_count
            else:
                weight = class_weight.compute_class_weight(
                    class_weight="balanced", classes=np.unique(y_train), y=y_train
                )
            samples_weight = np.array([weight[t] for t in y_train])
            samples_weight = torch.from_numpy(samples_weight)
            self.train_sampler = WeightedRandomSampler(
                weights=samples_weight.type("torch.DoubleTensor"),
                num_samples=len(samples_weight),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=8,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=8,
            persistent_workers=True,
        )


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


class DataModuleSelfSupervision(pl.LightningDataModule):
    def __init__(self, batch_size, species_count):
        super().__init__()
        self.batch_size = batch_size
        self.species_count = species_count
        self.transform = ContrastiveLearningViewGenerator(simclr_transform(224), 2)

    def prepare_data(self):
        self.dataset = PlantsDataset(self.transform, self.species_count)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=10,
            persistent_workers=True,
        )


"""#Metrics"""


class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
        self.add_state("rr", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ranks = self._mrr(preds, target, k=30)
        self.rr = torch.cat((self.rr, ranks), 0)
        return ranks.mean()

    def compute(self):
        return self.rr.mean()

    def _mrr(self, outputs: torch.Tensor, targets: torch.Tensor, k=100) -> torch.Tensor:
        k = min(outputs.size(1), k)
        targets = F.one_hot(targets, num_classes=outputs.size(1))
        _, indices_for_sort = outputs.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(targets, dim=-1, index=indices_for_sort)
        true_sorted_by_pred_shrink = true_sorted_by_preds[:, :k]

        values, indices = torch.max(true_sorted_by_pred_shrink, dim=1)
        indices = indices.type_as(values).unsqueeze(dim=0).t()
        result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

        zero_sum_mask = values == 0.0
        result[zero_sum_mask] = 0.0
        return result


"""#Models

##Resnet50
"""


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

from torchmetrics.functional import accuracy


class ModeloSelfSupervision(pl.LightningModule):
    def __init__(self, model, criterion, *args, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
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
            self.model.parameters(), lr=0.0003, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


class ModeloBase(pl.LightningModule):
    def __init__(self, model, criterion, *args, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.resultsPATH = "resultsPATH/" + ID + ".pkl"
        a_file = open(self.resultsPATH, "wb")
        pickle.dump({"ClassCount": classCount}, a_file)
        a_file.close()
        self.train_mrr = MRR()
        self.val_mrr = MRR()
        self.test_mrr = MRR()

    def forward(self, x):
        return self.model(x)

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def on_train_epoch_start(self):
        self.predsTrain = torch.zeros(0, dtype=torch.long, device="cpu")
        self.ysTrain = torch.zeros(0, dtype=torch.long, device="cpu")

    def on_validation_epoch_start(self):
        self.predsVal = torch.zeros(0, dtype=torch.long, device="cpu")
        self.ysVal = torch.zeros(0, dtype=torch.long, device="cpu")

    def on_test_epoch_start(self):
        self.predsTest = torch.zeros(0, dtype=torch.long, device="cpu")
        self.ysTest = torch.zeros(0, dtype=torch.long, device="cpu")

    def on_train_epoch_end(self):
        self.correlation(self.ysTrain.numpy(), self.predsTrain.numpy(), "train")

    def on_validation_epoch_end(self):
        self.correlation(self.ysVal.numpy(), self.predsVal.numpy(), "val")

    def on_test_epoch_end(self):
        self.correlation(self.ysTest.numpy(), self.predsTest.numpy(), "test")

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self(xs)
        p = torch.nn.functional.softmax(outputs, dim=1)
        self.train_mrr(p, ys)
        loss = self.loss(outputs, ys)
        preds = torch.argmax(outputs, dim=1)
        ys, preds = ys.cpu(), preds.cpu()
        self.predsTrain = torch.cat([self.predsTrain, preds])
        self.ysTrain = torch.cat([self.ysTrain, ys])
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train/general_acc", accuracy_score(ys, preds), on_step=True, on_epoch=True
        )
        self.log("train/mrr", self.train_mrr, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975, verbose=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0003, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, verbose=True
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self(xs)
        p = torch.nn.functional.softmax(outputs, dim=1)
        self.val_mrr(p, ys)
        loss = self.loss(outputs, ys)
        preds = torch.argmax(outputs, dim=1)
        ys, preds = ys.cpu(), preds.cpu()
        self.predsVal = torch.cat([self.predsVal, preds])
        self.ysVal = torch.cat([self.ysVal, ys])
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "val/general_acc", accuracy_score(ys, preds), on_step=True, on_epoch=True
        )
        self.log("val/mrr", self.val_mrr, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self(xs)
        p = torch.nn.functional.softmax(outputs, dim=1)
        self.test_mrr(p, ys)
        loss = self.loss(outputs, ys)
        preds = torch.argmax(outputs, dim=1)
        ys, preds = ys.cpu(), preds.cpu()
        self.predsTest = torch.cat([self.predsTest, preds])
        self.ysTest = torch.cat([self.ysTest, ys])
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test/general_acc", accuracy_score(ys, preds), on_step=False, on_epoch=True
        )
        self.log("test/mrr", self.test_mrr, on_step=False, on_epoch=True)
        return loss

    def metrics(self, ys, preds, phase):
        balanced_acc = balanced_accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, average="weighted")
        cohen_kappa = cohen_kappa_score(ys, preds)
        matthews = matthews_corrcoef(ys, preds)

        self.log(f"{phase}/balanced_acc", balanced_acc)
        self.log(f"{phase}/f1", f1)
        self.log(f"{phase}/cohen_kappa", cohen_kappa)
        self.log(f"{phase}/matthews", matthews)
        indices = (ys < MINORITY_INDEX).nonzero()
        ys_minority = np.take(ys, indices)
        preds_minority = np.take(preds, indices)
        self.log(
            f"{phase}/balanced_acc_minority",
            balanced_accuracy_score(ys_minority[0], preds_minority[0]),
        )

    def correlation(self, ys, preds, phase):
        self.metrics(ys, preds, phase)

        cm = confusion_matrix(ys, preds)
        x = np.take(classCount, np.unique(np.append(ys, preds)))
        acc = np.nan_to_num(cm.diagonal() / cm.sum(1))
        plot = plt.figure()
        ax = plt.gca()
        ax.scatter(x, acc)
        ax.set_xscale("log")
        ax.axvline(10)
        ax.set_ylim((0, 1))
        for axis in [ax.xaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.xlabel("Image Count")
        plt.ylabel("Accuracy")
        self.logger.experiment.log(
            {
                f"{phase}/acc_corr": [
                    wandb.Image(
                        plot,
                        caption=f"Acc : {round(np.sum(cm.diagonal()) / np.sum(cm),3)}, Pearson: {round(pearsonr(x, acc)[0],3)}, Spearman: {round(spearmanr(x, acc)[0], 3)}, Kendall: {round(kendalltau(x, acc)[0], 3)}",
                    )
                ]
            }
        )

        a_file = open(self.resultsPATH, "rb")
        checkpoint = pickle.load(a_file)
        a_file.close()
        a_file = open(self.resultsPATH, "wb")
        checkpoint[f"{self.current_epoch}-{phase}-preds"] = preds
        checkpoint[f"{self.current_epoch}-{phase}-ys"] = ys
        pickle.dump(checkpoint, a_file)
        a_file.close()


# Training

"""
CLASS_WEIGTHS = False
SAMPLING = Sampling.NONE
DATA_TRANSFORMS = True
MINORITY_INDEX = 18 if DATASET == Datasets.CRLeaves else 4196
SPECIES_COUNT = 254 if DATASET == Datasets.CRLeaves else 10000
TEST_SIZE = 0.5
BATCH_SIZE = 128
USE_INDEX = True
ID = None

datamodule = DataModule(
    dataset=DATASET,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
    species_count=SPECIES_COUNT,
    class_weigths=CLASS_WEIGTHS,
    sampling=SAMPLING,
    data_transforms=DATA_TRANSFORMS,
    use_index=USE_INDEX,
)
datamodule.prepare_data()
CRITERION = (
    torch.nn.CrossEntropyLoss(weight=datamodule.class_weights)
    if CLASS_WEIGTHS
    else torch.nn.CrossEntropyLoss()
)
if ID is None:
    ID = wandb.util.generate_id()
print(ID)

wandb_logger = WandbLogger(project="Thesis", id=ID, resume="allow")

module = Resnet101(out_features=SPECIES_COUNT)
model = ModeloBase(model=module, criterion=CRITERION)
trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=-1, max_epochs=30)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
wandb.finish()
"""

SPECIES_COUNT = 254 if DATASET == Datasets.CRLeaves else 10000
BATCH_SIZE = 128
ID = None

datamoduleSELF = DataModuleSelfSupervision(
    batch_size=BATCH_SIZE, species_count=SPECIES_COUNT
)
datamoduleSELF.prepare_data()
CRITERION = torch.nn.CrossEntropyLoss()
if ID is None:
    ID = wandb.util.generate_id()
print(ID)

wandb_logger = WandbLogger(project="ThesisTest", id=ID, resume="allow")

# module = Resnet50SelfSupervision(out_features=SPECIES_COUNT)
module = Resnet50(out_features=SPECIES_COUNT)
model = ModeloSelfSupervision(model=module, criterion=CRITERION)


from pytorch_lightning.callbacks import LearningRateMonitor

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=50,
    precision=16,
    callbacks=[LearningRateMonitor(logging_interval="epoch")],
)


trainer.fit(
    model,
    datamodule=datamoduleSELF,
    # ckpt_path="/home/ruben/Documents/Thesis/Thesis/2iohxu26/checkpoints/epoch=30-step=62062.ckpt",
)
wandb.finish()
"""

CLASS_WEIGTHS = False
SAMPLING = Sampling.NONE
DATA_TRANSFORMS = True
MINORITY_INDEX = 18 if DATASET == Datasets.CRLeaves else 4196
SPECIES_COUNT = 254 if DATASET == Datasets.CRLeaves else 10000
TEST_SIZE = 0.5
BATCH_SIZE = 256
USE_INDEX = True
ID = None


datamodule = DataModule(dataset=DATASET,batch_size=BATCH_SIZE, test_size=TEST_SIZE, species_count=SPECIES_COUNT, class_weigths=CLASS_WEIGTHS, sampling=SAMPLING, data_transforms=DATA_TRANSFORMS, use_index=USE_INDEX)
datamodule.prepare_data()
CRITERION = torch.nn.CrossEntropyLoss(weight=datamodule.class_weights) if CLASS_WEIGTHS else torch.nn.CrossEntropyLoss()
if ID is None:
  ID = wandb.util.generate_id()
print(ID)

wandb_logger = WandbLogger(project="Thesis", id=ID, resume="allow")

#module = Resnet50SelfSupervision(out_features=SPECIES_COUNT)
module = Resnet50(out_features=SPECIES_COUNT)
model = ModeloSelfSupervision.load_from_checkpoint("/home/ruben/Documents/Thesis/Thesis/2iohxu26/checkpoints/epoch=34-step=70070.ckpt",model=module, criterion=torch.nn.CrossEntropyLoss())

model2 = ModeloBase(model=model.model, criterion=CRITERION)

trainer = pl.Trainer(
    logger = wandb_logger,    
    accelerator = 'gpu',
    devices = -1,              
    max_epochs = 30
)

trainer.fit(model2, datamodule=datamodule)#, ckpt_path="/content/drive/MyDrive/Thesis/Experiments/RE/Thesis/1t2zksuu/checkpoints/epoch=9-step=40049.ckpt")
trainer.test(model2, datamodule=datamodule)
wandb.finish()
"""
