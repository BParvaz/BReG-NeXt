# Copyright (c) 2021 Regents of the University of California
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from argparse import ArgumentParser

import torch
import torchvision
import tfrecord
import numpy as np

# Old
"""
import pytorch_lightning
"""
# New
import pytorch_lightning
from pytorch_lightning.cli import LightningCLI

from BReGNeXt import BReGNeXt
from utils import ShuffleDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def focal_loss2(input_tensor, target_tensor, weight=None, gamma=2, reduction='mean'):
    log_prob = torch.nn.functional.log_softmax(input_tensor, dim=-1)
    probs = torch.exp(log_prob)
    return torch.nn.functional.nll_loss(((1 - probs) ** gamma) * log_prob,
                                        target_tensor, weight=weight, reduction=reduction)


_image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=(0, 0.05), contrast=(0.7,1.3), saturation=(0.6, 1.6), hue=0.08),
    torchvision.transforms.RandomResizedCrop((64,64)),
    torchvision.transforms.ToTensor(),
])


def decode_and_preprocess_image(features):
    print(features['label'])
    # Decode the image
    
    #Convert bytes to numpy array
    img = np.frombuffer(features['image_raw'],dtype=np.uint8)
    img = img.reshape(64, 64, 3)
    
    #And back to image
    img = img.astype(np.float32)/255.0
    
    # Convert to "torch tensor CHW"
    img_tensor = torch.from_numpy(img).permute(2,0,1)
    
    features['image_raw'] = img_tensor

    features['image_raw'] = _image_transform(features['image_raw'])
    features['image_raw'] = features['image_raw'] - torch.FloatTensor([0.5727663, 0.44812188, 0.39362228]).unsqueeze(-1).unsqueeze(-1)

    features['label'] = torch.tensor(features['label'],dtype=torch.long).squeeze()
    return features




class BReGNeXtPTLDriver(pytorch_lightning.LightningModule):
    def __init__(self, use_focal_loss = False, learning_rate = 0.0001):

        super(BReGNeXtPTLDriver, self).__init__()

        self._use_focal_loss = use_focal_loss
        self.learning_rate = learning_rate
        self._model = BReGNeXt()

    def training_step(self, batch, batch_idx):
        logits = self._model(batch['image_raw'])
        batch['label'] = batch['label'].reshape(-1)
        loss = focal_loss2(logits, batch['label']) if self._use_focal_loss else torch.nn.functional.cross_entropy(logits, batch['label'])
        accuracy = (logits.argmax(dim=-1) == batch['label']).float().mean()

        self.log('train/accuracy', accuracy, prog_bar=True,on_step=False, on_epoch=True)
        self.log('train/loss', loss, prog_bar=True,on_step=False,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._model(batch['image_raw'])
        batch['label'] = batch['label'].reshape(-1)
        loss = focal_loss2(logits, batch['label']) if self._use_focal_loss else torch.nn.functional.cross_entropy(logits, batch['label'])
        accuracy = (logits.argmax(dim=-1) == batch['label']).float().mean()

        self.log('val/accuracy', accuracy, prog_bar=True,on_step=False,on_epoch=True)
        self.log('val/loss', loss, prog_bar=True,on_step=False,on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)
        return [optimizer], [scheduler]


class FERDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, train_data_path: str, val_data_path: str, batch_size: int = 64):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size

    # to be implemented later

    def setup(self, stage=None):
        # load TFRecords, preprocess, etc.
        pass

    def train_dataloader(self):
        # return your DataLoader
        pass

    def val_dataloader(self):
        # return your DataLoader
        pass


if __name__ == '__main__':

    # deprecated, needs to be changed to implement argparsing
    """
        parser = ArgumentParser()
        parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss when training.')
        parser.add_argument('--train_data_path', type=str, help='The path to the .tfrecords file for training.', required=True)
        parser.add_argument('--val_data_path', type=str, help='The path to the .tfrecords file for evaluation.', required=True)
        parser = pytorch_lightning.Trainer.add_argparse_args(parser)
        args = parser.parse_args()
    """

    # indexpath = "train.tfrecords.index"
    # use relative directories rather than argparsing

    realtrain ="../../tfrecords/training_FER2013_sample.tfrecords"
    realval = "../../tfrecords/validation_FER2013_sample.tfrecords"

    testtrain = "../../jobscripts/training_single_class.tfrecords"
    testval = "../../jobscripts/validation_single_class.tfrecords"

    
    train_dataset = ShuffleDataset(tfrecord.torch.dataset.TFRecordDataset(
        data_path=realtrain,
        index_path=None,
        description={'image_raw': 'byte', 'label': 'int'},
        transform=decode_and_preprocess_image,
    ), 1024)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=4)

    valid_dataset = tfrecord.torch.dataset.TFRecordDataset(
        data_path=realval,
        index_path=None,
        description={'image_raw': 'byte', 'label': 'int'},
        transform=decode_and_preprocess_image,
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, num_workers=4)






    # Fit the model to the trainer.
    
    # use_focal_loss is defaulted to true, needs to be conv-ed to arg
    model = BReGNeXtPTLDriver(use_focal_loss=True)

    # Define conditions for early stopping
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
        min_delta=1e-5,
    )

    # Use the TensorBoard logger
    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir="logs",
        name="my_experiment"  # this becomes part of the folder
    )


    trainer = pytorch_lightning.Trainer(
        callbacks=[early_stopping],
        max_epochs=50,
        # devices instead of gpus
        # gpus
        devices = 1,
        accelerator = 'gpu',
        logger = logger
    )

    labelsA = []
    for i, sample in enumerate(train_dataset.dataset):
        label = sample['label']     # get the actual label tensor
        print(label)
        labelsA.append(label.item() if torch.is_tensor(label) else label)
        if i > 500:
            break

    print("Unique training labels seen:", set(labelsA))

    labelsB = []
    for i, sample in enumerate(valid_dataset):
        label = sample['label']
        print(label)
        labelsB.append(label.item() if torch.is_tensor(label) else label)
        if i > 500:
            break
    print("Unique validation labels seen:", set(labelsB))




    trainer.fit(model, train_dataloader, valid_dataloader)
