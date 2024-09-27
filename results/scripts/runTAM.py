import glob
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from triplanar_attention_mechanism import BinaryCrossEntropySorensenDiceLossFunction, TriplanarAttentionNetwork

# Dataset class handling MRI data
class Datamodule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.training_set, self.validation_set = random_split(
            self._Dataset(kwargs.get('root', '/home/vsharma/datasets')),
            kwargs.get('lengths', (524, 131)),
            torch.Generator().manual_seed(kwargs.get('seed', 42))
        )

    def train_dataloader(self):
        return DataLoader(self.training_set, self.hparams.get('batch_size', 4), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, self.hparams.get('batch_size', 4))

    # Dataset loader for inputs and targets
    class _Dataset(Dataset):
        def __init__(self, root):
            super().__init__()
            self.inputs    = sorted(glob.glob(os.path.join(root, "atlas/inputs/sub-*.nii.gz")))
            self.targets   = sorted(glob.glob(os.path.join(root, "atlas/targets/sub-*.nii.gz")))
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            input_img  = self._load(self.inputs[index])
            target_img = self._load(self.targets[index])
            seed       = np.random.randint(2147483647)  # Make a seed with numpy generator
            torch.manual_seed(seed)  # Apply this seed to input transforms
            input_img  = self.transform(input_img)
            torch.manual_seed(seed)  # Apply the same seed to target transforms
            target_img = self.transform(target_img)
            return input_img, target_img

        @staticmethod
        def _load(path):
            return torch.from_numpy(np.asarray(nib.load(path).dataobj, np.float32)).unsqueeze(0)


# Model definition including loss function and optimization
class Model(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss_function = BinaryCrossEntropySorensenDiceLossFunction(
            alpha  =kwargs.get('alpha', 0.7),
            epsilon=kwargs.get('epsilon', 1)
        )
        self.network = TriplanarAttentionNetwork(
            in_channels   =kwargs.get('in_channels', (1, 32, 64, 128)),
            out_channels  =kwargs.get('out_channels', 1),
            kernel_size   =kwargs.get('kernel_size', 4),
            stride        =kwargs.get('stride', 2),
            padding       =kwargs.get('padding', 1),
            output_padding=kwargs.get('output_padding', 0),
            dilation      =kwargs.get('dilation', 1),
            padding_mode  =kwargs.get('padding_mode', 'zeros'),
            negative_slope=kwargs.get('negative_slope', 0.2)
        )
        # Initialize weights for Conv3d and BatchNorm3d layers
        for module in self.network.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, std=kwargs.get('std', 0.02))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight, kwargs.get('mean', 1), kwargs.get('std', 0.02))
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return torch.sigmoid(self.network(x))

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'training')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'validation')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr   =self.hparams.get('lr', 0.0002),
            betas=self.hparams.get('betas', (0.5, 0.999))
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'validation/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _step(self, batch, loop):
        inputs, targets = batch
        outputs         = self.network(inputs)
        loss, bce, bsd  = self.loss_function(outputs, targets)
        self.log_dict({f'{loop}/loss': loss, f'{loop}/bce': bce, f'{loop}/bsd': bsd})
        return loss


# Script to run the model
if __name__ == '__main__':
    datamodule = Datamodule(batch_size=4)
    model = Model(alpha=0.7)

    # Model checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath               ='/home/vsharma/scripts/lesion_detection/checkpoints/',
        filename              ='{epoch}',
        save_top_k            =-1,     # Save all checkpoints
        mode                  ='min',  # Monitor minimum validation loss
        every_n_epochs        =50,
        enable_version_counter=True,
        save_last             =True    # Always save the last checkpoint
    )

    # CSV Logger
    csv_logger = CSVLogger('/home/vsharma/scripts/lesion_detection/logs/', name='logs')

    # Early stopping callback
    early_stopping_callback = EarlyStopping(monitor='validation/loss', patience=20, verbose=True, mode='min')

    # Trainer setup
    trainer = L.Trainer(
        callbacks =[checkpoint_callback, early_stopping_callback],
        max_epochs=500,
        logger    =csv_logger
    )

    # Start model training
    trainer.fit(model, datamodule=datamodule)
