import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from lightly.data import collate, LightlyDataset
from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
from stainlib.augmentation.augmenter import  HedColorAugmenter1
import torchvision.transforms as T
from lightly.transforms import RandomRotate

path_to_data = '/path/to/data/folder'
path_to_model = '/path/to/model.ckpt'

num_workers = 16
batch_size = 64
seed = 1
max_epochs = 25
input_size = 224

pl.seed_everything(seed)

class HedColorAug:
    def __init__(self, hed_thresh = 0.03):
        self.hed_thresh = hed_thresh
    def __call__(self, image):
        hed_lighter_aug =  HedColorAugmenter1(self.hed_thresh)
        hed_lighter_aug.randomize()
        return Image.fromarray(hed_lighter_aug.transform(np.array(image)))


imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class ImageCollateFunction(collate.BaseCollateFunction):
    def __init__(self,
                 input_size: int = 64,
                 min_scale: float = 0.15,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 hed_thresh: float = 0.3,
                 normalize: dict = imagenet_normalize):

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        transform = [T.RandomResizedCrop(size=input_size,
                                         scale=(min_scale, 1.0)),
                    RandomRotate(prob=rr_prob),
                    T.RandomHorizontalFlip(p=hf_prob),
                    T.RandomVerticalFlip(p=vf_prob),
                    HedColorAug(hed_thresh=hed_thresh),
                    T.ToTensor()
        ]

        if normalize:
            transform += [
             T.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
             ]
           
        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)


collate_fn = ImageCollateFunction(input_size = input_size,
                                 min_scale = 0.25,
                                 vf_prob = 0.5,
                                 hf_prob = 0.5,
                                 rr_prob = 0.5,
                                 hed_thresh = 0.3)

dataset_train_simclr = LightlyDataset(
    input_dir=path_to_data
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers,
    persistent_workers=True
)

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(path_to_model)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        resnet = load_model_weights(resnet, state_dict)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=0.00001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]

gpus = torch.cuda.device_count()
print(gpus)

model = SimCLRModel()

for name, p in model.named_parameters():
    if "backbone" in name:
        p.requires_grad = False


trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    devices=gpus,
    num_nodes=1
    )

trainer.fit(model, dataloader_train_simclr)
