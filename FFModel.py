from pathlib import Path

import albumentations as A
import albumentations.pytorch.transforms as Atorch
import h5py
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import xarray as xr


def get_fire_bands(data):
    
    blue = data[..., 1]
    green = data[..., 2]
    red = data[..., 3]

    nir = data[..., 7]
    veg_red_4 = data[..., 8]

    swir_1 = data[..., 10]
    swir_2 = data[..., 11]

    # Indices taken from https://www.mdpi.com/2072-4.astype("float32") / 10000.092/14/7/1727
    #
    # NBR
    nbr = (swir_2 - veg_red_4) / (swir_2 + veg_red_4 + 1.0e-8)

    # NDVI
    z1 = nir - red
    n1 = nir + red
    ndvi = np.divide(z1, n1, out=np.zeros_like(z1), where=n1 != 0.0)

    bands = [nir, swir_1, swir_2, nbr, veg_red_4, ndvi]
    return bands



class FFDataSet(torch.utils.data.Dataset):
    def __init__(self, filename, folds=(0, 1, 2, 3, 4),
                 channels=[],
                 include_pre=False,
                 transform=None) -> None:
        self._filename = filename
        self._fd = h5py.File(filename, "r")
        self._channels = channels
        self._transform = transform
        self._names = []
        for name in self._fd:
            if self._fd[name].attrs["fold"] not in folds:
                continue
            self._names.append((name, "post_fire"))
            if include_pre and "pre_fire" in self._fd[name]:
                pre_image = self._fd[name]["pre_fire"][...]
                # Include only "real" pre_fire images
                if np.mean(pre_image > 0) > 0.8:
                    self._names.append((name, "pre_fire"))
            

    def number_of_channels(self):
        return len(self._channels)

    def __getitem__(self, idx):
        name, state = self._names[idx]
        data = self._fd[name][state][...].astype("float32") / 10000.0
        if state == "pre_fire":
            mask = np.zeros((512, 512), dtype="float32")
        else:
            mask = self._fd[name]["mask"][..., 0].astype("float32")
       
        bands = get_fire_bands(data)
        # Stack indices into a new image in CHW format.
        image =  np.stack(bands)
        
        if self._transform:
            # Transpose image so we get HWC instead of CHW format.
            # Transform is responsible for transposing back as required by PyTorch.
            image = image.transpose((1, 2, 0))
            xfrm = self._transform(image=image, mask=mask)
            image, mask = xfrm["image"], xfrm["mask"]

        return {"image": image, "mask": mask[None, :]}

    def __len__(self) -> int:
        return len(self._names)






class FFModel(pl.LightningModule):
    def __init__(self, *,
                 datafile,
                 model,
                 encoder,
                 encoder_depth,
                 encoder_weights,
                 loss,
                 channels,
                 train_transform,
                 train_use_pre_fire,
                 n_cpus=1,
                 batch_size,
                 lr=0.00025,
                 max_epochs=50,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.datafile = datafile
        self.lr = lr
        self.channels = channels
        if model == "unet":
            decoder_channels = [2**(8 - d) for d in range(encoder_depth, 0, -1)]
            self.model = smp.Unet(encoder_name=encoder, encoder_depth=encoder_depth, encoder_weights=encoder_weights,
                                  decoder_channels=decoder_channels,
                                  in_channels=len(channels), classes=1)
        elif model == "unetpp":
            decoder_channels = [2**(8 - d) for d in range(encoder_depth, 0, -1)]
            self.model = smp.UnetPlusPlus(encoder_name=encoder, encoder_depth=encoder_depth, encoder_weights=encoder_weights,
                                          decoder_channels=decoder_channels,
                                          in_channels=len(channels), classes=1)
        elif model == "fpn":
            if encoder_depth == 3:
                upsampling = 1
            elif encoder_depth == 4:
                upsampling = 2
            elif encoder_depth == 5:
                upsampling = 4
            else:
                raise "FPN: Unsupported encoder depth {encoder_depth}."
            self.model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                 upsampling=upsampling,
                                 in_channels=len(channels), classes=1)
        elif model == "dlv3":
            self.model = smp.DeepLabV3(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                       in_channels=len(channels), classes=1)
        elif model == "dlv3p":
            if encoder_depth != 5:
                raise f"Unsupported encoder depth {encoder_depth} for DeepLabV3+ (must be 5)."
            self.model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                           in_channels=len(channels), classes=1)
        else:
            raise f"Unsupported model '{model}'."

        if loss == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss == "bce":
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        else:
            raise f"Unsupported loss function '{loss}'."

        self.train_transform = train_transform
        self.train_use_pre_fire = train_use_pre_fire
        self.n_cpus = n_cpus
        self.batch_size = batch_size

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch["image"], batch["mask"]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask.long(), mode="binary")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def train_dataloader(self):
        train_ds = FFDataSet(self.datafile, folds=[1, 2, 3, 4],
                                channels=self.channels,
                                transform=self.train_transform,
                                include_pre=self.train_use_pre_fire)
        train_dl = torch.utils.data.DataLoader(train_ds,
                                               batch_size=self.batch_size,
                                               num_workers=self.n_cpus,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=False)
        return train_dl

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def val_dataloader(self):
        val_ds = FFDataSet(self.datafile, folds=[0],
                              channels=self.channels,
                              transform=None,
                              include_pre=False)
        val_dl = torch.utils.data.DataLoader(val_ds,
                                             batch_size=self.batch_size,
                                             num_workers=self.n_cpus,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)
        return val_dl

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        # TODO: Can we do better? We should probably implement a learning rate schedule?
        return torch.optim.Adam(self.parameters(), lr=self.lr)