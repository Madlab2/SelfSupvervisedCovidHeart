from monai.networks.nets import UNet
import torch
import monai
from config import *




model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=DROPOUT,  # Read about dropout here: https://www.deeplearningbook.org/contents/regularization.html#pf20
)

loss_fn = monai.losses.MaskedDiceLoss(include_background=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()