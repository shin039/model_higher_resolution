################################################################################
# Higher Resolution Model
################################################################################

# ------------------------------------------------------------------------------
# Import
# ------------------------------------------------------------------------------
import math
import numpy as np
import torch
import torchvision

from torch            import nn
from torch.utils.data import Subset
from torchvision      import transforms

import matplotlib.pyplot as plt

from PIL import Image

# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------
class ImgDataManager():

  def __init__(self):
    transform = transforms.Compose([ transforms.ToTensor() ])

    # CIFAR-10
    X_train = torchvision.datasets.CIFAR10(root='./data', train=True , download=False, transform=transform) # 50000枚 * [RGB * 32px * 32px][Category No]
    X_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform) # 10000枚 * [RGB * 32px * 32px][Category No]

    # Save Classees
    self.classes = X_train.classes

    ## Data features
    #print(len(X_test))
    #print(len(X_test[100]))
    #print(X_test[0][0].shape)
    #print(type(X_test[0][0]))
    #print(X_test[0][1])
    #print(X_test.classes)

    # NOTE: 学習を早く回したいので、X_trainとX_testの数を減らす。
    train_indices = list(range(320))
    test_indices  = list(range(32))
    self.X_train = Subset(X_train, train_indices)
    self.X_test  = Subset(X_test , test_indices )

  def show_image(self, print_num, cifar10_obj):
    print_num = min(len(cifar10_obj), 100)
    col       = 10 # 横に10枚の画像を並べて表示する
    row       = math.ceil(print_num / col)

    fig, axes = plt.subplots(row, col, figsize=(18, 18))

    for idx in range(print_num):
      _row_idx = idx // col
      _col_idx = idx % col
      _img     = cifar10_obj[idx][0]

      # NOTE: 512pxのイメージにする場合のコード
      #resize  = 512
      #resizer = transforms.Resize((resize, resize), interpolation=Image.BICUBIC)
      #_img = resizer(_img)

      axes[_row_idx, _col_idx].imshow(_img.permute(1, 2, 0)) # imgのTensorの配列の並びを変える。 C * W * H -> W * H * C
      axes[_row_idx, _col_idx].set_title(self.classes[cifar10_obj[idx][1]])
      axes[_row_idx, _col_idx].axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# ------------------------------------------------------------------------------
# Model Class
# ------------------------------------------------------------------------------
class SRCNN(nn.Module):

  def __init__(self):
    self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=9, padding=4)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=3 , kernel_size=5, padding=2)
    self.activate = nn.ReLU()

  def forward(self, x):
    h = self.activate(self.conv1(x))
    h = self.activate(self.conv2(h))
    return self.conv3(h)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
# モデル、損失関数、オプティマイザーの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

def main():
  img_mng = ImgDataManager()

  X_train = img_mng.X_train
  X_test  = img_mng.X_test

  img_mng.show_image(100, X_test)


if __name__ == '__main__':
  #multiprocessing.freeze_support()
  main()
