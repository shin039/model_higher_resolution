import torch
import torchvision

import torchvision.datasets as dset
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
from torchvision.datasets import MNIST
import os
import math
import pylab
import matplotlib.pyplot as plt

from PIL import Image


num_epochs    = 100  # エポック数
batch_size    = 100  # バッチサイズ
learning_rate = 1e-3 # 学習率

train      = True  # 学習を行うかどうかのフラグ
pretrained = False # 事前に学習したモデルがあるならそれを使う
save_img   = True  # ネットワークによる生成画像を保存するかどうのフラグ

def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 3, x.shape[2], x.shape[3])
  return x

#データセットを調整する関数
resize = 64
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, ), (0.5, )),
  transforms.Resize((resize, resize), interpolation=Image.BICUBIC)
])

# --------------------------------------------------------------------------------
#訓練用データセット
# --------------------------------------------------------------------------------
# origin: 
#dataset = dset.ImageFolder(root='./drive/My Drive/face/',
#                              transform=transforms.Compose([
#                              transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
#                              transforms.RandomHorizontalFlip(),
#                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#                              transforms.ToTensor(),
#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                          ])) 

cf_10_test     = torchvision.datasets.CIFAR10(root  = './data', train = False, download = False, transform = transform)
train_data_num = 100
test_indices   = list(range(train_data_num))
dataset        = Subset(cf_10_test , test_indices )

#データセットをdataoaderで読み込み
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------------------------------------------------------------
#低解像度画像を高解像度化するニューラルネットワーク
# --------------------------------------------------------------------------------
class SizeDecoder(nn.Module):
    def __init__(self):
        super(SizeDecoder, self).__init__()
        nch_g = 64

        # - - - - - - - - - - - - -
        # エンコーダ部分
        # - - - - - - - - - - - - -
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, nch_g, 3, 2, 1),
            nn.BatchNorm2d(nch_g),
            nn.ReLU()
        )  # (3, 64, 64) -> (64, 32, 32)

        self.layer2 = nn.Sequential(
            nn.Conv2d(nch_g, nch_g*2, 3, 2, 1),
            nn.BatchNorm2d(nch_g*2),
            nn.ReLU()
        )  # (64, 32, 32) -> (128, 16, 16)

        self.layer3 = nn.Sequential(
            nn.Conv2d(nch_g*2, nch_g*4, 3, 2, 1),
            nn.BatchNorm2d(nch_g*4),
            nn.ReLU()
        )  # (128, 16, 16) -> (256, 8, 8)

        self.layer4 = nn.Sequential(
            nn.Conv2d(nch_g*4, nch_g*8, 3, 2, 1),
            nn.BatchNorm2d(nch_g*8),
            nn.ReLU()
        )  # (256, 8, 8) -> (512, 4, 4)

        # - - - - - - - - - - - - -
        # デコーダ部分
        # - - - - - - - - - - - - -
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(nch_g*8, nch_g*4, 4, 2, 1),
            nn.BatchNorm2d(nch_g*4),
            nn.ReLU()
        )  # (512, 4, 4) -> (256, 8, 8)

        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(nch_g*4, nch_g*2, 4, 2, 1),
            nn.BatchNorm2d(nch_g*2),
            nn.ReLU()
        )  # (256, 8, 8) -> (128, 16, 16)

        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(nch_g*2, nch_g, 4, 2, 1),
            nn.BatchNorm2d(nch_g),
            nn.ReLU()
        )  # (128, 16, 16) -> (64, 32, 32)

        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(nch_g, int(nch_g/2), 4, 2, 1),
            nn.BatchNorm2d(int(nch_g/2)),
            nn.Tanh()
        )  # (64, 32, 32) -> (32, 64, 64)

        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(int(nch_g/2), 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )  # (32, 64, 64) -> (3, 128, 128)

    def forward(self, z):
        z = self.layer1(z)
        z1 = z
        z = self.layer2(z)
        z2 = z
        z = self.layer3(z)
        z3 = z
        z = self.layer4(z)

        z = self.layer7(z) + z3
        z = self.layer8(z) + z2
        z = self.layer9(z) + z1
        z = self.layer10(z)
        z = self.layer11(z)
        return z

class SRCNN(nn.Module):
  def __init__(self):
    super(SRCNN, self).__init__()
    self.conv1    = nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=9, padding=4)
    self.conv2    = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
    self.conv3    = nn.Conv2d(in_channels=32, out_channels=3 , kernel_size=5, padding=2)
    self.activate = nn.ReLU()

  def forward(self, x):
    print("### 1 ", x.shape)
    h = self.activate(self.conv1(x))
    print("### 2 ", h.shape)
    h = self.activate(self.conv2(h))
    print("### 3 ", h.shape)
    h = self.conv3(h)
    print("### 4 ", h.shape)
    return h

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():

    #もしGPUがあるならGPUを使用してないならCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #ネットワークを呼び出し
    #model = SizeDecoder().to(device)
    model = SRCNN().to(device)

    #事前に学習しているモデルがあるならそれを読み込む
    if pretrained:
        param = torch.load('./Size_Decoder.pth')
        model.load_state_dict(param)

    #誤差関数には二乗誤差を使用
    criterion = nn.MSELoss()

    #更新式はAdamを適用
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    loss_train_list = []
    loss_test_list= []
    for epoch in range(num_epochs):

        print(epoch)

        for data in dataloader:

            img, num = data
            #img --> [batch_size,1,32,32]
            #imgは元画像
            #imgをGPUに載せる
            img = img.to(device)

            print("### original size =>", img.shape)

            # ===================forward=====================

            #_imgは高解像度画像を低解像度に変換した画像
            # TODO: DEBUG: for SCRNN
            #_img = (img[:,:,::2, ::2] + img[:,:,1::2, ::2] + img[:,:,::2, 1::2] + img[:,:,1::2, 1::2])/4
            _img = img


            _img =_img.to(device)

            print("### input size =>", _img.shape)

            #ネットワークの出力結果
            output = model(_img)
            print("output.shape => ", output.shape)
            #もし学習するなら
            if train:
                #ネットワークの出力と高解像度画像との誤差を損失として学習

                # ===================backward====================
                loss = criterion(output, img)
                print("loss => ", loss)
                #勾配を初期化
                optimizer.zero_grad()

                #微分値を計算
                loss.backward()

                #パラメータを更新
                optimizer.step()


            else:#学習しないなら
                break
        # ===================log========================

        # モデルは保存しない
        #if train == True:
        #   #モデルを保存
        #   torch.save(model.state_dict(), './Size_Decoder.pth')


    #もし生成画像を保存するなら
    if save_img:
        value = int(math.sqrt(batch_size))

        pic = to_img(img.cpu().data)
        pic = torchvision.utils.make_grid(pic,nrow = value)
        save_image(pic, './real_image_{}.png'.format(epoch))  #元画像を保存

        pic = to_img(_img.cpu().data)
        pic = torchvision.utils.make_grid(pic,nrow = value)
        save_image(pic, './input_image_{}.png'.format(epoch))  #入力画像を保存

        pic = to_img(output.cpu().data)
        pic = torchvision.utils.make_grid(pic,nrow = value)
        save_image(pic, './image_{}.png'.format(epoch))  #生成画像   

if __name__ == '__main__':
    main()
