import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
import multiprocessing

# SRCNNモデルの定義
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# DAEモデルの定義
class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        # エンコーダー
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # デコーダー
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        )
        # アップサンプリング
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# カスタムデータセットクラスの修正
class CIFAR10SuperRes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        target = transforms.Resize((128, 128), interpolation=Image.BICUBIC)(img)
        return img, target

# サンプル画像の生成（オプション）
def save_sample_images(model, dataset, device, num_samples=5):
  model.eval()
  with torch.no_grad():
    for i in range(num_samples):
      input_img, target = dataset[i]
      input_img = input_img.unsqueeze(0).to(device)
      output = model(input_img).squeeze().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
      
      # 画像を保存
      Image.fromarray((output * 255).astype(np.uint8)).save(f'sample_output_{i}.png')
      Image.fromarray((target.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f'sample_target_{i}.png')
      Image.fromarray((input_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f'sample_input_{i}.png')


def main():
    # データセットとデータローダーの準備
    transform = transforms.Compose([transforms.ToTensor()])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 320件のサブセットを作成
    subset_indices = np.random.choice(len(full_trainset), 320, replace=False)
    trainset = Subset(full_trainset, subset_indices)
    
    trainset = CIFAR10SuperRes(trainset)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # モデル、損失関数、オプティマイザーの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEBUG: model = DAE().to(device)
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    print("### Start Epoch")
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # モデルの保存
    torch.save(model.state_dict(), 'dae_cifar10_32to128_subset.pth')

    # テストデータでの評価
    full_testset   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset_testset = np.random.choice(len(full_testset), 32, replace=False)

    testset    = CIFAR10SuperRes(subset_testset)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    # DEBUG: モデルのロード
    # model.load_state_dict(torch.load('dae_cifar10_32to128_subset.pth', map_location=device))

    # 評価
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # PSNR計算
            mse = nn.MSELoss()(outputs, targets)
            psnr = 10 * torch.log10(1 / mse)
            total_psnr += psnr.item()

    average_psnr = total_psnr / len(testloader)
    print(f'Average PSNR on test set: {average_psnr:.2f} dB')

    # サンプル画像を生成
    save_sample_images(model, testset, device)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
