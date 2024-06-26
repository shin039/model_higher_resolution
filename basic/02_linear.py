# 線形関数の学習

# ------------------------------------------------------------------------------
# Import
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# 正解の関数
# ------------------------------------------------------------------------------
def answer(x, y, z):
  return x + y * 2.0 + z + 3.0


# ------------------------------------------------------------------------------
# データ生成
# ------------------------------------------------------------------------------
sample_num = 3200
batch_size = 32

input_data  = torch.randn(sample_num, 3)
output_data = answer(input_data[:, 0], input_data[:, 1], input_data[:, 2]).unsqueeze(1)  # ターゲットデータを生成

dataset     = TensorDataset(input_data, output_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ------------------------------------------------------------------------------
# ネットワークの定義
# ------------------------------------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc1(x)

model = SimpleNN()

# ------------------------------------------------------------------------------
# 損失関数と最適化
# ------------------------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ------------------------------------------------------------------------------
# トレーニング
# ------------------------------------------------------------------------------
model.train()
num_epochs = 10

for epoch in range(num_epochs):
  for idx, (inputs, targets) in enumerate(data_loader):
    output = model(inputs)
    loss = criterion(output, targets)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (idx+1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ------------------------------------------------------------------------------
# テスト
# ------------------------------------------------------------------------------
# 評価モードに切り替え
model.eval()

# 新しいデータでモデルをテスト
test_input  = torch.tensor([[1.0, 2.0, 3.0]])
test_output = model(test_input)
print(f'Test Input: {test_input.tolist()}')
print(f'Predicted Sum: {test_output.item()}')

# 実際の和を計算
actual_output = answer(test_input[0, 0], test_input[0, 1], test_input[0, 2])
print(f'Actual Sum: {actual_output}')

# ------------------------------------------------------------------------------
# テストデータで評価 (ほぼ学習が終わったデータでテストしているので、グラフに成長性は見られない。)
# ------------------------------------------------------------------------------
test_sample_num  = 100
test_input_data  = torch.randn(test_sample_num, 3)
test_output_data = answer(test_input_data[:, 0], test_input_data[:, 1], test_input_data[:, 2]).unsqueeze(1)

test_dataset = TensorDataset(test_input_data, test_output_data)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_losses = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        test_losses.append(test_loss.item())

# ------------------------------------------------------------------------------
# 損失をプロット
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Test Sample')
plt.ylabel('Loss')
plt.title('Test Loss for 100 Samples')
plt.legend()
plt.show()
