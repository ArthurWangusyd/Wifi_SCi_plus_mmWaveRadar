#Use this file to test the data lodaing and your model
#If you use single_file method, use signle_import to inport data
#If you use multi_files method, use multi_import to import data
#DO NOT RUN THIS FILE BEFORE YOUR DATA IS READY TO USE!
import torch
from torch import nn
from ML_data_import import single_import,multi_import

Multi_files_save_path = r'D:\Data_process\test'
label_name = 'label.csv'
train_loader, val_loader = multi_import(Multi_files_save_path,label_name,0.2,32)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 3D convolution layer
        self.conv = nn.Conv3d(1, 32, kernel_size=(80, 3, 5), stride=(1, 1, 1), padding=(15, 1, 1))

        # 3D pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Activation function
        self.relu = nn.ReLU()

        # 使用随机数据获取Flatten之后的大小，同时应用pooling来获取正确的输出大小
        input = torch.randn(32, 1, 151, 10, 20)
        output_feature = self.conv(input)
        output_feature = self.pool(output_feature)
        n_size = output_feature.view(32, -1).size(1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(n_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Model()

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Use GPU if available
device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if torch.cuda.is_available():
    print("---------使用的是显卡cuda 0---------")
else:
    print("---------使用的是cpu---------")

# R2 score function
def r2_score(y_true, y_pred):
    var_y = torch.var(y_true)
    mse = torch.mean((y_true - y_pred) ** 2)
    return 1 - mse / var_y

# Train and validate
train_losses = []
val_losses = []
val_r2_scores = []
n_epochs = 21
print("训练开始")

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        # 确保输入数据的形状正确
        data = data.view(data.shape[0], 1, data.shape[1], data.shape[2], data.shape[3]).to(device)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, target in val_loader:
            # 同样确保输入数据的形状正确
            data = data.view(data.shape[0], 1, data.shape[1], data.shape[2], data.shape[3]).to(device)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item()
            all_targets.append(target.detach())
            all_outputs.append(output.detach())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)  # 注意这里你可以也更名为val_losses
    scheduler.step(val_loss)

    # Compute R2 score
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    val_r2_score = r2_score(all_targets, all_outputs)
    val_r2_scores.append(val_r2_score)
    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch + 1}, Training Error: {train_loss ** 0.5:.4f}, Validation Error: {val_loss ** 0.5:.4f}, Validation R2 score: {val_r2_score:.4f}')
