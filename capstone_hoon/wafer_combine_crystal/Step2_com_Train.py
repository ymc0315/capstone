import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import memtorch

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "processed_multiclass_wafer_data.pth")
model_save_path = os.path.join(current_dir, "memristor_multiclass_model.pth")

try:
    data = torch.load(data_path, weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"🚀 학습 데이터 로드 완료! (Train: {len(X_train)}개, 클래스: {len(classes)}개)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
    exit()

class WaferDataset(Dataset):
    def __init__(self, tensors, labels, transform=None):
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        x = self.tensors[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.RandomRotation((0, 0))
    ])
])

train_dataset = WaferDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3단 CNN 구조 유지 (9개 클래스 분류)
class ImprovedWaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedWaferCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = ImprovedWaferCNN(len(classes))

print("⚡ TaO2 기반 멤리스터 크로스바 매핑 중...")
reference_memristor = memtorch.bh.memristor.VTEAM
TaO2_params = {
    'r_on_variation': 0.03,  
    'r_off_variation': 0.05, 
}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params=TaO2_params,
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 25 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)

print(f"🏋️ 9-Class 최적화 학습 시작 (Epochs: {epochs})...")
memristive_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = memristive_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {accuracy:.2f}%")

torch.save(memristive_model.state_dict(), model_save_path)
print(f"\n✅ 학습 완료! 모델이 저장되었습니다: {model_save_path}")