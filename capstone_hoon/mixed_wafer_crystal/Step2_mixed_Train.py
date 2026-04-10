import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import memtorch

# 1. 전처리된 "다중 분류" 데이터 불러오기 (파일명 변경)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "processed_multiclass_wafer_data.pth")

try:
    data = torch.load(file_path, weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    # 여기서 classes의 길이가 9(정상1+결함8)로 잡히면서 모델의 최종 출력 개수도 자동으로 9가 됩니다!
    print(f"🚀 학습 데이터 로드 완료! (Train: {len(X_train)}개, Classes: {len(classes)}개)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
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

# 증강(Augmentation) 파이프라인
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
            nn.MaxPool2d(2, 2)  
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), # 과적합 방지
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # num_classes가 9로 들어갑니다.
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = ImprovedWaferCNN(len(classes))

# 3. [핵심] MemTorch에 TaO2 특성 주입
reference_memristor = memtorch.bh.memristor.VTEAM
TaO2_params = {
    # TaO2 소자의 우수한 선형성과 상대적으로 낮은 산포(Variation)를 모사
    'r_on_variation': 0.03,  # ON 저항 산포 3%
    'r_off_variation': 0.05, # OFF 저항 산포 5%
}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params=TaO2_params,
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)
print("⚡ TaO2 기반 멤리스터 모델 매핑 성공!")

optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)
epochs = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00005)
criterion = nn.CrossEntropyLoss()

print(f"🏋️ 학습 시작 (Epochs: {epochs})...")
memristive_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = memristive_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {current_lr:.6f}")

# 저장 파일명도 multiclass로 변경
save_path = os.path.join(current_dir, "memristor_multiclass_model.pth")
torch.save(memristive_model.state_dict(), save_path)
print(f"✅ 학습 완료! \n저장 위치: {save_path}")