# Step2_mixed_Train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import memtorch

# 1. 전처리된 데이터 불러오기 (Step 1에서 64x64로 전처리되었다고 가정)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")

try:
    data = torch.load(file_path, weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"🚀 학습 데이터 로드 완료! (Train: {len(X_train)}개, Classes: {len(classes)}개)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    exit()

# --- [데이터 증강용 커스텀 데이터셋] ---
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

# 데이터 증강 파이프라인 (회전, 반전 추가)
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

# 2. 모델 정의 (64x64 해상도 대응 및 구조 고도화)
class ImprovedWaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedWaferCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64 -> 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = ImprovedWaferCNN(len(classes))

# 3. MemTorch 매핑 (비이상성 파라미터 추가)
reference_memristor = memtorch.bh.memristor.VTEAM
realistic_params = {
    'r_off_variation': 0.01,  # 저항 산포 5% 모사 (QAT의 핵심 1)
    'r_on_variation': 0.01,
}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params=realistic_params,
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)
print("⚡ 멤리스터 모델 매핑 성공 (비이상성 파라미터 적용)!")

# 4. 학습 설정
# weight_decay=1e-4 삭제, lr=0.001로 변경
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)
epochs = 30 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00005)
criterion = nn.CrossEntropyLoss()

# 5. 학습 루프 (Training Loop)
print(f"🏋️ 학습 시작 (Epochs: {epochs})...")

# 하드웨어 인식 학습(QAT)을 위한 가중치 노이즈 주입 함수
def inject_weight_noise(model, noise_std=0.01):
    """학습 중 가중치에 미세한 노이즈를 주입하여 멤리스터 양자화/산포 에러에 대한 내성을 기름"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

memristive_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # 순전파 전, 가중치에 미세한 노이즈 주입 (QAT 모사)
        # inject_weight_noise(memristive_model, noise_std=0.005)
        
        # 멤리스터 크로스바 기반 추론
        outputs = memristive_model(inputs)
        loss = criterion(outputs, labels)
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 에포크 종료 후 스케줄러 업데이트
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {current_lr:.6f}")

# 6. 학습된 모델 저장
torch.save(memristive_model.state_dict(), "memristor_wafer_model.pth")
print("✅ 고도화된 학습 완료 및 'memristor_wafer_model.pth' 저장됨!")