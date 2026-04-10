# Step2_Train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import memtorch

# 1. 전처리된 데이터 불러오기
try:
    data = torch.load("processed_mixed_wafer_data.pth", weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"🚀 학습 데이터 로드 완료! (Train: {len(X_train)}개, Classes: {len(classes)}개)")
except FileNotFoundError:
    print("❌ 'processed_mixed_wafer_data.pth' 파일을 찾을 수 없습니다. Step1을 먼저 실행해 주세요.")
    exit()

# 2. 모델 정의
class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2) # 32x32 -> 15x15
        self.fc = nn.Linear(16 * 15 * 15, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = WaferCNN(len(classes))

# 3. MemTorch 매핑 (멤리스터 크로스바 구조로 변환)
reference_memristor = memtorch.bh.memristor.VTEAM
memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params={},
                                        tile_shape=(128, 128), 
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)

print("⚡ 멤리스터 모델 매핑 성공!")

# 4. 학습 설정
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(memristive_model.parameters(), lr=0.001)

# 5. 학습 루프 (Training Loop)
epochs = 5  # 초기 테스트용 5회 (정식 실험 시 20회 이상 추천)
print(f"🏋️ 학습 시작...")

memristive_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # 멤리스터 크로스바 기반 추론
        outputs = memristive_model(inputs)
        loss = criterion(outputs, labels)
        
        # 역전파 및 가중치(컨덕턴스) 업데이트
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {running_loss/len(train_loader):.4f}")

# 6. 학습된 모델 저장
torch.save(memristive_model.state_dict(), "memristor_wafer_model.pth")
print("✅ 학습 완료 및 'memristor_wafer_model.pth' 저장됨!")

