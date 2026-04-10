# Step3_Evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch

# 1. 데이터 및 모델 로드
print("📦 검증용 데이터 로딩 중...")
data = torch.load("processed_mixed_wafer_data", weights_only=False)
X_test, y_test, classes = data['X_test'], data['y_test'], data['classes']

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 모델 구조 정의 (Step 2와 동일)
class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.fc = nn.Linear(16 * 15 * 15, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = WaferCNN(len(classes))

# 3. MemTorch 매핑 및 학습된 가중치 불러오기
reference_memristor = memtorch.bh.memristor.VTEAM
memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params={},
                                        tile_shape=(128, 128))

# 학습된 멤리스터 가중치(pth) 로드
memristive_model.load_state_dict(torch.load("memristor_wafer_model.pth", weights_only=False))
memristive_model.eval() # 평가 모드 전환

# 4. 예측 수행
print("🧐 멤리스터 모델 성능 테스트 시작...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = memristive_model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. 결과 분석 및 시각화
print("\n📝 [분류 보고서]")
print(classification_report(all_labels, all_preds, target_names=classes))

# 혼동 행렬(Confusion Matrix) 그리기
plt.figure(figsize=(12, 9))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title('Wafer Defect Classification (Memristor-based PIM Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()