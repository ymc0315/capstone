# Step3_mixed_Evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch

# 1. 파일 위치를 현재 디렉토리로 고정
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")

print("📦 검증용 데이터 로딩 중...")
try:
    data = torch.load(data_path, weights_only=False)
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"✅ 데이터 로드 성공: {data_path}")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
    exit()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. [수정됨] Step 2와 완벽히 동일한 구조의 ImprovedWaferCNN 정의
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
            nn.Dropout(0.5),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = ImprovedWaferCNN(len(classes))

# 3. MemTorch 매핑 및 학습된 가중치 불러오기
print("⚡ 멤리스터 모델 매핑 및 가중치 로드 중...")
reference_memristor = memtorch.bh.memristor.VTEAM
realistic_params = {
    'r_off_variation': 0.05,
    'r_on_variation': 0.05,
}

memristive_model = memtorch.patch_model(model, 
                                        memristor_model=reference_memristor,
                                        memristor_model_params=realistic_params,
                                        tile_shape=(128, 128),
                                        adc_bitwidth=8, 
                                        dac_bitwidth=8)

model_weights_path = os.path.join(current_dir, "memristor_wafer_model.pth")

try:
    memristive_model.load_state_dict(torch.load(model_weights_path, weights_only=False))
    memristive_model.eval() 
    print(f"✅ 가중치 로드 완료: {model_weights_path}")
except FileNotFoundError:
    print(f"❌ 가중치 파일을 찾을 수 없습니다: {model_weights_path}")
    exit()

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
target_names = [str(c) for c in classes]

print(classification_report(all_labels, all_preds, target_names=target_names))

plt.figure(figsize=(12, 9))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Greens')

plt.title('Wafer Defect Classification (Memristor-based PIM Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

output_image = os.path.join(current_dir, "evaluation_result.png")
plt.savefig(output_image)
print(f"📊 결과 그래프가 '{output_image}'로 정확히 저장되었습니다.")
plt.show()