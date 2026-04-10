import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import memtorch

# 1. 파일 위치 고정 (multiclass 파일 로드)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "processed_multiclass_wafer_data.pth")

print("📦 검증용 데이터 로딩 중...")
try:
    data = torch.load(data_path, weights_only=False)
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    print(f"✅ 데이터 로드 성공! ({len(classes)}개의 클래스 감지됨)")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
    exit()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Step 2와 동일한 TaO2 파라미터 적용
print("⚡ TaO2 멤리스터 모델 매핑 및 가중치 로드 중...")
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

# 저장된 multiclass 가중치 불러오기
model_weights_path = os.path.join(current_dir, "memristor_multiclass_model.pth")

try:
    memristive_model.load_state_dict(torch.load(model_weights_path, weights_only=False))
    memristive_model.eval() 
    print(f"✅ 가중치 로드 완료: {model_weights_path}")
except FileNotFoundError:
    print(f"❌ 가중치 파일을 찾을 수 없습니다: {model_weights_path}")
    exit()

print("🧐 TaO2 기반 멤리스터 모델 다중 분류 성능 테스트 시작...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = memristive_model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------------------------------------------
# 5. 결과 분석 및 시각화 (결함별 상세 퍼센티지 출력 포함)
# -------------------------------------------------------------
print("\n📝 [TaO2 기반 PIM 가속기 분류 보고서 (전체 요약)]")
target_names = [str(c) for c in classes]

print(classification_report(all_labels, all_preds, target_names=target_names))

print("\n🔍 [각 결함 유형별 상세 정확도 (Accuracy per Class)]")
cm = confusion_matrix(all_labels, all_preds)

for i, defect_name in enumerate(target_names):
    total_samples = cm[i].sum()
    if total_samples > 0:
        correct_preds = cm[i][i]
        accuracy = (correct_preds / total_samples) * 100
        print(f" 🔹 {defect_name:<10} : {accuracy:>5.1f}% ({correct_preds}/{total_samples}개)")
    else:
        print(f" 🔹 {defect_name:<10} : 테스트 데이터 없음")
print("-" * 50)

# 시각화 (혼동 행렬)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')

plt.title('Wafer Defect Classification (TaO2 Memristor-based PIM)')
plt.xlabel('Predicted Defect Type')
plt.ylabel('True Defect Type')
plt.xticks(rotation=45)

output_image = os.path.join(current_dir, "multiclass_evaluation_result.png")
plt.savefig(output_image, bbox_inches='tight')
print(f"📊 결과 그래프가 '{output_image}'로 저장되었습니다.")
plt.show()