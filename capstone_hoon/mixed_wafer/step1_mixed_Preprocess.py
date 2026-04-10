import os
import sys
import pandas as pd
import numpy as np
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 경로 설정
# 현재 파이썬 스크립트가 위치한 폴더의 절대 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "archive", "Wafer_Map_Datasets.npz")

# 2. 데이터 로드
print("📦 대용량 데이터셋 로딩 중...")
data = np.load(file_path, allow_pickle=True)

# 3. DataFrame 변환
try:
    df = pd.DataFrame({
        'waferMap': list(data['arr_0']), 
        'failureType': list(data['arr_1'])
    })
    print("✅ 데이터프레임 변환 성공!")
except Exception as e:
    print(f"🚨 변환 중 에러 발생: {e}")
    sys.exit()

print("⚙️ 데이터 필터링 중...")

# 4. 라벨 데이터 1차원 평탄화
df['failureType'] = df['failureType'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

label_counts = df['failureType'].value_counts()
print(f"📊 확인된 라벨 종류 및 개수:\n{label_counts}")

# 5. 불균형 해결 (Under-sampling)
if 'none' in label_counts.index:
    none_class = 'none'
elif 'None' in label_counts.index:
    none_class = 'None'
else:
    none_class = label_counts.idxmax()
    print(f"💡 'none' 문자가 없어서, 가장 개수가 많은 '{none_class}' 라벨을 정상(none)으로 간주합니다.")

df_none = df[df['failureType'] == none_class]
sample_size = min(5000, len(df_none))
df_none = df_none.sample(sample_size, random_state=42)

df_defect = df[df['failureType'] != none_class]
df_balanced = pd.concat([df_none, df_defect]).reset_index(drop=True)

# 6. 전처리 함수 및 텐서 변환 (64x64 사이즈업 적용)
def preprocess_wafer_maps(wafer_series):
    processed = []
    for m in wafer_series:
        # 웨이퍼 맵 사이즈를 (64, 64)로 확대 통일
        res = resize(m, (64, 64), preserve_range=True, order=0)
        processed.append(res / 2.0)
    return np.array(processed)

print("⚙️ 64x64 리사이징 및 텐서 변환 중... (잠시만 기다려주세요)")
X = torch.FloatTensor(preprocess_wafer_maps(df_balanced['waferMap'].values)).unsqueeze(1)

# 라벨 인코딩
le = LabelEncoder()
y_labels = df_balanced['failureType'].values
y = torch.LongTensor(le.fit_transform(y_labels))

# 7. 학습/검증 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': le.classes_
}

# [핵심 수정 부분] 저장할 경로를 '현재 스크립트가 있는 폴더(current_dir)'로 확실하게 고정
save_path = os.path.join(current_dir, "processed_mixed_wafer_data.pth")
torch.save(data_to_save, save_path)

print(f"✅ 전처리 완료! 아래 경로에 정확히 저장되었습니다.\n저장 위치: {save_path}")