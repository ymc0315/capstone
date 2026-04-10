import os
import sys
import pandas as pd
import numpy as np
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 1. 경로 설정 (현재 디렉토리 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "archive", "Wafer_Map_Datasets.npz")

# 2. 데이터 로드
print("📦 대용량 데이터셋 로딩 중...")
try:
    data = np.load(file_path, allow_pickle=True)
    df = pd.DataFrame({
        'waferMap': list(data['arr_0']), 
        'failureType': list(data['arr_1'])
    })
    print("✅ 데이터프레임 변환 성공!")
except Exception as e:
    print(f"🚨 변환 중 에러 발생: {e}")
    sys.exit()

print("⚙️ 데이터 필터링 및 다중 분류 라벨 정리 중...")

# 3. 라벨 전처리 및 결측치 제거
# numpy array나 list 형태로 되어 있는 라벨을 문자열로 추출
def extract_label(label):
    if isinstance(label, (list, np.ndarray)) and len(label) > 0:
        return str(label[0]).strip()
    elif isinstance(label, str):
        return label.strip()
    return None

df['failureType'] = df['failureType'].apply(extract_label)

# 라벨이 아예 없거나 길이를 알 수 없는 쓰레기 데이터 제거
df = df.dropna(subset=['failureType'])
df = df[df['failureType'] != '']
df = df[df['failureType'] != '[]']

# 4. 다중 분류를 위한 데이터 밸런싱 (Under-sampling 'none')
# 실제 WM-811K 데이터셋은 9개의 클래스로 이루어져 있습니다.
print(f"📊 원본 라벨 분포:\n{df['failureType'].value_counts()}")

# 'none'(정상) 데이터가 압도적으로 많으므로, 이를 결함 데이터 총합과 비슷한 수준으로 맞춤
none_class = 'none' if 'none' in df['failureType'].values else 'None'

df_none = df[df['failureType'] == none_class]
df_defects = df[df['failureType'] != none_class]

# 정상 데이터 수를 6000개로 제한 (학습 속도와 밸런스를 위함)
sample_size = min(6000, len(df_none))
df_none_sampled = df_none.sample(sample_size, random_state=42)

# 샘플링된 정상 데이터와 모든 결함 데이터를 합침
df_balanced = pd.concat([df_none_sampled, df_defects]).reset_index(drop=True)
print(f"\n⚖️ 밸런싱 후 라벨 분포:\n{df_balanced['failureType'].value_counts()}")

# 5. 전처리 함수 및 텐서 변환 (64x64 사이즈업 적용)
def preprocess_wafer_maps(wafer_series):
    processed = []
    for m in wafer_series:
        # 웨이퍼 맵 사이즈를 (64, 64)로 확대 통일
        res = resize(m, (64, 64), preserve_range=True, order=0)
        processed.append(res / 2.0)
    return np.array(processed)

print("\n⚙️ 64x64 리사이징 및 텐서 변환 중... (약 1~2분 소요될 수 있습니다)")
X = torch.FloatTensor(preprocess_wafer_maps(df_balanced['waferMap'].values)).unsqueeze(1)

# 6. 다중 라벨 인코딩 (0 ~ 8 사이의 정수로 변환)
le = LabelEncoder()
y_labels = df_balanced['failureType'].values
y = torch.LongTensor(le.fit_transform(y_labels))

print(f"🏷️ 인코딩된 클래스 목록: {le.classes_}")

# 7. 학습/검증 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': le.classes_ # 9개의 클래스 이름 저장
}

# 기존 이진 분류 파일과 겹치지 않게 이름을 multiclass로 변경
save_path = os.path.join(current_dir, "processed_multiclass_wafer_data.pth")
torch.save(data_to_save, save_path)

print(f"\n✅ 다중 분류 전처리 완료! 아래 경로에 저장되었습니다.\n저장 위치: {save_path}")