import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
lswmd_path = os.path.join(current_dir, "archive", "LSWMD.pkl")

# 1. LSWMD.pkl에서 원본 데이터 로드
print("📦 LSWMD.pkl에서 9개 다중 클래스 데이터 추출 중...")
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

try:
    with open(lswmd_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {lswmd_path}")
    sys.exit()

# 2. 복잡하게 얽혀있는 라벨 구조에서 정확한 텍스트만 추출
def extract_label(x):
    if len(x) > 0 and len(x[0]) > 0:
        return str(x[0][0])
    return None

df['failureType'] = df['failureType'].apply(extract_label)
df = df.dropna(subset=['failureType'])

# 3. 데이터 밸런싱 (none 클래스가 너무 많으므로 6000개로 축소)
print(f"📊 원본 라벨 분포:\n{df['failureType'].value_counts()}")

df_none = df[df['failureType'] == 'none']
df_defect = df[df['failureType'] != 'none']

sample_size = min(6000, len(df_none))
df_none_sampled = df_none.sample(sample_size, random_state=42)

# 결함 데이터와 축소된 정상 데이터를 병합
df_balanced = pd.concat([df_none_sampled, df_defect]).reset_index(drop=True)
print(f"\n⚖️ 밸런싱 후 9개 라벨 분포:\n{df_balanced['failureType'].value_counts()}")

# 4. 64x64 리사이징 및 텐서 변환
def preprocess_wafer_maps(wafer_series):
    processed = []
    for m in wafer_series:
        res = resize(m, (64, 64), preserve_range=True, order=0)
        processed.append(res / 2.0)
    return np.array(processed)

print("\n⚙️ 64x64 리사이징 및 텐서 변환 중... (잠시만 기다려주세요)")
X = torch.FloatTensor(preprocess_wafer_maps(df_balanced['waferMap'].values)).unsqueeze(1)

# 문자열 라벨을 0~8의 정수로 인코딩
le = LabelEncoder()
y_labels = df_balanced['failureType'].values
y = torch.LongTensor(le.fit_transform(y_labels))

print(f"🏷️ 인코딩된 클래스 목록: {le.classes_}")

# 5. 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': le.classes_ # 9개의 실제 결함 이름 저장
}

save_path = os.path.join(current_dir, "processed_multiclass_wafer_data.pth")
torch.save(data_to_save, save_path)

print(f"\n✅ 9개 클래스 전처리 완료! (저장 위치: {save_path})")