import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 예전 pandas 호환성 해결 코드
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

# 2. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "archive", "Wafer_Map_Datasets.npz")

# 3. 데이터 로드 (pickle 방식 사용)
print("📦 대용량 데이터셋 로딩 중...")
with open(file_path, 'rb') as f:
    # Python 2에서 생성된 데이터를 Python 3에서 읽기 위해 encoding='latin1' 설정
    # pickle.load를 사용하므로 이 아래에 있던 pd.read_pickle은 삭제했습니다.
    df = pickle.load(f, encoding='latin1') 

# 데이터 필터링: failureType 정보가 있는 데이터만 추출
df = df[df['failureType'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

# 4. 불균형 해결 (Under-sampling)
# 'none' 데이터가 너무 많으므로 5000개만 샘플링
df_none = df[df['failureType'].apply(lambda x: x[0][0] == 'none')].sample(5000, random_state=42)
df_defect = df[df['failureType'].apply(lambda x: x[0][0] != 'none')]
df_balanced = pd.concat([df_none, df_defect]).reset_index(drop=True)

# 5. 전처리 함수 정의
def preprocess_wafer_maps(wafer_series):
    processed = []
    for m in wafer_series:
        # 웨이퍼 맵 사이즈를 (32, 32)로 통일
        res = resize(m, (32, 32), preserve_range=True, order=0)
        processed.append(res / 2.0) # 0, 1, 2 값을 0~1 사이로 정규화
    return np.array(processed)

print("⚙️ 리사이징 및 텐서 변환 중... (잠시만 기다려주세요)")
X = torch.FloatTensor(preprocess_wafer_maps(df_balanced['waferMap'].values)).unsqueeze(1)

le = LabelEncoder()
y_labels = df_balanced['failureType'].apply(lambda x: x[0][0]).values
y = torch.LongTensor(le.fit_transform(y_labels))

# 6. 학습/검증 분리 및 저장
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 딕셔너리 형태로 저장 (라벨 인코더 정보 포함)
data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'classes': le.classes_
}

torch.save(data_to_save, "processed_wafer_data.pth")
print("✅ 전처리 완료! 'processed_wafer_data.pth'로 저장되었습니다.")