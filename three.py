import pandas as pd
import numpy as np
import re
import gc

# Scikit-learn 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score

# 모델링 라이브러리
from lightgbm import LGBMClassifier
from scipy.sparse import hstack

print("라이브러리 임포트 완료")

# --- 1. 데이터 로드 ---
try:
    train_df = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('./test.csv', encoding='utf-8-sig')
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    print("데이터 로드 완료")
except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일이 현재 디렉토리에 있는지 확인해주세요.")
    exit()

test_df = test_df.rename(columns={'paragraph_text': 'full_text'})

# --- 2. 피처 엔지니어링 함수 정의 ---
def create_meta_features(df):
    df_copy = df.copy()
    df_copy['full_text'] = df_copy['full_text'].fillna('')
    df_copy['text_len'] = df_copy['full_text'].apply(len)
    df_copy['word_count'] = df_copy['full_text'].apply(lambda x: len(x.split()))
    df_copy['sentence_count'] = df_copy['full_text'].apply(lambda x: len(re.split(r'[.?!]', x)) - 1)
    df_copy['sentence_count'] = df_copy['sentence_count'].replace(-1, 1)
    df_copy['avg_sentence_len'] = df_copy['word_count'] / (df_copy['sentence_count'] + 1e-6)
    df_copy['unique_word_ratio'] = df_copy['full_text'].apply(lambda x: len(set(x.split())) / (len(x.split()) + 1e-6))
    return df_copy[['text_len', 'word_count', 'sentence_count', 'avg_sentence_len', 'unique_word_ratio']]

print("메타 피처 생성 함수 정의 완료")

# --- 3. 교차 검증(Cross-Validation) 준비 ---
# 학습 데이터와 타겟, 테스트 데이터 정의
X = train_df[['title', 'full_text']]
y = train_df['generated']
X_test = test_df[['title', 'full_text']]

# Stratified K-Fold 설정 (5-Fold)
N_SPLITS = 5
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 각 Fold의 검증 점수와 테스트 예측 결과를 저장할 변수 초기화
oof_scores = []
test_predictions = np.zeros(len(test_df))

# --- 이전에 Optuna로 찾은 최적 파라미터를 여기에 입력해주세요 ---
# 이 파라미터는 예시이며, 직접 찾으신 `최적 AUC: 0.9319`를 기록했던 파라미터로 대체하는 것이 가장 좋습니다.
best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 1878,
    'learning_rate': 0.02534578393674641,
    'num_leaves': 42,
    'max_depth': 11,
    'subsample': 0.8299995878675975,
    'colsample_bytree': 0.6119338573294863,
    'reg_alpha': 1.458996637040331,
    'reg_lambda': 9.204843139384666
}

# --- 4. 교차 검증 루프 실행 ---
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"--- Fold {fold + 1}/{N_SPLITS} 시작 ---")

    # 1. Fold에 맞게 학습/검증 데이터 분할
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- 데이터 누수 방지를 위해 전처리를 루프 안에서 수행 ---
    
    # 2. 메타 피처 생성 및 스케일링 (학습 데이터 기준으로 fit)
    scaler = StandardScaler()
    X_train_meta = create_meta_features(X_train)
    X_val_meta = create_meta_features(X_val)
    X_test_meta = create_meta_features(X_test)

    X_train_meta_scaled = scaler.fit_transform(X_train_meta)
    X_val_meta_scaled = scaler.transform(X_val_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    
    # 3. TF-IDF 벡터화 (학습 데이터 기준으로 fit)
    get_title = FunctionTransformer(lambda x: x['title'], validate=False)
    get_text = FunctionTransformer(lambda x: x['full_text'].fillna(''), validate=False)
    
    vectorizer = FeatureUnion([
        ('title', Pipeline([('selector', get_title), ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000))])),
        ('full_text', Pipeline([('selector', get_text), ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=15000, min_df=3))])),
    ])
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. 최종 피처 결합
    X_train_combined = hstack([X_train_vec, X_train_meta_scaled])
    X_val_combined = hstack([X_val_vec, X_val_meta_scaled])
    X_test_combined = hstack([X_test_vec, X_test_meta_scaled])

    # 5. 모델 학습 및 예측
    model = LGBMClassifier(**best_params)
    model.fit(X_train_combined, y_train)
    
    val_preds = model.predict_proba(X_val_combined)[:, 1]
    fold_auc = roc_auc_score(y_val, val_preds)
    oof_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

    # 테스트 데이터 예측 결과를 평균내기 위해 누적
    test_predictions += model.predict_proba(X_test_combined)[:, 1] / N_SPLITS
    
    # 메모리 관리
    del X_train, X_val, y_train, y_val, X_train_combined, X_val_combined
    gc.collect()

print("\n--- 교차 검증 완료 ---")
print(f"평균 검증 AUC: {np.mean(oof_scores):.4f} (표준편차: {np.std(oof_scores):.4f})")

# --- 5. 최종 제출 파일 생성 ---
sample_submission['generated'] = test_predictions
submission_filename = './three_submission.csv'
sample_submission.to_csv(submission_filename, index=False)

print(f"\n최종 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")