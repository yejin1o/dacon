import pandas as pd
import numpy as np
import re

# Scikit-learn 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score

# 모델링 및 최적화 라이브러리
from lightgbm import LGBMClassifier
import optuna
from scipy.sparse import hstack

print("라이브러리 임포트 완료")

# --- 1. 데이터 로드 ---
try:
    # 로컬 환경용 경로
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    print("데이터 로드 완료: train.csv, test.csv")
except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일이 현재 디렉토리에 있는지 확인해주세요.")
    exit()

# test 데이터의 컬럼명을 train과 일치시킴
test = test.rename(columns={'paragraph_text': 'full_text'})

# --- 2. 피처 엔지니어링 (메타 피처 생성) ---
def create_meta_features(df):
    """텍스트의 통계적 특징(메타 피처)을 생성하는 함수"""
    df_copy = df.copy()
    # NaN 값을 빈 문자열로 대체하여 오류 방지
    df_copy['full_text'] = df_copy['full_text'].fillna('')
    
    df_copy['text_len'] = df_copy['full_text'].apply(len)
    df_copy['word_count'] = df_copy['full_text'].apply(lambda x: len(x.split()))
    # 마침표, 물음표, 느낌표를 기준으로 문장 수 계산
    df_copy['sentence_count'] = df_copy['full_text'].apply(lambda x: len(re.split(r'[.?!]', x)) -1)
    df_copy['sentence_count'] = df_copy['sentence_count'].replace(-1, 1) # 문장이 없는 경우 1로 처리
    
    # [수정된 부분] 0으로 나누기 오류 방지
    df_copy['avg_sentence_len'] = df_copy['word_count'] / (df_copy['sentence_count'] + 1e-6)
    
    df_copy['unique_word_ratio'] = df_copy['full_text'].apply(lambda x: len(set(x.split())) / (len(x.split()) + 1e-6))
    return df_copy[['text_len', 'word_count', 'sentence_count', 'avg_sentence_len', 'unique_word_ratio']]

print("메타 피처 생성 함수 정의 완료")

# 메타 피처 생성
train_meta = create_meta_features(train)
test_meta = create_meta_features(test)

# 생성된 메타 피처 스케일링
scaler = StandardScaler()
train_meta_scaled = scaler.fit_transform(train_meta)
test_meta_scaled = scaler.transform(test_meta)

print("메타 피처 생성 및 스케일링 완료")

# --- 3. 데이터 분할 ---
X = train[['title', 'full_text']]
y = train['generated']

# 데이터를 분할할 때 인덱스를 유지하여 나중에 메타 피처와 결합
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# 인덱스를 사용하여 분할된 데이터에 맞는 메타 피처를 가져옴
X_train_meta_scaled = train_meta_scaled[X_train.index]
X_val_meta_scaled = train_meta_scaled[X_val.index]

print("데이터 분할 완료")

# --- 4. TF-IDF 벡터화 ---
get_title = FunctionTransformer(lambda x: x['title'], validate=False)
get_text = FunctionTransformer(lambda x: x['full_text'].fillna(''), validate=False) # NaN 처리 추가

# TF-IDF 벡터화기 정의 (성능을 위해 max_features 증가)
vectorizer = FeatureUnion([
    ('title', Pipeline([('selector', get_title),
                        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000))])),
    ('full_text', Pipeline([('selector', get_text), 
                            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=15000, min_df=3))])),
])

# TF-IDF 피처 변환
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(test[['title', 'full_text']])

print("TF-IDF 벡터화 완료")

# --- 5. TF-IDF 피처와 메타 피처 결합 ---
X_train_combined = hstack([X_train_vec, X_train_meta_scaled])
X_val_combined = hstack([X_val_vec, X_val_meta_scaled])
X_test_combined = hstack([X_test_vec, test_meta_scaled])

print("최종 피처 결합 완료. 최종 학습 피처 shape:", X_train_combined.shape)


# --- 6. 하이퍼파라미터 튜닝 (Optuna + LightGBM) ---
def objective(trial):
    """Optuna가 최적화할 목적 함수"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train_combined, y_train,
              eval_set=[(X_val_combined, y_val)],
              eval_metric='auc',
              callbacks=[optuna.integration.LightGBMPruningCallback(trial, "auc")]) # 조기 종료 콜백
    
    val_probs = model.predict_proba(X_val_combined)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    return auc

print("\nOptuna를 사용한 하이퍼파라미터 튜닝 시작...")
# 최적화 실행 (n_trials 횟수를 늘리면 더 좋은 성능을 기대할 수 있으나 시간이 오래 걸림)
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=30) # 테스트를 위해 30회 시도, 실제로는 50~100회 이상 권장

print(f"튜닝 완료. 최적 AUC: {study.best_value:.4f}")
print("최적 하이퍼파라미터:", study.best_trial.params)


# --- 7. 최종 모델 학습 및 추론 ---
# 최적의 파라미터로 최종 모델 정의 및 학습
best_params = study.best_trial.params
final_model = LGBMClassifier(**best_params, objective='binary', metric='auc', verbosity=-1, random_state=42, n_jobs=-1)
final_model.fit(X_train_combined, y_train)

print("\n최적 파라미터로 최종 모델 학습 완료")

# 테스트 데이터에 대한 예측 확률 계산
test_probs = final_model.predict_proba(X_test_combined)[:, 1]
print("테스트 데이터 추론 완료")


# --- 8. 제출 파일 생성 ---
sample_submission['generated'] = test_probs
# 사용자가 요청한 파일명으로 수정
submission_filename = './three_submission.csv'
sample_submission.to_csv(submission_filename, index=False)

print(f"\n제출 파일 '{submission_filename}' 생성이 완료되었습니다.")