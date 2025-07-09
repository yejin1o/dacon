# =========================================================================
# 0. 기본 라이브러리 임포트
# =========================================================================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from konlpy.tag import Okt

# =========================================================================
# 1. JAVA_HOME 경로 설정 (가장 먼저 실행)
# =========================================================================
# 이전에 찾았던 32비트 자바(JDK) 설치 경로를 정확하게 입력해주세요.
# (폴더 구분 기호'\'는 파이썬에서 두 번(\\) 써줘야 합니다.)
java_path = "C:\Program Files\Java\jdk-24" # <<< 본인의 경로로 수정
os.environ['JAVA_HOME'] = java_path

# =========================================================================
# 2. 데이터 로드 및 분리
# =========================================================================
print("데이터 로드를 시작합니다...")
train_df = pd.read_csv('./traom.csv', encoding='utf-8-sig')
test_df = pd.read_csv('./test.csv', encoding='utf-8-sig')
sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')

X = train_df[['title', 'full_text']]
y = train_df['generated']

# 훈련/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# =========================================================================
# 3. [속도 개선] 훈련 데이터 샘플링
# =========================================================================
# frac=0.1은 전체 훈련 데이터의 10%만 빠르게 테스트하기 위함입니다.
# 최종 모델을 학습시킬 때는 이 부분을 주석 처리(#)하거나 frac=1.0으로 변경하세요.
SAMPLING_FRAC = 0.1 
X_train_sample = X_train.sample(frac=SAMPLING_FRAC, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

print(f"전체 훈련 데이터: {len(X_train)} 개")
print(f"샘플링된 훈련 데이터: {len(X_train_sample)} 개 (전체의 {SAMPLING_FRAC*100}%)")

# =========================================================================
# 4. 전처리 파이프라인 정의 (Okt 형태소 분석기 적용)
# =========================================================================
okt = Okt()
def okt_tokenizer(text):
    # 단어의 원형을 찾아주어 분석 성능을 높입니다.
    return okt.morphs(text, stem=True)

# 데이터프레임에서 특정 컬럼을 선택하는 기능을 가진 변수 정의
get_title = FunctionTransformer(lambda x: x['title'], validate=False)
get_text = FunctionTransformer(lambda x: x['full_text'], validate=False)

# title과 full_text를 각각 다른 TfidfVectorizer로 처리하고, 그 결과를 합칩니다.
vectorizer = FeatureUnion([
    ('title', Pipeline([('selector', get_title),
                        ('tfidf', TfidfVectorizer(
                            tokenizer=okt_tokenizer,
                            ngram_range=(1, 2),
                            max_features=3000
                        ))])),
    ('full_text', Pipeline([('selector', get_text),
                            ('tfidf', TfidfVectorizer(
                                tokenizer=okt_tokenizer,
                                ngram_range=(1, 2),
                                max_features=10000
                            ))])),
])

# =========================================================================
# 5. 피처 변환 및 모델 학습 (샘플링된 데이터 사용)
# =========================================================================
print("\n샘플링된 데이터로 피처 변환을 시작합니다. (시간이 다소 소요될 수 있습니다)")
X_train_vec = vectorizer.fit_transform(X_train_sample)
X_val_vec = vectorizer.transform(X_val)
print("피처 변환 완료!")

# XGBoost 모델 정의 (성능 개선을 위한 하이퍼파라미터 추가)
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 모델 학습 (조기 종료 기능 포함)
print("\n모델 학습을 시작합니다...")
xgb.fit(X_train_vec, y_train_sample,
        eval_set=[(X_val_vec, y_val)],
        early_stopping_rounds=50,
        verbose=100
       )

# =========================================================================
# 6. 검증 및 예측, 제출 파일 생성
# =========================================================================
print("\n검증 및 예측을 수행합니다...")
val_probs = xgb.predict_proba(X_val_vec)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {auc:.4f}")

# 테스트 데이터 전처리 및 예측
test_df = test_df.rename(columns={'paragraph_text': 'full_text'})
X_test = test_df[['title', 'full_text']]
X_test_vec = vectorizer.transform(X_test)
probs = xgb.predict_proba(X_test_vec)[:, 1]

# 제출 파일 생성
submission_filename = 'submission_final_sample.csv'
sample_submission['generated'] = probs
sample_submission.to_csv(f'./{submission_filename}', index=False)
print(f"\n제출 파일 '{submission_filename}'이 성공적으로 생성되었습니다.")