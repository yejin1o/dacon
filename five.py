import pandas as pd
import numpy as np
import os

print("라이브러리 임포트 완료")

# --- 1. 이전 제출 파일 로드 ---
try:
    base_submission = pd.read_csv('./baseline_submission.csv')
    # 사용자께서 'three_submission.csv'로 저장하셨으므로 해당 파일명을 사용합니다.
    cv_submission = pd.read_csv('./three_submission.csv')
    sample_submission = pd.read_csv('./sample_submission.csv')
    print("이전 제출 파일 로드 완료: baseline_submission.csv, three_submission.csv")

except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일이 현재 디렉토리에 있는지 확인해주세요.")
    exit()

# --- 2. 여러 가중치를 시도하며 앙상블 파일 생성 (Grid Search) ---

# 시도해볼 CV 모델의 가중치 리스트
# 0.5부터 1.0 전까지 0.05 간격으로 테스트 (0.5, 0.55, 0.6, ... 0.95)
weights_to_try = np.arange(0.5, 1.0, 0.05)

print("\n--- 여러 가중치로 앙상블 파일 생성을 시작합니다 ---")

for weight_cv in weights_to_try:
    weight_base = 1.0 - weight_cv
    
    # 소수점 두 자리로 깔끔하게 표시
    w_cv_str = f"{weight_cv:.2f}"
    w_base_str = f"{weight_base:.2f}"
    
    print(f"가중치 테스트 중 -> CV 모델: {w_cv_str}, Baseline 모델: {w_base_str}")

    # 두 모델의 예측 확률값을 가중 평균
    ensemble_preds = (cv_submission['generated'] * weight_cv) + (base_submission['generated'] * weight_base)
    
    # 제출 파일 생성
    # 파일 이름에 가중치를 포함하여 어떤 설정의 결과인지 알 수 있게 함
    submission_filename = f'./ensemble_w_cv_{w_cv_str}.csv'
    
    temp_submission = sample_submission.copy()
    temp_submission['generated'] = ensemble_preds
    temp_submission.to_csv(submission_filename, index=False)

print("\n--- 모든 앙상블 파일 생성이 완료되었습니다! ---")