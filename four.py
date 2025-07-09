import pandas as pd

print("라이브러리 임포트 완료")

# --- 1. 이전 제출 파일 로드 ---
try:
    # 가장 처음 만들었던 단순한 모델의 예측 결과
    base_submission = pd.read_csv('./baseline_submission.csv')
    
    # 교차 검증(CV)을 통해 만든 가장 성능이 좋은 모델의 예측 결과
    # 사용자께서 'three_submission.csv'로 저장하셨으므로 해당 파일명을 사용합니다.
    cv_submission = pd.read_csv('./three_submission.csv')
    
    # 제출 형식 샘플 파일
    sample_submission = pd.read_csv('./sample_submission.csv')

    print("이전 제출 파일 로드 완료: baseline_submission.csv, three_submission.csv")

except FileNotFoundError as e:
    print(f"오류: {e.filename} 파일이 현재 디렉토리에 있는지 확인해주세요.")
    exit()


# --- 2. 가중 평균 앙상블 ---

# 가중치 설정 (성능이 더 좋은 모델에 높은 가중치를 부여)
# 이 비율은 직접 0.6, 0.8 등으로 바꿔보며 테스트해볼 수 있습니다.
weight_cv = 0.7
weight_base = 0.3

print(f"앙상블 가중치 -> CV 모델: {weight_cv}, Baseline 모델: {weight_base}")

# 두 모델의 'generated' 예측 확률값을 가중 평균하여 새로운 예측값 생성
ensemble_preds = (cv_submission['generated'] * weight_cv) + (base_submission['generated'] * weight_base)


# --- 3. 최종 앙상블 제출 파일 생성 ---
sample_submission['generated'] = ensemble_preds

submission_filename = './four_submission.csv'
sample_submission.to_csv(submission_filename, index=False)

print(f"\n앙상블 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")
print("이 파일을 제출하여 성능 향상 여부를 확인해보세요!")