# BERT 기반 자연어처리 모델의 미세 조정을 통한 한국어 리뷰 감성 분석: 입력 시퀀스 길이 최적화
**Fine-tuning BERT-based NLP Models for Sentiment Analysis of Korean Reviews: Optimizing the sequence length** <br>
Sunga Hwang, Seyeon Park, Beakcheol Jang<br>
Journal of Internet Computing and Services, Vol. 25, No. 4, pp. 47-56, Aug. 2024 <br>
Paper: https://www.jics.or.kr/digital-library/101121 <br>

### 1 : [서론]

- 텍스트 리뷰는 소비자 경험을 전달하는 중요한 정보원으로써, 감성 분석을 통해 소비자 만족도를 파악하고, 제품 및 서비스 개선에 중요한 인사이트를 제공
- 그러나 이러한 리뷰 데이터는 종종 부정확하거나 불균형을 이루는 만족도 점수 분포와 텍스트의 다양성으로 인해 분석에 어려움을 겪음
- 본 연구는 의류 플랫폼 산업 관련 리뷰 감성 분석을 기반으로 고객에게 가장 적합한 소비를 할 수 있도록 하는 프로세스 제안

### 2 : [관련 연구]

| 항목 | 세부 설명 |
| --- | --- |
| BERT 기반 사전학습 모델 | BERT: Transformer 기반 encoder 구조로 MLM(Masked Language Model)과 NSP(Next Sentence Prediction) 방식 사용 |
|  | RoBERTa: NSP를 제거하고 대규모 배치와 학습 데이터를 활용해 성능을 향상시킨 모델 |
|  | ALBERT: 파라미터 공유와 factorized embedding 등을 통해 모델 경량화 |
|  | KLUE: 한국어 전용 평가 벤치마크 데이터셋으로 다양한 BERT 모델(KcELECTRA, KcBERT 등) 성능 테스트 |
| BERTopic | 텍스트 데이터 벡터화 후 UMAP(Unified Manifold Approximation and Projection)과 HDBSCAN(Hierarchical Density-Based Spatial Clustering of Applications with Noise)을 사용해 토픽 추출 |
| 한국어 리뷰 분석 | 영화 리뷰와 쇼핑 리뷰 분석 연구 사례 참고 |
|  | GPT-4 API를 활용해 리뷰 레이블 재조정, 데이터 불균형 문제 해결, 리뷰 카테고리화 및 감성 점수 분석 수행 |

### 3 : [방법론]

![image](https://github.com/SungaHwang/JICS-NLP-2024/assets/74399508/ba627b63-bdcf-4dad-8f0e-268eefbef2c7)

**3.1 데이터 세트 수집 및 전처리**

- 의류 쇼핑 플랫폼 M사에서 구매자의 텍스트 리뷰와 구매 만족도 점수 크롤링
    - Python 언어와 Selenium, BeautifulSoup 라이브러리 등을 활용하여 웹 스크래핑을 진행
    - 2023년 10월 5일부터 9일 기준으로 M사에 업로드된 회원 후기 중 ‘스타일 후기, ‘상품 후기’, ‘일반 후기’를 모두 수집
    - 구매 만족도 점수가 1에서 3인 것은 부정(label:0)으로, 4와 5인 것은 긍정(label:1)으로 판단
- 데이터 라벨 재조정 및 불균형 완화
    - GPT-4의 API를 사용하여 기존 구매 만족도를 재조정
    - 긍정과 부정의 비율을 약 6:4로 설정
 
<img width="539" alt="스크린샷 2025-01-27 오후 9 33 55" src="https://github.com/user-attachments/assets/b386c85b-194c-4fb3-901a-113b543938b2" />

**3.2 미세 조정: 한국어 텍스트 감성 분류**

- 해당 연구의 리뷰 길이 분포는 그림 2와 같이 평균 길이 약 12, 최대 길이 107로 확인
    - 16, 32, 64, 128으로 설정하여 결과를 비교

<img width="297" alt="스크린샷 2025-01-27 오후 9 34 11" src="https://github.com/user-attachments/assets/5f8f9307-a93b-4eef-9a36-a69e7f1e7f02" />

- 미세 조정에 사용할 모델은 허깅페이스에서 제공하는 klue/bert, klue/roberta, kykim/albert-kor, beomi/kcbert, beomi/KcELECTRA 모델 활용
    - 이때 입력 시퀀스 길이에 따른 성능과 메모리 소모량 측정에 초점을 둠

**3.3 리뷰 카테고리 분류**

- 새로 입력될 리뷰 데이터에 대해 카테고리별 감성 점수를 나타내고자 BERTopic을 사용하여 토픽 모델링 진행
    - 엘보우 방법(Elbow Method)으로 토픽의 개수 선정
- 리뷰 데이터 세트에 속해 있는 카테고리를 파악해 Okt 형태소 분석기를 사용하여분류된 카테고리에서 개수가 많이 나온 명사를 기준으로 카테고리별 키워드 사전 구축
    - 해당되는 명사가 포함된 카테고리로 해당 리뷰를 분류

**3.4 감성 점수 추출**

- 재조정한 부정, 긍정 라벨 값을 각각 0과 1로 지정한 다음, 원 핫 인코딩을 한 후 부정인 경우와 긍정인 경우의 확률값을 sotfmax 함수를 활용하여 구함
<img width="640" alt="스크린샷 2025-01-27 오후 9 34 47" src="https://github.com/user-attachments/assets/19022b31-110d-46da-9cd4-d8eec7e10b17" />

### 4 : [실험 결과]
<img width="559" alt="스크린샷 2025-01-27 오후 9 35 05" src="https://github.com/user-attachments/assets/0a9a146f-514b-44a3-8b64-fc69b1345ecd" />
<img width="564" alt="스크린샷 2025-01-27 오후 9 35 21" src="https://github.com/user-attachments/assets/c7847b77-a840-4b24-9b67-1ef1667e5e3f" />
<img width="305" alt="스크린샷 2025-01-27 오후 9 35 37" src="https://github.com/user-attachments/assets/aea86ad7-6f77-4113-a67a-e6d2e6f2a295" />

### 5 : [결론 및 향후 연구]

- 실험 결과, 입력 시퀀스 길이가 64와 128 일 때 대부분의 모델이 가장 높은 성능을 보임
    - 특히 KcELECTRA 모델이 입력 길이 64와 128 모두 정확도 92%이상으로 가장 우수한 성능을 기록
- 메모리 사용량은 입력 길이가 128일 때가 가장 많음
    - 이는 한국어 의류 쇼핑 플랫폼, 혹은 길이가 평균 약 12인 리뷰 데이터에 대해서 메모리 사용량과 정확도를 함께 고려한다면, KcELECTRA 모델의 입력 시퀀스 길이를 64로 설정하여 미세 조정 할 때 가장 적합한 결과를 제공할 것임을 시사
