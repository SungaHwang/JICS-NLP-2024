# BERT 기반 자연어처리 모델의 미세 조정을 통한 한국어 리뷰 감성 분석: 입력 시퀀스 길이 최적화
**Fine-tuning BERT-based NLP Models for Sentiment Analysis of Korean Reviews: Optimizing the sequence length** <br>
Sunga Hwang, Seyeon Park, Beakcheol Jang<br>
Journal of Internet Computing and Services, Vol. 25, No. 4, pp. 47-56, Aug. 2024 <br>
Paper: https://www.jics.or.kr/digital-library/101121 <br>

본 연구는 BERT 기반 자연어처리 모델들을 미세 조정하여 한국어 리뷰 데이터를 대상으로 감성 분석을 수행하는 방법을 제안한다.
이 과정에서 입력 시퀀스 길이에 변화를 주어 그 성능을 비교 분석함으로써 입력 시퀀스 길이에 따른 최적의 성능을 탐구하고자 한다.
이를 위해 의류 쇼핑 플랫폼 M사에서 수집한 텍스트 리뷰 데이터를 활용한다. 웹 스크래핑을 통해 리뷰 데이터를 수집하고, 데이터 전처리
단계에서는 긍정 및 부정 만족도 점수 라벨을 재조정하여 분석의 정확성을 높였다. 구체적으로, GPT-4 API를 활용하여 리뷰
텍스트의 실제 감성을 반영한 라벨을 재설정하고, 데이터 불균형 문제를 해결하기 위해 6:4 비율로 데이터를 조정하였다. 의류
쇼핑 플랫폼에 존재하는 리뷰들을 평균적으로 약 12 토큰의 길이를 띄었으며, 이에 적합한 최적의 모델을 제공하기 위해 모델링 단
계에서는 BERT기반 사전학습 모델 5가지를 활용하여 입력 시퀀스 길이와 메모리 사용량에 집중하여 성능을 비교하였다. 실험 결과,
입력 시퀀스 길이가 64일 때 대체적으로 가장 적절한 성능 및 메모리 사용량을 나타내는 경향을 띄었다. 특히, KcELECTRA 모델이
입력 시퀀스 길이 64에서 가장 최적의 성능 및 메모리 사용량을 보였으며, 이를 통해 한국어 리뷰 데이터의 감성 분석에서 92%이상
의 정확도와 신뢰성을 달성할 수 있었다. 더 나아가, BERTopic을 활용하여 새로 입력되는 리뷰 데이터를 카테고리별로 분류하고, 최종
구축한 모델로 각 카테고리에 대한 감성 점수를 추출하는 한국어 리뷰 감성 분석 프로세스를 제공한다.

## Overall Framework
![image](https://github.com/SungaHwang/JICS-NLP-2024/assets/74399508/ba627b63-bdcf-4dad-8f0e-268eefbef2c7)

## Fine-tuning Result
![image](https://github.com/SungaHwang/JICS-NLP-2024/assets/74399508/522266e2-79c4-4c83-86ef-378826e1bc36)
![image](https://github.com/SungaHwang/JICS-NLP-2024/assets/74399508/2fe6369d-3f34-4a19-8002-566679303729)
