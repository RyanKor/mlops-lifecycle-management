# Model Analysis



## 1. Overview

### 주 학습 내용

- TF Model Analysis
- Model Debugging
- Model Robustness
  - 위의 Model Debugging & Robustness를 이해하는 것이 아래 2가지에 대한 내용을 이해하는데 중요한 역할을 담당
  - Sensitivity Analysis
  - Adversarial Attacks

- Continuous Evaluation & Monitoring
  - When we retrain a model



## 2. 모델 학습 & 배포 이후에 해야할 일은 무엇인가?

- 모델은 충분한 성능을 내고 있는지? (Is model performing well?)
- 모델의 성능이 향상될 여지가 있는지? (Is there scope for improvement?)
- 데이터가 미래에 변화될 가능성이 있는지? (Can the data change in future?)
  - 대표적으로 Fashion CV 분야
- 데이터셋이 생성된 이후로 데이터가 데이터셋을 변화시킬 여지가 있는지? (Has the data changed  since you created your training dataset?)



## 3. High Level에서 모델 성능 분석하는 방법

1. Black Box Evaluation

   - 모델의 내부 세부 정보를 몰라도 테스트 오류와 같은 정확도 및 손실과 같은 메트릭에 대해 모델을 테스트할 수 있다. (Models can be tested for metrics like accuracy and losses like test error without knowing internal details)

     - TF에서 특별히 metric을 지정하지 않고 `model.fit` 을 하면 확인할 수 있는 결과 값
     - Tensorboard를 통한 평가가 대표적인 Black Box Evaluation의 예시다.

     ![image](https://user-images.githubusercontent.com/40455392/149864503-241df4bf-ea4f-4a3a-b8d8-7bbe5a69356e.png)

2. Model Introspection

   - 모델의 보다 세밀한 평가를 위해 하드웨어 부품을 쪼개서 품질 검사하는 것처럼 파트별로 검사하는 것이 가능하다 (For finer evalutaion, models can be inspected part by part)
     - 새로운 아키텍처를 실험할 때, 데이터가 새로운 모델의 각 layer별로 어떻게 흐르는지 (flow) 관찰하는 것이 가능하다.
     - 모델을 조정하고 반복하며 (adjust & iterate) 성능 개선 및 효율성을 높이는 것이 가능하다.
   - 모델의 최종 결과 값 뿐만 아니라 각 Layer 별로 세세한 성능 평가 진행이 가능

   ![image](https://user-images.githubusercontent.com/40455392/149864697-2a824e63-cbfb-4e7c-8569-544c8975f4fc.png)

   - 왼쪽의 이미지는 CNN을 이용해 최대 활성화과 적용되어 있는 다양한 필터들이 있고, 이를 사용해 특정 클래스에 속해 있는 일련의 이미지들에서 어떤 Layer가 특정 데이터의 패턴을 학습하고 있는지 확인하는 것이 가능하다.
   - 오른쪽 강아지 이미지는 Class Activation Map 예시인데, 이미지의 어떤 파트가 우리가 요구하는 예측 값의 클래스인지를 판단하는 것이 가능하다.



## 4. Performance Metrics vs Optimization Objectives

### Performance Metrics

- 회귀, 분류, 객체 탐지 등 task에 따라 성능을 측정하는 metric은 달라질 수 있음
- 최종 목표에 따른 Task의 Type 내에서 performance metric은 다를 수 있음
- 성능 측정은 최적화 이후에 이뤄짐



### Optimization Objectives

- 머신러닝은 개발자가 정의한 문제를 목적 함수로 공식화한다. (Machine Learning formulates the problem statement into an objective function)
- 이에 따라 학습 알고리즘은 각 변수가 로컬 & 글로벌 최소값으로 수렴하는 최적의 값을 찾는다. (Learning algorithms find optimum values for each variable to converge into local/global minima)



## 5. Top level aggregate metrics vs slicing

- 대부분의 경우 metric은 전체 데이터셋을 대상으로 계산되어 평가됨 (Aggregate Metrics)
  - 이 경우, model이 특정 상황에서 성능이 떨어지는 것을 catch하는 것이 어렵다.
    - particular [Customers | Products | Stores | Days of the week | etc.]
- slicing은 데이터셋을 서브셋으로 구성해 모델 성능을 작동하는지를 다루는 것임



## 6. TFMA (Tensorflow Model Analysis)

![image](https://user-images.githubusercontent.com/40455392/149869591-c227341f-9850-499f-9039-4f46567acf10.png)

- 모델 성능의 깊이 있는 분석을 하기 위해 사용되는 오픈소스 프레임워크
- 파이프라인의 한 파트로 포함되기도하고, TFX의 컴포넌트 중 하나로 사용되기도함.

#### Architecture

![image](https://user-images.githubusercontent.com/40455392/149870139-48079794-20c1-4344-8203-a2aa01ca8e90.png)

TFMA 파이프라인의 주요 컴포넌트 4개

- Read Inputs
  - Csv or TF record 등의 데이터를 읽어들임
- Extractors
  - Read Input 값에서 받은 결과 값을 dictionary 형태로 Evaluator에 전달
  - 키포인트는 Slice를 이용해 결과 값을 추출
- Evaluators
  - Apach Beam에 의해 데이터 평가가 분산처리됨
  - Evaluator는 여럿이며, 사용자에 의한 Custom Evaluator를 생성하는 것도 가능
- Write Results
  - 결과 값을 disk에 기록

## 7. One Model vs Multiple Models over time

- Tensorboard와 TFMA는 서로 다른 stage의 개발 프로세스에서 이용된다.
  - 하나의 모델의 훈련 과정을 검사(inspect)하는데 이용된다.
  - 즉, 훈련 프로세스의 모니터링이 목적
  - metrics이 적용되는 streaming을 시각화하여 global training set의 다수의 모델들을 보여준다.
- Tensorboard는 모델 훈련 프로세스 그 자체에 쓰이며, TFMA는 훈련이 끝난 모델의 심도 있는 분석에 이용된다.
- 반대로 TFMA는 모델 훈련이 끝나면 훈련된 모델을 버전별로 관리하여 성능을 분석한다.

- 또한 TFMA는 하나의 모델에 대한 다수의 버전을 시각화해서 보여주는 특징을 갖고 있음

![image](https://user-images.githubusercontent.com/40455392/149878394-7456c724-3044-4b4c-8b93-2be9e6388729.png)

## 8. Aggregate vs sliced metrics

- 모델 성능을 표기할 때, 보통 훈련이 끝나고 모든 결과를 집속(aggregate)해서 보여주는데, 이게 모델 성능을 왜곡하는 문제를 야기한다.
- Aggregate metric은 결과적으로 specific slices에선 underperform하게 동작할 가능성이 높다.
- 때문에 Slice metric을 측정하는 방식은 조금 더 입자 레벨 (Granular Level)에서의 모델 성능을 측정하는 것이 가능함
  - 이 경우, 훈련 데이터의 mislabel이나 과적합 & 과소적합을 탐지하는 것이 가능하다.
  - 즉, 데이터의 세부적인 Feature가 더 중요한지 판단하는 것을 가늠할 수 있다.

![image](https://user-images.githubusercontent.com/40455392/149878461-088d6b91-14fe-4824-944a-e2f2eb431145.png)



## 9. Streaming vs full-pass metrics

- Tensorboard는 Mini-batch를 기준으로 Metric을 측정하는 것을 혼동할 수 있는데, 이것을 Streaming Metric이라고 하며, 관찰한 미니 배치에 기반한 근사치 값을 의미한다.
- 하지만 TFMA는 Apach Beam을 사용해 평가 데이터셋 전체를 통과하는 과정(full-pass)을 수행하는데, 이게 metric을 정확하게 계산할 수 있을 뿐만 아니라 분산 프로세스 벡엔드를 사용해 Beam Pipeline을 실행하는 것이 가능하므로, 대규모 평가 데이터 집합까지 확장하는 것이 가능하다.
- TFMA는 지정된 데이터 세트에 대해 전체 패스(full-pass)를 수행하여 TF 평가 작업자가 계산한 것과 동일한 텐서 흐름 메트릭을 더 정확하게 계산할 수 있다.
- TFMA는 식별되지 않았거나 모델에 정의되지 않은 추가 메트릭스를 계산하도록 구성될 수도 있다. 
- 또한 평가 데이터 세트가 특정 세그먼트에 대한 메트릭을 계산하기 위한 슬라이스인 경우, 각 세그먼트는 적은 수의 예제만 포함할 수 있다. 
  - 메트릭을 정확하게 계산하려면 이러한 예에 대한 결정론적 전체 통과가 중요하다.

![image](https://user-images.githubusercontent.com/40455392/149878702-6b89cfd9-2779-4d51-923a-618f7d7f9659.png)



## 10. TFMA in Practice

- TFMA가 여러 데이터 슬라이스에 대해 모델을 평가하는 방법 (Analyse impact of different slices of data over various metrics)
- TFMA가 시간 경과에 따라 메트릭을 추적하는 방법에 대해 알아봅니다. (How to track metrics over time?)
- 코드 실습 내용이므로 짧게 3줄 요약하면
  - TFX 파이프라인 중 TFMA를 구성해 예제를 작업
  - 9번에서 언급했던 전체 데이터에 대한 metric의 정밀한 계산이 가능하다.
  - 시간대에 따른 metric 변화 양상을 추적하는 것이 가능하다.

![image](https://user-images.githubusercontent.com/40455392/149935091-15f168ea-ee49-45f6-be85-b49063444cad.png)





## 11. Model Debugging

- 모델을 디버깅하는 것을 얘기하기 위해 선행되어야 하는 것은 **모델의 견고성 (Model Robustness)**이다.

- 모델의 견고성을 확인하는 것은 단순한 모델 성능 또는 일반화의 측정에서 벗어나는 단계이다. (Robustness is much more than generalization)
- 그럼 어떤 모델이 견고한 모델인가? 
  - 하나 이상의 특징이 상당히 급격하게 변경되더라도 결과가 일관되게 정확하다면, 강력한 것으로 간주
  - 문제는 데이터가 변화함에 따라 점진적이고 예측 가능한 방식으로 변화하는 모델과 갑자기 완전히 다른 결과를 만들어내는 모델 사이에는 분명한 차이가 존재하는 것
- 모델의 견고성을 측정하는 방법 -> 적어도 모델의 훈련 중에는 안된다.
  - 또한 훈련 데이터셋을 견고성 측정에 이용해서도 안된다.
- 모델 훈련할 때 가장 많이 했던 것 -> 훈련 / 검증 / 테스트 데이터셋으로 분리
  - 데이터 분할의 가장 큰 목적 : 검증 단계에서도 모형에 의해 완전히 보이지 않는 검정 분할을 사용하여 모형 견고성을 검정할 수 있습니다.
  - metrics 는 모델 유형에 따라 훈련에 사용하는 것과 동일한 유형이 될 것

![image](https://user-images.githubusercontent.com/40455392/149941118-61d13e38-c71f-4fb4-906f-94bcfd3867f6.png)

- 그래서 결국 모델 디버깅이란, 모델에서 문제를 찾고 수정하며 모델 견고성을 개선하는 데 초점을 맞춘 새로운 분야

  - 디버깅이라는 단어 자체가 Software Engineering에서 유래된 말
  - 즉, 모델 디버깅은 모델 위험 관리 / 기존 모델 점검 및 소프트웨어 테스트에서 방법을 차용한 것임

- ML Model의 가장 큰 단점 

  - 모델 훈련 과정을 투명하게 보기 어려워 BlackBox로 가려져 있다는 얘기를 많이함.
    - 모델 디버깅은 데이터의 내부 흐름을 강조하여 모델의 투명도를 개선
  - 사회적 차별을 야기
    - 우리가 작업한 모델이 특정 그룹의 사람들 (인종, 민족, 국가 등)에게만 높은 성능을 보일 수 있음
  - 모델 취약성 & 개인정보 노출
    - 개인 정보 등을 학습하게 될 때 익명화 되어 있지 않으면 문제가 될 수 있다.
  - 시간 경과에 따른 데이터 분포 변화로 모델 성능 저하 문제

  ![image](https://user-images.githubusercontent.com/40455392/149942592-1d1697d9-bddd-4a2d-8b50-3851d0fa0927.png)

- 위의 문제를 해결하는 가장 널리 사용되는 디버깅 방법 3가지

  - 벤치마킹 모델
  - 민감도 분석
  - 잔차 분석

  ![image](https://user-images.githubusercontent.com/40455392/149942792-83344c8d-f58d-4c05-94cd-eba30f665b0b.png)

## 12. Benchmark Model

- 벤치마크 모델으로 넘어가기 전에, **벤치마크 데이터셋**을 함께 알고 갈 필요가 있음

  - **벤치마크 데이터셋** : 공통된 기준으로 인공지능 정확도를 평가하고 경쟁할 수 있는 기반이며, 인공지능 발전에 핵심 역할을 담당
    - 예) ImageNet, KLUE, CIFAR-10
  - **벤치마킹 모델** : 벤치마킹 모델은 문제의 기준을 설정하기 위한 개발을 시작하기 전에 사용되는 작고 간단한 모델
    - 사용자가 생성한 모델이 벤치마크 테스트를 통과하면 벤치마크 모델이 견고한 디버깅 도구 역할

  ![image](https://user-images.githubusercontent.com/40455392/149942952-ac684014-f5f2-4ac6-8131-310e45be6608.png)

## 13. Sensitivity Analysis

- 적대적 공격에 대한 취약성을 포함하여 모델의 성능을 평가하는 중요한 방법
- 민감도 분석은 각 기능이 모형의 예측에 미치는 영향을 조사하여 모형을 이해하는 데 도움

- 다른 형상은 일정하게 유지하면서 단일 형상 값을 변경하여 실험하고 모형 결과를 관찰하고 형상 값을 변경하면 모델이 크게 달라지게 된다.
  - 그리고 이런 feature들이 예측 결과에 큰 영향을 미치게 된다.
- 민감도 분석을 수행하는 더 강력한 방법 중 하나는 텐서 흐름 팀이 만든 What-if 도구를 사용하는 것이다.
- What-if 도구를 사용하면 민감도 분석 결과를 시각화하여 성능을 이해하고 디버깅할 수 있습니다.

![image](https://user-images.githubusercontent.com/40455392/149958324-82da0f3b-6092-4f2d-b964-e6d0fcff4cbb.png)

- 민감도 분석을 위한 가장 일반적인 접근법들

  - Random Attack
    - 많은 랜덤 입력 데이터를 생성하고 모형 출력을 테스트
    - Random Attack은 모든 종류의 예상치 못한 소프트웨어와 수학 버그를 드러낼 수 있음
    - ML 시스템의 디버깅을 어디서부터 해야할지 막막하다면, Random Attack은 좋은 시작점이 될 수 있다.
  - Partial dependence plots
    - 하나 또는 두 개의 형상의 한계 효과와 이러한 형상이 모형 결과에 미치는 영향을 보여 준다.
    - Partial dependence plots은 레이블과 특정 피쳐 간의 관계가 선형 단조로운지 또는 더 복잡한지 여부를 보여 줄 수 있습니다. 
    - PDPbox와 PyCEbox를 사용하면, Partial dependence plots를 구성하는 것이 가능함.

- 모델이 민감하게 반응한다는 것은 그만큼 취약하다는 뜻

  - 따라서, 모델에 취약성이 있는지 테스트하고 분석에 따라 공격에 보다 탄력적으로 대처할 수 있도록 모델을 강화해야 할 수 있다.

  ![image](https://user-images.githubusercontent.com/40455392/149962230-e74a3c8c-d265-4dbe-8f75-8ed22772a84a.png)

- 예시 : 타조 분류 모델에 훈련한 이미지 왜곡

  ![image](https://user-images.githubusercontent.com/40455392/149962634-f305b532-06a0-4d73-a990-0b5b73bbb0da.png)
  - 위의 이미지들은 타조를 분류하는 이미지 모델을 적용해 오른편에 있는 이미지를 생성한 상황임
  - 얼핏보기에는 큰 차이가 없어보이나, 자율주행 차량 등에 이용되는 이미지 학습 등에서는 교통 표지판, 다른 차량, 사람 등을 인식하는 것이 중요한데, 위와 같은 모델을 적용해 데이터를 왜곡할 경우, 차량 충돌 등의 일상 생활 등의 재앙을 가져다 줄 수 있는 문제가 발생함
  - 스팸 및 피싱 메일 통과를 돕는 소프트웨어의 경우, 이메일 서비스에 부정적인 영향을 줄 수 있음

  ![image](https://user-images.githubusercontent.com/40455392/149964455-bc4f290c-00e2-46c3-b6fb-d98b4cb3f3ed.png)

- 마지막으로, 더 많은 목표에 큰 영향을 미치는 애플리케이션을 (Mission critical application) 머신러닝에 의존하기 때문에 보안에 미치는 영향을 고려해야 합니다. 

  - 여행가방 스캐너는 기본적으로 물건 분류기에 불과하지만 공격에 취약하다면 결과가 위험할 수 있음

  ![image](https://user-images.githubusercontent.com/40455392/149964554-c700b070-d1f9-4464-983b-1aec264a7509.png)

  - 프라이버시 및 보안을 연구하는 산업 그룹인 프라이버시 포럼의 미래는 기계 학습에 의해 가능해진 보안 및 프라이버시 해악이 대략 두 가지 범주로 분류된다고 제안
    - 정보와 행동(Information & Behavior Harm)
    - 정보 손상은 의도하지 않거나 예상치 못한 정보 유출과 관련되며, 반면에 행동 손상은 모델의 예측 또는 결과에 영향을 미치는 모델 자체의 동작 조작과 관련이 있음
    - 정보 손상을 살펴보자. 멤버십 추론 공격은 모델 출력의 샘플을 기반으로 개인의 데이터가 모델을 훈련시키는 데 사용되었는지 여부를 추론하는 것을 목표로 함
    - 겉보기에는 복잡해 보이지만, 이러한 공격은 자주 가정되는 것보다 훨씬 덜 정교함을 요구한다는 것이 밝혀졌다. 
  - 모델 반전 공격은 모델 출력을 사용하여 교육 데이터를 재생성
    - 잘 알려진 한 예로, 연구자들은 개인의 얼굴 이미지를 재구성할 수 있음

- 취약점 공격에 대한 측정 모듈 2가지

![image](https://user-images.githubusercontent.com/40455392/149964959-dd53be53-0740-4d9e-bf0c-99b972ee76d0.png)



## 14. Residual Analysis

- 대부분의 경우 예측과 실측값 사이의 거리를 측정해야 하기 때문에 회귀 모델에 사용할 수 있다.
- 그러나 많은 경우 온라인 또는 실시간 시나리오에서 어려울 수 있기 때문에 비교를 위해 Ground Truth 값이 있어야 한다.

- 일반적으로 residual 값이 임의의 분포 값을 띄고 있는 것을 원한다.
  - 잔차 사이의 상관 관계를 발견하면 일반적으로 모형을 개선할 수 있다는 신호

![image](https://user-images.githubusercontent.com/40455392/149966222-a5a42fab-fe91-49d8-bd3f-d76b1701483d.png)

- 잔차 그림을 사용하여 잔차 분포를 육안으로 검사하는 경우가 많습니다. 
- 모형이 잘 훈련되어 있고 데이터의 예측 정보를 캡처한 경우에는 잔차를 랜덤하게 분포시켜야 합니다.
- 그러나 체계적이거나 상관된 잔차가 있으면 모형이 캡처하지 못한 예측 정보가 있음을 나타냅니다. 
- 그런 다음 모형을 개선할 방법을 찾을 수 있습니다. 
- 그러면 잔차 분석을 수행할 때 무엇을 목표로 해야 할까요? 잔차는 사용 가능하지만 형상 벡터에서 제외된 다른 형상과 상관되지 않아야 한다. 
- 다른 피쳐로 잔차를 예측할 수 있는 경우 해당 피쳐를 피쳐 벡터에 포함해야 합니다. 이렇게 하려면 사용되지 않는 피쳐가 잔차와의 상관 관계를 검사해야 합니다. 또한 인접 잔차는 서로 상관되어서는 안 됩니다. 
- 즉, 자기 상관되어서는 안 됩니다. 한 잔차를 사용하여 다음 잔차를 예측할 수 있는 경우 모형에 포착되지 않은 예측 정보가 있습니다. 
- 종종(항상은 아님) 잔차 그림에서 이 값을 시각적으로 확인할 수 있으므로 순서를 정하는 것이 이를 이해하는 데 중요합니다. 



## 15. Model Remediation

