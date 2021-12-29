# Week 1 : Overview of the ML LifeCycle and Deployment

### Overview

- 머신러닝 프로세스의 전체 과정을 다루는 것이 MLops
- 아래 이미지가 전체 머신러닝 생애 주기(Life Cylce)를 도식화한 내용임

- 내가 경험해 부분은 아래 프로세스에서 Scoping / Modeling / 그리고 배포에서 Deploy in production 부분만 해당됨
- 실제로 데이터를 정의하고, 설계하는 부분과 라벨링해주는 작업, 그리고 배포된 모델을 모니터링하고 시스템의 관점에서 유지할 수 있는 경험을 해보지는 못한 상황임
  - 내가 이 수업을 수강해야하는 이유이기도하다.
- 사실상 이 한 장이 수업에 대한 모든 내용을 포함하고 있다고 봐도 무방할 것이다.
- 모델 중심 AI 개발보다 데이터 중심의 AI 개발이 훨씬 효율적으로 모델 성능을 높일 수 있기 때문에 MLOps에서 체계적으로 데이터의 질을 개선하는 툴과 프로세스를 설계하는 것이 좋다.
- MLOps (AI 시스템을 관리하는 과정)에서 가장 중요한 일은 **데이터의 일관성을 유지할 수 있도록 체계화하는 일이다.**

<img width="714" alt="스크린샷 2021-12-20 오전 11 14 01" src="https://user-images.githubusercontent.com/40455392/146702266-b522de86-cd2a-46c2-8983-13fab31d3d42.png">



### I've learned to train a ML model now, what do I do?

- 결국 Production으로 사용해 보는 게 아니라면, 아무리 잘 만든 모델이라한들 이 모델로 만들 수 있는 가치의 총량은 한계가 존재함.
- 사업하는 사람이 아니더라도, 본인이 머신러닝 커리어를 생각하고 가는 상황이라면 배포 경험에 대해 인터뷰에서 물어볼 수 있는 상황은 충분히 존재함.
- 아래는 간단한 배포 상황에 대한 예시임

![image](https://user-images.githubusercontent.com/40455392/146702992-9d7a20a8-d404-49ad-8de4-f1a17cc05413.png)

- 배포하는 환경에서 설정해 볼 수 있는 practical한 지식들을 공유하게 될 것이고, 환경 설정 Know-how 등을 함께 배우게 될 것이다.
- 그리고 production environment는 우리가 작성한 머신러닝 코드 이상의 것을 요구하는데, 대표적인 상황이 아래의 이미지다.
- 머신러닝 코드는 전체 ML Project의 Code에선 5 ~ 10% 밖에 차지하지 않는다.

![image](https://user-images.githubusercontent.com/40455392/146703484-e0073224-bcae-4ec6-bc01-503a1a6fe550.png)



- [D. Sculley et.al, NIPS 2015: Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

- 위 논문에서 언급한 내용에서는, 환경 설정 (Configuration), 데이터 수집 및 분석 (Data Collection & Vertification), 자원 관리 (Machine Resource Management), 모델 서빙 환경 (Serving Infrastructure) 등의 컴포넌트들이 ML System을 구성하는 중요한 요소들이며 현실에서의 ML System에서 머신러닝 코드가 전부라는 생각을 갖는 게 굉장히 위험하다는 말을 서두에 꺼낸다.
- 현실적으로 ML Code를 만들고 system을 배포하는 것보다 유지 & 관리하는 이슈는 더 어렵고 비용도 많이 소요된다.



### Case Study : Speech Recognition

---

- Speech Recognition의 경우, 이미 스마트폰, 태플릿 PC, 데스크탑, 노트북 등에 마이크가 내장 되어 있어 정확하게 예측할 수 있는 머신러닝 분야 중 하나이기 때문에 케이스 스터디로 적합

![image](https://user-images.githubusercontent.com/40455392/146705868-c2c066af-f2db-43cb-966a-ea8e852d1027.png)

- Scoping 단계에서 프로젝트를 정의하고, 프로젝트 정의의 예시로
  - 목소리 연구를 위한 음성 인식 업무를 수행하는 프로젝트라 정의내리는 것이 가능함
  - 그 후, Key Metrics는 어떤 것을 지정할 것인가?
    - Accuracy, Latency, Throughput 등을 지표로 활용할 수 있음
  - 프로젝트 수행을 위한 가용 자원과 timeline은 어떻게 잡을 것인가?



![image](https://user-images.githubusercontent.com/40455392/146706312-099efb71-48e3-4f87-acc6-197131dd0606.png)

- Scoping을 정의했다면, Data Stage의 단계에서 데이터를 정의해야함.
  - 베이스라인을 구성할 때
    - 데이터의 라벨링이 일정하게 진행되었는지?
    - 각 음성 클립의 전후로 목소리가 들어가지 않는 상황 (Silence)이 얼마나 길게 존재하는지?
    - 각 음성 클립의 볼륨(volume, 소리 크기)를 어떻게 정규화(normalization)할 것인지?



![image](https://user-images.githubusercontent.com/40455392/146706630-233a6241-4b3b-4294-a8ee-4f510386578a.png)

- 데이터 정의가 끝나면 본격적인 모델링 단계로
  - Code(Algorithm / Model)
  - Hyperparameters
  - Data
- 를 사용해 모델을 구성한다.



![image](https://user-images.githubusercontent.com/40455392/146707008-c0420e6a-1cdd-4261-bbf4-79296af90b7d.png)

- 배포 이후에는 모바일 등 음성을 인식하는 하드웨어에서 VAD (Voice Activity Detection) 모듈을 사용하게 된다.
- 오디오 클립을 배포한 모델의 예측 값 서버로 보내서 transcript & search result를 반환한다.



## What makes deployment hard?

머신러닝의 production 으로서 활용하기 위한 deployment에서 일반적으로 2가지 어려움이 존재하는데

- Machine Learning itself or Statistical issue (통계 이슈)
- Software Engineering Issue

라고 할 수 있음

배포한 모델 성능이 시간이 지날 수록 떨어질 수 있는데 음성 인식에서는 보통

- Gradual Change
- Sudden Shock

으로 얘기할 수 있고, 사람들이 사용하는 언어는 신조어가 조금씩 새롭게 생성되기 때문에 기존에 훈련시킨 데이터만으로는 새롭게 생성되는 데이터를 대응하기 어려운 점이 존재함.

따라서, ML System을 배포하면 데이터가 변화하는 것을 감지하고 관리할 수 있는 모니터링 시스템이 함께 포함되어야 할 것임.

이에 대해 함께 언급되는 개념은

1. Concept Drift : **시간이 지남에 따라 모델링 대상의 통계적 특성이 바뀌는 현상**으로 x에 따른 y값이 바뀌는 것을 의미함.
2. Data Drift : 모델의 성능 저하를 야기 하는 입력 데이터의 변경을 의미한다.



## Common Deployment Cases

1. new product / capability
2. Automate / assist with manual task
3. Replace previous ML System

Key Ideas :

- Gradual ramp up with monitoring (모니터링을 점진적으로 증가시키는 방법)
- Rollback

### 머신 러닝 배포에 사용할 수 있는 방법 

1. **Canary Deployment (카나리 새를 떠올리자.)**

![image](https://user-images.githubusercontent.com/40455392/146722268-ab050cdb-cb34-4f64-a384-5b5fd8f1b9f0.png)

- 카나리 배포는 옛날 광부들이 광산에 유독가스가 있는지 확인하기 위해 가스에 민감한 카나리아를 광산에 가지고 들어간 거에서 아이디어를 얻은 배포 방법입니다. 

- 신규 버전을 배포할 때 한꺼번에 앱의 전체를 교체하는게 아니라 기존 버전을 유지한 채로 일부 버전만 신규 버전으로 올려서 신규 버전에 버그나 이상은 없는지를 사용자 반응은 어떤지 확인하는데 유용하게 사용하는 방법입니다. 

- 쿠버네티스의 기본 디플로이먼트로는 디플로이먼트에 속한 포드들을 하나씩이던 한꺼번에든 모두 교체하는 방식이기 때문에 이런 카나리 배포를 하기에는 어려움이 있습니다. 

- 하지만 라벨을 이용하면 쿠버네티스에서도 카나리 배포를 할 수 있습니다.
  - 즉, GKE를 사용해 배포하는 것이 가능하다.

2. **Blue-Green Deployment**

![image](https://user-images.githubusercontent.com/40455392/146722686-729bd4fa-59d1-472b-bfdd-d5558ac104e0.png)

- 기존에 띄워져 있는 포드 개수와 동일한 개수 만큼의 신규포드를 모두 띄운 다음에 신규 포드가 이상없이 정상적으로 떴는지 확인한 다음에 들어오는 트래픽을 한번에 신규포드쪽으로 옮기는 방법
- 롤백하는 과정이 쉽다 (롤백 : 신규 버전에 문제가 있을 때 과거 버전으로 돌아가는 것이 쉽다.)
- 다만, 서버를 2배로 띄워야 되기 때문에 비용이 더 많이 든다는 단점이 있다.

3. **Shadow mode deployment**

- ML 알고리즘을 이용하여 어떠한 output을 뽑아내고는 있지만, 특수한 이유로 그 output을 어떠한 의사결정에도 사용하지 않는 상황이라고 생각하면 될듯함
- 다만 추후 해당 값을 모니터링함으로써 에러 분석 시 사용하거나 해당 output이 의사 결정에 사용시 도움이 된다는 판단이 내려지면 해당 output을 이용하면될듯



4. Rolling deplotyment

## Monitoring

- 배포한 머신러닝 성능 측정을 위한 지표로 삼을 수 있는 것들에 대한 예시

![image](https://user-images.githubusercontent.com/40455392/146726278-c216a96e-2f9c-4e35-b134-52a6b26bdb1b.png)

- 결국 모델의 성능을 production side에서 일정 수준 이상 유지하기 위해서는 발생하는 오류를 분석하고 모델의 재배포가 필요함
- 배포와 모니터링도 모델의 주기를 함께 따라갈 수 밖에 없음

![image](https://user-images.githubusercontent.com/40455392/146726515-c1ddc7fe-87eb-4b9c-a511-ed2409ecbb69.png)

## Review

- 가까이서 보면 Data Scientist, Data Engineer, Machine Learning Engineer가 서로 다른 영역의 일을 맡아서 하는 것처럼 보이지만, 멀리서보면 결국 이 포지션들 모두 ML에 기여하는 사람들이므로 ML Lifecycle을 관리하고 꾸준히 모니터링해줘야한다.

![image](https://user-images.githubusercontent.com/40455392/146723548-87bcdf88-bb74-4267-88ff-cfeacd6a01ac.png)

- AI를 이용한 자동화 수준은 결국 사람이 선택하는 것이다.
- 하나의 프로세스를 전부 AI가 할 수 있게 자동화하거나 AI assitance & 부분 자동화는 사람이 원하는 수준에 맞게 반복해 나가면서 조정하는 과정임

![image](https://user-images.githubusercontent.com/40455392/146724864-72495bee-c4d3-4ee1-929f-85e886f50ef6.png)

## Reference

- [Concept and Data Drift](https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb)
- [A Chat with Andrew on MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo)
- [Monitoring ML Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
- Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX). [http://arxiv.org/abs/2010.02013 ](http://arxiv.org/abs/2010.02013)
- Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies. http://arxiv.org/abs/2011.09926
- [Data Collection and Quality Challenges in Deep Learning: A Data-Centric AI Perspective](https://arxiv.org/pdf/2112.06409.pdf)

- [D. Sculley et.al, NIPS 2015: Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

