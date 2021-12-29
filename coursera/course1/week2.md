

# Week 2 : Select and Train a Model	



## AI System = Code (algorithm/model) + Data

- 지난 수십년 동안 지속적으로 발전시켜왔던 것은 Code 분야였다.
- 그러나 많은 프로젝트에서 데이터의 경우 자신에게 주어진 problem에 맞게 얼마든지 커스터마이징하는 것이 가능하고, 기존 모델의 성능을 끌어올릴 수 있다는 것을 확인하게 됨.

### Challenges in model development

```
1. Doing well on training set (usually measured by average training error)

2. Doing well on dev/test sets.

3. Doing well on business metrics/project goals.
```

- 모델이 발전되어가는 과정에서 1 & 2번 사항만으로도 모델 정확도를 높이는 방향으로 코드를 설계함 (즉 평균적인 오류 값을 감소시키는 방향으로)

- 그러나 1 & 2번을 만족한다고, 3번 사항이 만족되는 것이 아님.
  - 즉, 머신러닝이 business problem을 충분히 해결하지 못할 수 있음

---

## Why low average error isn't good enough

![image](https://user-images.githubusercontent.com/40455392/147060068-386efbad-39b8-49f5-9fce-d646bb38a23e.png)



- 여기서 `navigational queries` 를 검색 엔진에 사용가자 입력했을 때, 원하는 결과가 안나오면 서비스를 떠나기 쉽다.
- 테스트 셋의 정확도가 높은 게 항상 정답은 아니다 -> 수치에 속지 말자!
- 키워드 검색에서는 사용자가 입력한 단어와 가장 유사한 값을 보여줘야할텐데, 이 때 값을 평등하게 보여주는 것은 의미가 없다.
- 즉, 입력값과 가장 유사한 예측값을 모델이 측정할 수 있도록 몇 몇 값에 가중치를 더 줌으로써 원하는 결과를 찾게 해준다.



![image](https://user-images.githubusercontent.com/40455392/147061820-8ad9c17e-5b2b-4002-a256-d385dcaa6449.png)

- 이것은 위의 검색 사례와는 반대 사례인데, 모델이 값을 평등하게 측정해야하는 관점이다.
- 대출을 받을 때, 민족 / 성별 / 사는 지역 / 언어 등으로 차별을 해서는 안되며, 상품 추천 모델을 만들 때도 물건을 잘 팔리는 몇 몇 브랜드, 제품군 및 소매업체, 그리고 특정 카테고리만 사용자들에게 노출된다면, 추천에서 소외된 사람들은 그 추천 모델을 신뢰하지 않을 가능성이 높다.

---

### (잠깐 보는) 나의 사례

![image](https://user-images.githubusercontent.com/40455392/143972788-42081cc3-1ae2-4d58-bb29-6e0c82aebb0f.png)



- 다음과 같이 국문 채용 공고 카테고리의 class imbalance가 있다면, 이건 특정 클래스에 가중치를 높이게끔 학습해야하는가, 아니면 평등하게 적용해야하는가?
  - 나의 정답 : class imbalance를 해소하는 것이 맞다.
  - 이유 : 특정 키워드 또는 특정 필드의 유저에게만 채용 공고를 보이는 것이 아니기 때문이며, 사용자가 찾고자하는 글과 가장 유사한 카테고리를 찾는 것이 핵심이다.

- 해결책

```python
# 데이터 불균형을 감안하여 class weight 조절해주기
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

# '경영, 비즈니스', '개발', '디자인', '마케팅, 광고']
weights = {0:1.06730769, 1:0.5527388, 2:1.8115727, 3:1.42473746}

class_weights

# > array([1.06730769, 0.5527388 , 1.8115727 , 1.42473746])
```

---

### 학습 데이터가 굉장히 불균형할 때 (의료 데이터에서 질병 여부를 판별할 때 예시)

![image](https://user-images.githubusercontent.com/40455392/147063491-985dbf9e-247a-40f5-ae96-728b7d57a5a8.png)

- 비교적 흔한 병들의 경우 데이터 확보가 쉽기 때문에 모델을 구성하면 높은 예측 값을 보일 수 있음
- 예를 들어 Effusion은 10,000장의 이미지를 학습했고 Hernia는 100장의 이미지를 학습 후 예측값을 보여줬다면, 기본적인 입력 데이터 양의 차이로 Hernia가 약간의 성능 차이가 있다는 것을 볼 수 있다.
- 의학적인 관점에서 진단 시스템이 명백한 Hernia 사례를 무시하는 것은 허용하지는 않는다.
- 만약 환자가 나타나고 엑스레이가 그들이 Hernia 가지고 있다는 것을 분명히 보여준다면, 그 진단을 놓치는 학습 알고리즘은 문제가 될 것이다, 그러나 이것은 상대적으로 드문 상황이기 때문에, 알고리즘의 전체적인 평균 test set 정확도는 그렇게 나쁘지 않았고, 사실 그 알고리즘은 탈장의 모든 경우를 완전히 무시할 수 있었다. 
- Hernia 사례가 드물고 평균  정확도가 시험 세트의 모든 예에 동일한 가중치를 준다면 알고리즘이 이 평균 시험 세트 정확도를 손상시키지 않고 그것을 거의 무시할 수 있기 때문에 이 평균 test 정확도에 약간의 영향만 미쳤다.



## Establish Baseline

![image](https://user-images.githubusercontent.com/40455392/147065164-a581bb78-f31c-48d4-be8a-454ab5396081.png)

- HLP(Human Level Performance)를 모델의 정확도와 비교할 수 있는 잣대로 활용하면서, Car Noise부분의 정확도 개선이 필요하다는 점을 확인할 수 있음
  - 반대로 Low Bandwidth는 정확도가 낮아도, HLP와 거의 오차가 없기 때문에 baseline 구성에서 개선점이 언급될 부분이 없음

![image](https://user-images.githubusercontent.com/40455392/147065764-1789587f-89a5-48a0-97ec-0fd2ff3d852b.png)

- Baseline으로 기준을 세울 때 HLP가 비정형 데이터와 정형화 되어 있는 데이터에서 큰 차이가 있음
- 비정형 데이터는 사람이 인식하는 것이 매우 뛰어나므로 baseline으로 사용하는 것이 가능
- 그러나 비정형 데이터는 직관적으로 HLP에서 확인이 어려우므로 HLP를 사용하면 안됌

![image](https://user-images.githubusercontent.com/40455392/147066224-91d28723-79d8-45a4-9582-4a9ea1f59791.png)

- 정리 : Baseline을 세우는 것은 모델 학습으로 무엇이 가능한지를 판별할 수 있음
- 이러한 baseline 케이스들은 어떤 에러를 감소시킬 수 있고, 감소시킬 수 없는지를 볼 수 있음.

---

![image](https://user-images.githubusercontent.com/40455392/147067785-9a9cbca4-d32d-42e9-bb37-ba9a07f0facd.png)

- 가장 첫번째로 시도해야할 것은 강의, 블로그 오픈소스 등 baseline으로 사용할 수 있는 정보를 결국 최대한 탐색해야한다.
- 오픈소스로 할 수 있는 것이라면 찾아보는 것이 좋다 (아마 소스코드가 직접적으로 있으니, 바로 이용할 수 있기 때문으로 추측)

- 좋은 데이터와 좋은 알고리즘은 좋은 성능을 이끌어낸다.



![image](https://user-images.githubusercontent.com/40455392/147067102-10346dc7-4eac-46e7-9311-e9304955f1d7.png)

- Baseline이 설정되어 있다면 배포 제약 조건을 고려하는 것이 맞다.
- 그러나 baseline이 설정되어 있지 않은 상황이라면 baseline부터 설계하는 것이 더 효율





![image](https://user-images.githubusercontent.com/40455392/147067317-c6551976-7563-4ee9-8bcd-0444e8c03266.png)

- 대규모 데이터를 하루 이상 꼬박 학습시키기 전에 소규모 데이터로 overfit시켜 테스트를 진행해보는 것이 좋다.
- 이 과정으로 버그를 확인할 수 있으며, 불필요한 시간 낭비를 줄일 수 있다.
- 일반적으로 소규모 데이터에서도 에러가 발생하면, 더 큰 데이터에서는 필연적으로 에러를 피할 수 없다.
- 결국 분석한 에러를 어떻게 알고리즘의 성능 향상으로 이어줄 것인지를 고민해야한다.

---

![image](https://user-images.githubusercontent.com/40455392/147070207-373d1ecc-6ff8-4b08-b9bf-9469142af33f.png)





![image](https://user-images.githubusercontent.com/40455392/147071396-7dbd816e-07c4-4248-ad6e-e42aec321baf.png)



![image](https://user-images.githubusercontent.com/40455392/147071522-c151f9de-df59-469d-8a78-e9dad2c2a3fe.png)

- 카테고리들 중에서 작업 우선순위를 정해야한다
  - 위의 4가지 사항들로 얼마나 개선시킬 수 있는지, 개선할 카테고리 값이 얼마나 자주 등장하는지, 그리고 카테고리 정확도 개선이 어느정도로 쉬운지, 카테고리 향상이 럴마나 중요한지 등을 우선순위에서 정하는 것이 좋다.

![image](https://user-images.githubusercontent.com/40455392/147071714-f0342ba0-eef2-492c-845e-7c5c5b88cce8.png)



## Skewed Datasets







## Reference

If you wish to dive more deeply into the topics covered this week, feel free to check out these optional references. You won’t have to read these to complete this week’s practice quizzes. [Establishing a baseline](https://blog.ml.cmu.edu/2020/08/31/3-baselines/)

[Error analysis](https://techcommunity.microsoft.com/t5/azure-ai/responsible-machine-learning-with-error-analysis/ba-p/2141774)

[Experiment tracking](https://neptune.ai/blog/ml-experiment-tracking)



**Papers**

Brundage, M., Avin, S., Wang, J., Belfield, H., Krueger, G., Hadfield, G., … Anderljung, M. (n.d.). Toward trustworthy AI development: Mechanisms for supporting verifiable claims∗. Retrieved May 7, 2021http://arxiv.org/abs/2004.07213v2

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep double descent: Where bigger models and more data hurt. Retrieved from http://arxiv.org/abs/1912.02292