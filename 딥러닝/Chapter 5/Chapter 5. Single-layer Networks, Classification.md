분류 문제의 목표는, $\mathbf{x}\in\mathbb{R}^{D}$ 의 input vector를 입력 받아, $K$ 개의 discrete classes $C_{k}, \ k=1,\dots,K$에 입력받은 input vector를 할당해주는 것. 
input vector는 **decision regions**으로 나뉘게 되고, decision regions의 경계를 **decision boundaries**, 혹은 **decision surfaces**라고 부른다.

# 5.1 Discriminant Functions
우리는 linear discrimination으로 관심 범위를 제한. 따라서 deicison boundary가 hyperplane.
input vector : $D\text{ - dimensional}$ -> decision surfaces : $(D-1)\text{ - dimensional}$

$$\mathbf{x} \mapsto \mathcal{C}_k$$

## 5.1.1 Two Classes
$$y(\mathbf{x})=\mathbf{w}^\mathrm{T}\mathbf{x}+w_0$$
여기서, $\mathbf{w} : \text{weight vector}, w_{0} : \text{bias}$
$y(\mathbf{x}) > 0$ 이면 $\mathcal{C}_1$에, 아니라면 $\mathcal{C}_2$에 분류해줌.
따라서 decision boundary는 $y(\mathbf{x})=0$ 임.

![[5.1.png]]
위 그림에서 input vector가 2차원이라, decision boundary는 1차원이 된 것을 볼 수 있음.

### 몇 가지 properties
- '$\mathbf{w}$ 가 decision boundary의 orientation을 결정해준다는 것'을 아래 과정을 통해 이해할 수 있음.
	임의의 두 벡터 $\mathbf{x}_A$ 와 $\mathbf{x}_B$ 가 둘 다 decision boundary 위에 존재한다고 해보자,
	그렇다면, $y(\mathbf{x}_A)=y(\mathbf{x}_B)=0$ 임.
	$y(\mathbf{x}_A)-y(\mathbf{x}_B)=\mathbf{w}^{T}(\mathbf{x}_{A}-\mathbf{x}_{B})=0$ 이고,
	$\mathbf{x}_{A}-\mathbf{x}_{B}$ 는 decision boundary 위의 임의의 벡터임. 
	따라서 $\mathbf{w}$는 decision boundary 위의 모든 벡터와 orthogonal하고, 이를 통해 $\mathbf{w}$가 decision boundary의 orientation을 지정해준다는 것을 이해할 수 있음.

- origin으로 부터 decision boundary까지의 거리를 아래와 같이 나타낼 수 있음.
	임의의 벡터 $\mathbf{x}$가 decision boundary 위에 존재한다고 해보자,
	그렇다면, $y(\mathbf{x})=\mathbf{w}^{T}\mathbf{x}+w_{0}=0$ 임. 
	그리고 우리가, origin부터 decision boundary까지의 거리를 아래와 같이 표현할 수 있음.
	$$\text{distance}={{\mathbf{w}} \over {||\mathbf{w}||}}\cdot\mathbf{x}=\frac{\mathbf{w}^\mathrm{T}\mathbf{x}}{\|\mathbf{w}\|}=-\frac{w_0}{\|\mathbf{w}\|}.$$
	이를 통해, $w_{0}$가 decision boundary와 origin 사이의 거리를 나타낸다. 즉, $w_{0}$가 decision boundary의 위치를 결정한다는 것을 알 수 있습니다.

- $y(\mathbf{x})$는 decision surface로 부터의 perpendicular distance $r$을 나타낸다는 것을 알 수 있음.
	$$\mathbf{x}=\mathbf{x}_\perp+r\frac{\mathbf{w}}{\|\mathbf{w}\|}.$$
	$\mathbf{x}$를 위처럼 나타낼 수 있고, 적절한 식 변형을 통해
	$$r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|}.$$
	를 얻어낼 수 있다는데 모르겠어요... help me...

- $y(\mathbf{x})=\mathbf{w}^\mathrm{T}\mathbf{x}+w_0$에서, $x_{0}=1$인 더미변수를 추가하고, $\widetilde{\mathbf{w}}=(w_0,\mathbf{w}),\widetilde{\mathbf{x}}=(x_0,\mathbf{x})$로 정의하면, $$\begin{aligned}
y(\mathbf{x})&=\mathbf{w}^{\mathrm{T}}\mathbf{x}+w_{0}\\
&=\widetilde{\mathbf{w}}^{\mathrm{T}}\widetilde{\mathbf{x}}.
\end{aligned}$$
	처럼 나타낼 수 있음.

## 5.1.2 Multiple Classes($K$ classes)
- one-versus-the-rest classifier
$K-1$ 개의 classifier가 필요함.

- one-versus-one classifier
$K(K-1)/2$ 개의 classifier가 필요함.

위 두 개의 어려움을 피할 수 있음. How?
by considering a single $K$-class discriminant comprising $K$ linear functions
$$y_{k}(\mathbf{x})=\mathbf{w}^{\mathrm{T}}_{k}\mathbf{x}+w_{k0}$$
위 함수는 $\mathbf{x}$를 $y_k(\mathbf{x})$가 제일 높게 나오는 클래스 $\mathcal{C}_k$에 할당해줌.
그리고 이 경우 두 클래스 $\mathcal{C}_k$ 와 $\mathcal{C}_j$ 의 decision boundary는, $y_{k}(\mathbf{x})=y_{j}(\mathbf{x})$ 임.
그리고 이 decision boundary는 항상 singly connected고 convex임. 증명은 책에 있음.

## 5.1.3 1-of-$K$ coding
분류 문제에서는 target variable을 one-hot-encoding 하는 것이 좋음.

만약 우리가 $K=5$개의 클래스가 있고, 한 데이터가 2번 클래스에 속한다면,
$$\mathbf{t}=(0,1,0,0,0)^{\mathbf{T}}$$처럼 코딩하면 됨.
이렇게 코딩하면, 우리는 $t_{k}$를 $\mathbf{x}$가 $\mathcal{C}_k$에 속할 확률이라고 이해할 수 있음.

## 5.1.4 Least squares for classification
1-of-$K$ coding을 통해 $K$ 개의 클래스를 분류하는 문제를 푼다고 해보자.
그러면 각 클래스 $\mathcal{C}_{k}$는 아래 식 처럼 표현이 된다.
$$y_k(\mathbf{x})=\mathbf{w}_k^\mathrm{T}\mathbf{x}+w_{k0} \quad k=1,\dots,K$$
이를 하나로 묶어서 아래처럼 표현하자. (자세한 설명은 필기로)
$$\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{{\mathrm{T}}}\widetilde{\mathbf{x}}$$
그리고 우리는 이제 $\widetilde{\mathbf{W}}$를 sum-of-square error function을 최소화 해서 구할 것이다.
error function은 아래와 같이 표현된다.
$$E_D(\widetilde{\mathbf{W}})=\frac12\mathrm{Tr}\left\{(\widetilde{\mathbf{X}}\widetilde{\mathbf{W}}-\mathbf{T})^\mathrm{T}(\widetilde{\mathbf{X}}\widetilde{\mathbf{W}}-\mathbf{T})\right\}$$
얘를 $\widetilde{\mathbf{W}}$에 대해 미분하고 해석적인 해를 구하면, 
$$\widetilde{\mathbf{W}}=(\widetilde{\mathbf{X}}^{{\mathrm{T}}}\widetilde{\mathbf{X}})^{-1}\widetilde{\mathbf{X}}^{{\mathrm{T}}}\mathbf{T}=\widetilde{\mathbf{X}}^{\dagger}\mathbf{T}$$
임을 알 수 있다. 따라서 우리의 discriminant function은 아래와 같다.
$$\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{\mathrm{T}}\widetilde{\mathbf{x}}=\mathbf{T}^{\mathrm{T}}\left(\widetilde{\mathbf{X}}^{\dagger}\right)^{\mathrm{T}}\widetilde{\mathbf{x}}.$$

### 단점
데이터의 true distribution이 Gaussian과 많이 다르면, LS는 좋지 않은 결과를 줌.
LS는 outlier에 매우 민감함.
아래 그림에서 볼 수 있듯이 decision boundary가 확 달라짐.
![[5.2.png]]

# 5.2 Decision Theory
우리가 input vector $\mathbf{x}$와 그에 따른 target vector $\mathbf{t}$ 를 입력받았다고 해보자.
우리의 목표는 새로운 $\mathbf{x}$가 주어졌을 때 $\mathbf{t}$ 를 예측하는 것.
이때 $\mathbf{t}$ 는 class label이 될 것임.

## 5.2.1 Misclassification rate
우리의 목표가, 가능한 misclassification을 적게 만드는 것이라고 해보자.
예를 들어 두 개의 클래스로 분류하는 문제에서, mistake를 할 확률은 아래와 같이 표현된다.
$$\begin{aligned}p(\mathrm{mistake})&=\quad p(\mathbf{x}\in\mathcal{R}_1,\mathcal{C}_2)+p(\mathbf{x}\in\mathcal{R}_2,\mathcal{C}_1)\\&=\quad\int_{\mathcal{R}_1}p(\mathbf{x},\mathcal{C}_2) \mathrm{d}\mathbf{x}+\int_{\mathcal{R}_2}p(\mathbf{x},\mathcal{C}_1) \mathrm{d}\mathbf{x}.\end{aligned}$$

그리고 mistake할 확률을 최소화 시키는 것은, posterior probability가 최대가 되는 선택을 함으로써 이뤄질 수 있다.

비슷하게, $K$ 개의 class를 분류하는 문제에서는 맞을 확률을 최대화 하는게 더 편하다.
$$\begin{aligned}p(\mathrm{correct})&=\quad\sum_{k=1}^Kp(\mathbf{x}\in\mathcal{R}_k,\mathcal{C}_k)\\&=\quad\sum_{k=1}^K\int_{\mathcal{R}_k}p(\mathbf{x},\mathcal{C}_k) \mathrm{d}\mathbf{x},\end{aligned}$$
그리고 이 경우, 맞을 확률은 $\mathcal{R}_k$이, posterior probability를 최대화 하도록 선택됐을 때 최대화된다.

## 5.2.2 Expected Loss
하지만 실제로 우리의 목표는, 단순히 misclassification을 최소화 하는 것 보다 더 복잡하다.
예를 들어, 암이 없는 환자를 암이 있다고 오분류 하는 것과, 암이 있는 환자를 암이 없다고 오분류 하는 것은 그 위험이 매우 다르다.

우리는 위와 같은 문제를 loss function(=cost function)을 소개함으로서 다룰 수 있다.
이때 loss function은 우리가 특정 행동이나 결정을 함으로써 발생하는 loss의 measure 이다.
+) 어떤 사람들은 loss function의 반대로 utility function을 이야기 하면서, utility function을 maximize하는 것으로 보기도 한다. 우리가 $\text{utility function} = - \text{loss function}$으로 정의하면 이는 같은 문제가 된다.

우리는 loss function을 최소화 하는 solution을 찾는 것이 목표이다.
average loss는 아래와 같이 표현된다.
$$\mathbb{E}[L]=\sum_k\sum_j\int_{\mathcal{R}_j}L_{kj}p(\mathbf{x},\mathcal{C}_k) \mathrm{d}\mathbf{x}.$$
이때 $L_{kj}$는 주어진 input이 실제로는 $\mathcal{C}_k$에 속하지만, 우리가 이를 $\mathcal{C}_j$로 분류한 개수를 말한다.
그리고 이 $L_{kj}$ 들이 모여 loss matrix를 만든다.
	loss matrix 예)
	![[5.7.png]]

우리의 목표는 위 loss가 최소화 되는 $\mathcal{R}_{j}$를 찾는 것이기에, 우리는 아래 식을 최소화 하면 된다.
$$\sum_kL_{kj}p(\mathbf{x},\mathcal{C}_k)$$
그리고 $p(\mathbf{x},\mathcal{C}_k) = p(\mathcal{C}_k|\mathbf{x})p(\mathbf{x})$ 이고, $p(\mathbf{x})$ 는 모든 클래스에서 공통된 항이기에, 결국 우리는 최종적으로 아래 식을 최소화 하면 된다.
$$\sum_kL_{kj}\ p(\mathcal{C}_k|\mathbf{x})$$

## 5.2.3 The reject option
분류하기 어려운 사례에 대해서 결정을 미루는 것.
적절한 임계값 $\theta$를 설정해서, posterior가 그 임계값을 넘지 못하는 경우, 결정을 하지 않음.
![[5.3.png]]

## 5.2.4 Inference and decision
분류 문제는, training data로 모델을 학습시키는 **inference stage**와, posterior probability를 이용해 예측을 하는 **decision stage**로 나뉨.
아니면 이 두 과정을 한번에 하는 함수(discriminant function)를 학습시키는 방법도 있음.

decision 문제를 풀어내기 위한 세가지 다른 접근법이 있음.
### 첫 번째 접근법
$p(\mathbf{x}|\mathcal{C}_k)$를 결정하기 위해 inference 문제를 풀고,
$p(\mathcal{C}_k)$를 결정.
그리고 나서는 아래 Bayes' Theorem을 활용해,
$$p(\mathcal{C}_k|\mathbf{x})=\frac{p(\mathbf{x}|\mathcal{C}_k)p(\mathcal{C}_k)}{p(\mathbf{x})}$$
posterior probability $p(\mathcal{C}_k|\mathbf{x})$를 찾음. 얘는 일반적으로 training set에서 각 클래스의 비율로 쉽게 찾을 수 있음.

이때 분모 ${p(\mathbf{x})}$ 는 아래와 같이 찾아질 수 있음.
$$p(\mathbf{x})=\sum_kp(\mathbf{x}|\mathcal{C}_k)p(\mathcal{C}_k).$$

아니면, joint distribution $p(\mathbf{x},\mathcal{C}_k)$를 바로 구해도 됨.

#### 특징
이 방법은 요구되는 것이 많다는 단점이 있음.
하지만 데이터의 분포 $p(\mathbf{x})$를 추정하기에, 새로운 input이 들어왔을 때 outlier detection(or novelty detection)이 가능함.

### 두 번째 접근법
posterior probability $p(\mathcal{C}_k|\mathbf{x})$를 바로 구할 수도 있음.
이러한 접근법을 **discriminative models** 라고 부름.

#### 특징
우리가 오로지 분류 클래스를 알고싶은게 목적이라면, 첫 번째 접근법은 약간 계산 낭비 일 수 있음.
분류 클래스만을 알고 싶다면 $p(\mathcal{C}_k|\mathbf{x})$를 추정하면 ok.

### 세 번째 접근법
$\mathbf{x}$가 주어졌을 때 얘에게 class label을 바로 할당해주는 discriminant function $f(\mathbf{x})$를 찾을 수도 있음.
inference stage와 decision stage를 하나로 합친 것.

이 접근법은 아래 그림에서 연두색 선을 찾는 것과 같음.
![[5.4.png]]

#### 특징 : posterior를 구하지 않음.
세 번째 접근법은 posterior probability를 구하지 않는다는 특징이 있음.

하지만 우리는 아래와 같은 이유로 posterior probability를 구해야할 강력한 이유가 있음.
#### 그럼에도 posterior probability를 계산해야하는 이유
#####  Minimizing Risk
loss matrix의 값들이 자꾸 바뀐다고 가정해보자.
만약 우리가 posterior를 구해놓았다면, 바뀐 loss matrix에 대해 다시 minimum risk decision criterion을 구할 수 있다.

만약 posterior를 구하지 않고, 오로지 discriminant function을 알고 있다면, 우리는 loss matrix에 변화가 있을 때 마다 inference problem을 다시 풀어야한다.
##### Reject option
posterior 는 우리가 reject option을 설정할 수 있게 해준다. 이는 missclassification rate를 최소화 하거나 expected loss를 최소화 하는데 도움을 줌.
##### Compensating for class priors
데이터가 불균형한 경우에 대처 가능.
posterior를 사용하면, 각 클래스의 prior를 반영할 수 있음.
##### Combining models
어떤 경우, 복잡한 문제는 자잘한 문제 여러 개로 쪼개서 풀어내는게 효과적일 수 있음.
만약 우리가 분류에 사용할 설명변수가 두개인 경우, 이 두개를 하나로 합쳐서 사용하는 것 보다 아래처럼 각각의 결과를 하나로 합칠 수 있음.
$$\begin{aligned}p(\mathcal{C}_k|\mathbf{x}_\mathrm{I},\mathbf{x}_\mathrm{B})&\propto\quad p(\mathbf{x}_\mathrm{I},\mathbf{x}_\mathrm{B}|\mathcal{C}_k)p(\mathcal{C}_k)\\&\propto\quad p(\mathbf{x}_\mathrm{I}|\mathcal{C}_k)p(\mathbf{x}_\mathrm{B}|\mathcal{C}_k)p(\mathcal{C}_k)\\&\propto\quad\frac{p(\mathcal{C}_k|\mathbf{x}_\mathrm{I})p(\mathcal{C}_k|\mathbf{x}_\mathrm{B})}{p(\mathcal{C}_k)}.\end{aligned}$$
이 경우 두 설명변수의 독립 가정이 필요한데, 독립 가정이 필요 없는 방법은 추후에 배움.

##### Optimization
output이 클래스가 아니라 확률인 모델을 사용하면, 파라미터에 대해 미분하며 gradient-based optimization을 할 수 있음.

## 5.2.5 Classifier accuracy

| **종류**         | **약어** | **설명**                          | **비고**       |
| -------------- | ------ | ------------------------------- | ------------ |
| True Positive  | TP     | 실제로 positive 인 대상을 positive로 분류 |              |
| False Positive | FP     | 실제로 negative 인 대상을 positive로 분류 | Type 1 error |
| True Negative  | TN     | 실제로 negative 인 대상을 negative로 분류 |              |
| False Negative | FN     | 실제로 positive인 대상을 negative로 분류  | Type 2 error |

| **Metric**           | **Formula**                     | **설명**                           |
| -------------------- | ------------------------------- | -------------------------------- |
| Accuracy             | (TP + TN) / (TP + TN + FP + FN) | 전체 중에 잘 분류된 대상                   |
| Precision            | TP / (TP + FP)                  | test positive 중에 actual positive |
| Recall (Sensitivity) | TP / (TP + FN)                  | actual positive 중에 test positive |
| False positive rate  | FP / (FP + TN)                  | actual negative 중에 오분류된 대상       |
| False discovery rate | FP / (FP + TP)                  | test positive 중에 오분류된 대상         |

예를 들어, 암을 분류하는 문제에서, 
precision은 암이 있다고 판정된 사람이 실제로 암이 있을 확률
recall은 실제로 암이 있는 사람이 암이 있다고 판정될 확률
false positive rate는 실제로 암이 없는 사람이 암이 있다고 판정될 확률
false discovery rate는 암이 있다고 판정된 사람이 실제로는 암이 없을 확률

### Trade-offs
Decision boundary를 바꾸면, 위 값들 사이에는 trade-off가 있음.
$$\begin{aligned}
&N_{\mathrm{FP}}/N =E \\
&N_{\mathrm{TP}}/N =D+E \\
&N_{\mathrm{FN}}/N =B+C \\
&N_{\mathrm{TN}}/N =A+C 
\end{aligned}$$
![[5.5.png]]

#### 주요 영역 설명 (Thanks to GPT)
1. **A 영역**:
   - $p(x, \mathcal{C}_1)$ 곡선 아래의 왼쪽 부분으로, 클래스 $\mathcal{C}_1$에 속하는 샘플들이 올바르게 클래스 $\mathcal{C}_1$로 분류된 영역입니다.
   - 즉, 클래스 $\mathcal{C}_1$을 정확하게 분류한 경우입니다. **True Negative (TN)**.

2. **B 영역**:
   - $p(x, \mathcal{C}_1)$와 $p(x, \mathcal{C}_2)$ 곡선이 겹치는 부분 중에서, 클래스 $\mathcal{C}_1$에 속하는 샘플들이 잘못 클래스 $\mathcal{C}_2$로 분류된 영역입니다.
   - 이 영역은 False Positive (FP)를 나타냅니다. 클래스 $\mathcal{C}_1$의 샘플이 잘못 분류되어 클래스 $\mathcal{C}_2$로 분류된 경우입니다.

3. **C 영역**:
   - $p(x, \mathcal{C}_2)$ 곡선 아래의 영역으로, 클래스 $\mathcal{C}_2$에 속하는 샘플이 잘못 클래스 $\mathcal{C}_1$로 분류된 영역입니다.
   - 이는 False Negative (FN)를 나타냅니다. 클래스 $\mathcal{C}_2$의 샘플이 잘못 분류되어 클래스 $\mathcal{C}_1$로 분류된 경우입니다.

4. **D 영역**:
   - $p(x, \mathcal{C}_2)$ 곡선 아래의 오른쪽 부분으로, 클래스 $\mathcal{C}_2$에 속하는 샘플들이 올바르게 클래스 $\mathcal{C}_2$로 분류된 영역입니다.
   - 즉, 클래스 $\mathcal{C}_2$을 정확하게 분류한 경우입니다. True Positive (TP)

5. **E 영역**:
   - $p(x, \mathcal{C}_2)$와 $p(x, \mathcal{C}_1)$ 곡선이 겹치는 부분 중에서, 클래스 $\mathcal{C}_2$에 속하는 샘플들이 잘못 클래스 $\mathcal{C}_1$로 분류된 영역입니다.
   - 이 영역은 False Negative (FN)입니다. 클래스 $\mathcal{C}_2$의 샘플이 잘못 분류되어 클래스 $\mathcal{C}_1$로 분류된 경우입니다.

## 5.2.6 ROC Curve
type 1 error와 type 2 error 사이에는 trade-off가 있음.
이 trade-off를 잘 시각화한게 ROC Curve.
위 그림에서 decision boundary를 $-\infty$ 에서 $\infty$ 까지 바꾸어 가며, 각 상황에서 True positive rate와 False positive rate를 계산하여 점을 찍은 것.
![[5.6.png]]

Confusion matrix는 ROC Curve 위의 한 점을 나타냄.

ROC Curve가 좌측 상단 코너에 가까울 수록 좋은 성능의 분류기임.
좌측 하단 코너는, 분류기가 전부 negative로 예측하는 상황.
우측 상단 코너는, 분류기가 전부 positive로 예측하는 상황. 

### ROC Curve의 전체적인 모습을 나타내는 두 가지 metric
- **AUC**
Area Under the Curve 로, 0.5에 가까우면 성능이 낮은 분류모델, 1에 가까우면 성능이 좋은 분류모델
- **F-score**
$$\begin{gathered}
\text{F} =\frac{2\times\text{precision}\times\text{recall}}{\text{precision}+\text{recall}} \\
=\frac{2N_{\mathrm{TP}}}{2N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{FN}}}. 
\end{gathered}$$