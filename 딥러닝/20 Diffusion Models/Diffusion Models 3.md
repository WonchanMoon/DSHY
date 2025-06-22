# 20.3 Score Matching
지금까지 다룬 디퓨전 모델은, **score matching에 기반을 둔 생성형 모델**과 관련이 있다.

#### score function
$$
\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \ln p(\mathbf{x}).
$$

$\mathbf{s}(\mathbf{x})$는 데이터 벡터 $\mathbf{x}$와 같은 차원을 가지며,
각 원소 : $s_{i}(\mathbf{x})=\partial\ln p (\mathbf{x}) / \partial x_{i}$
즉, $\mathbf{s}$의 원소 $\mathbf{s}_{i}$는 각각 $\mathbf{x}$의 원소 $\mathbf{x}_{i}$에 대응됨.

스코어 함수는, **데이터가 있는 위치 $\mathbf{x}$에서, 확률분포 $p(\mathbf{x})$가 증가하는 방향을 가리키고 있는 함수**.

여기서 스코어 함수가 왜 유용한지 알 수 있는데,
스코어 함수만 잘 학습해도, 원래 확률분포 $p(\mathbf{x})$를 거의 완전히 재현할 수 있음.
![[20.3.1.png]]

## 20.3.1 Score loss function
우리는 데이터를 생성하는 확률분포 $p(\mathbf{x})$의 스코어 함수 $\nabla_{\mathbf{x}} \ln p(\mathbf{x})$를 근사하는 $s(\mathbf{x}, \mathbf{w})$를 구할 것임.

#### loss function
$$
J(\mathbf{w}) = \frac{1}{2} \int \left\| \mathbf{s}(\mathbf{x}, \mathbf{w}) - \nabla_{\mathbf{x}} \ln p(\mathbf{x}) \right\|^2 p(\mathbf{x}) \, \mathrm{d}\mathbf{x}.
$$

신경망을 사용하여 $\mathbf{s}(\mathbf{x}, \mathbf{w})$를 나타내는 방법에는 크게 두 가지가 있음.
1. $\mathbf{x}$와 같은 차원의 출력을 가지는 신경망을 구성하여, 각 $\mathbf{s}_{i}$를 독립적으로 예측
2. $\phi(\mathbf{x})$를 근사하는 신경망을 구성하고, 자동미분으로 $\nabla_{\mathbf{x}}\phi(\mathbf{x})$를 계산

하지만 두번째 방법은, 계산비용이 너무 높아서 첫번째 방법을 선호

## 20.3.2 Modified Score Loss
우리가 위의 loss function을 계산하려해도, $\nabla_{\mathbf{x}} \ln p(\mathbf{x})$를 모르기 때문에 계산을 할 수가 없다. 
하지만 우리는 유한한 개수의 데이터 $\mathcal{D}=(\mathbf{x}_{1},...,\mathbf{x}_{N})$을 갖고 있기 때문에, 우리는 empirical distribution을 구할 수 있음.
$$
p_{\mathcal{D}}(\mathbf{x}) = \frac{1}{N} \sum_{n=1}^{N} \delta(\mathbf{x} - \mathbf{x}_n).
$$
> $\delta(\mathbf{x})$는 Dirac Delta Function

하지만 empirical distribution 식은 미분이 불가능함.
-> kernel density estimation으로 미분이 가능한 함수를 만들어버림.

> 이해를 돕기 위한 그림 설명
> https://www.wikiwand.com/en/articles/density_estimation
> ![[20.3.2.png]]
> 빨간색 : empirical distribution
> 회색 점선 : Gaussian kernel을 씌운 결과
> 검은색 점선 : Kernel Density estimation
> 파란 실선 : True

$$
q_{\sigma}(\mathbf{z}) = \int q(\mathbf{z} \mid \mathbf{x}, \sigma) p(\mathbf{x}) \, \mathrm{d}\mathbf{x}
$$
이때 $q(\mathbf{z}\mid\mathbf{x},\sigma)$는 **noise kernel**이고 가장 흔한 선택은 **가우시안 커널**
$$
q(\mathbf{z} \mid \mathbf{x}, \sigma) = \mathcal{N}(\mathbf{z} \mid \mathbf{x}, \sigma^2 \mathbf{I}).
$$

따라서 $p(\mathbf{x})$ 대신 $q_{\sigma}(\mathbf{z})$를 선택함으로써, loss function을 다음과 같이 바꿀 수 있다.
$$
J(\mathbf{w}) = \frac{1}{2} \int \left\| \mathbf{s}(\mathbf{z}, \mathbf{w}) - \nabla_{\mathbf{z}} \ln q_{\sigma}(\mathbf{z}) \right\|^2 q_{\sigma}(\mathbf{z}) \, \mathrm{d}\mathbf{z}.
$$
그리고,
$$
q_{\sigma}(\mathbf{z}) = \int q(\mathbf{z} \mid \mathbf{x}, \sigma) p(\mathbf{x}) \, \mathrm{d}\mathbf{x}
$$
임을 이용하여 loss function을 다시 작성하면,
$$
J(\mathbf{w}) = \frac{1}{2} \iint \left\| \mathbf{s}(\mathbf{z}, \mathbf{w}) - \nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}, \sigma) \right\|^2 q(\mathbf{z} \mid \mathbf{x}, \sigma) p(\mathbf{x}) \, \mathrm{d}\mathbf{z} \, \mathrm{d}\mathbf{x} + \text{const.}
$$
이다.

그리고 우리가 $p(\mathbf{x})$를 empirical하게 구했다는 것을 반영해주면,
$$
J(\mathbf{w}) = \frac{1}{2N} \sum_{n=1}^{N} \int \left\| \mathbf{s}(\mathbf{z}, \mathbf{w}) - \nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}_n, \sigma) \right\|^2 q(\mathbf{z} \mid \mathbf{x}_n, \sigma) \, \mathrm{d}\mathbf{z} + \text{const.}
$$
이다.

#### Diffusion Model과의 유사성
kernel density estimation에서의 score function은 아래와 같음.
$$
\nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}, \sigma) = -\frac{1}{\sigma} \boldsymbol{\epsilon}
$$

Diffusion Model에서,
$$
q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}_t \mid \sqrt{\alpha_t} \mathbf{x}, (1 - \alpha_t)\mathbf{I})
$$
의 score function은 아래와 같음
$$
\nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}, \sigma) = -\frac{1}{\sqrt{1 - \alpha_t}} \boldsymbol{\epsilon}
$$

스코어 기반 모델에서는 스코어 함수, 즉 $-\boldsymbol{\epsilon}/\sigma$를 예측하는 것이 목표였고,
디퓨전 모델에서는 $\boldsymbol{\epsilon}$을 예측하는 것이 목표이므로,
두 모델의 본질은 같다고 볼 수 있음.
## 20.3.3 Noise Variance
Song and Ermon, 2019; Luo. 2022에 따르면 스코어 기반 모델이 학습 된 후, 새로운 샘플을 생성하는 과정에서 다음과 같은 세 가지의 문제점이 있음.
1. 데이터가 낮은 차원의 manifold 위에 존재할 경우, 확률분포 $p(\mathbf{x})$가 0이 되는 경우가 있어서 $\ln p(\mathbf{x})$가 존재하지 않는 경우가 있음.
2. 데이터 밀도가 낮은 곳에서의 score 함수 추정은 부정확해짐
3. 데이터의 분포가 독립인 분포들의 mixture일 경우, 샘플링이 잘 작동하지 않음.

#### 이러한 문제를 해결하기 위해
다양한 노이즈 값에서 스코어 함수를 학습할 수 있음
$\sigma_{1}^{2}<\cdots<\sigma_{T}^{2}$로 노이즈 값을 늘려가면서, 다양한 노이즈에서 스코어 함수를 학습.
$$
\frac{1}{2} \sum_{i=1}^{L} \lambda(i) \int \left\| \mathbf{s}(\mathbf{z}, \mathbf{w}, \sigma_i^2) - \nabla_{\mathbf{z}} \ln q(\mathbf{z} \mid \mathbf{x}_n, \sigma_i) \right\|^2 q(\mathbf{z} \mid \mathbf{x}_n, \sigma_i) \, \mathrm{d}\mathbf{z}
$$

## 20.3.4 Stochastic differential equations
Diffusion Model은, 점진적으로 노이즈를 추가하는 방식으로 작동.
이런 점진적인 과정을 **무한히 많은 단계**로 확장하면, **연속적인 시간 개념**으로 볼 수 있음.
따라서 노이즈를 넣는 과정을 **Stochastic Differential Equation**으로 모델링 할 수 있음.

#### Forward Process
$$
\mathrm{d}\mathbf{z} = \underbrace{\mathbf{f}(\mathbf{z}, t)\, \mathrm{d}t}_{\text{drift}} + \underbrace{g(t)\, \mathrm{d}\mathbf{v}}_{\text{diffusion}}
$$
drift 항은, 노이즈가 추가되면서 데이터가 특정 방향으로 흘러가도록 해주는데, 일반적으로는 0
diffusion 항이 노이즈를 추가하는 항.

#### Reverse Process
$$
\mathrm{d}\mathbf{z} = \left\{ \mathbf{f}(\mathbf{z}, t) - g^2(t) \nabla_{\mathbf{z}} \ln p(\mathbf{z}) \right\} \mathrm{d}t + g(t)\, \mathrm{d}\mathbf{v}
$$
역방향 역시 SDE로 표현됨.
$\nabla_{\mathbf{z}} \ln p(\mathbf{z})$가 score 함수.

#### SDE를 ODE로 바꿀수도 있는데,
$$
\frac{\mathrm{d}\mathbf{z}}{\mathrm{d}t} = \mathbf{f}(\mathbf{z}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{z}} \ln p(\mathbf{z}).
$$
이렇게 ODE로 하는게 더 좋음.
