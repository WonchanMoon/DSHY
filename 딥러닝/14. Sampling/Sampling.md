딥러닝에서는, 확률분포 $p(\mathbf{z})$로 부터 $\mathbf{z}$를 생성해야하는 상황이 많다.
이때 $\mathbf{z}$는, univariate Gaussian으로부터 샘플링된 스칼라 값일 수도 있고,
고해상도의 이미지일 수도 있으며, Deep Neural Network로 정의된 생성형 모델일 수도 있다.

이러한 examples를 생성하는 일련의 과정을 **sampling**이라고 하며, **Monte Carlo Sampling**이라고 하기도 한다.

딥러닝에서의 **sample, 샘플**은 생성된 값을 의미하고, 통계학에서 값들의 집합을 의미하던 것과는 다르다. 

# 14.1 Basic Sampling Algorithms
여기서는, 주어진 분포로부터 랜덤 샘플을 생성하기 위한 간단한 전략들을 소개.
하지만 샘플링은 컴퓨터 알고리즘을 통해 이루어지기 때문에, 실제로는 pseudo-random이고, 무작위성 테스트를 통과해야한다.
> 컴퓨터는 본질적으로 결정론적(deterministic) 장치이기 때문에, **완전한 랜덤 값을 생성할 수는 없음**.
> 따라서 pseudo-random 알고리즘을 통해 **랜덤처럼 보이는 숫자들을 생성**.
> 그리고나서 이렇게 생성된 숫자들이 정말 무작위인지(통계적으로 랜덤한 분포를 만족하는지)를 검정

## 14.1.1 Expectations
우리가 샘플을 뽑을 때, 그 샘플 자체에 관심이 있을 수도 있지만, **expectation을 계산하는 것**이 우리의 목표일 수 있다.
예를 들어 우리가 확률분포 $p(\mathbf{z})$에 대해, $f(\mathbf{z})$의 기댓값을 구하고 싶다고 생각해보자.
$$\mathbb{E}[f] = \int f(\mathbf{z}) p(\mathbf{z}) \, dz
$$
적분 계산이 간단하다면 그냥 적분하면 되지만, 함수가 복잡한 경우 이 적분은 어려울 수 있다.

### 가장 기본적인 IDEA
확률분포 $p(\mathbf{z})$를 따르는 $L$ 개의 independent 샘플 $\mathbf{z}^{(l)},\text{where }l=1,...,L$ 을 뽑아서 기댓값을 approximate.
$$\overline{f} = \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}^{(l)})
$$

만약 샘플 $\mathbf{z}^{(l)}$가 확률분포 $p(\mathbf{z})$에서 뽑아졌다면, 아래 식을 만족한다.
$$\mathbb{E}[\overline{f}]=\mathbb{E}[f(\mathbf{z})]$$
따라서 estimator $\overline{f}$는 correct mean(올바른 기댓값?)이 된다.
> 올바른 기댓값?  :  샘플링을 통해 계산된 추정치가, 이론적인 기댓값을 제대로 반영하다는 것.
> $p(\mathbf{z})$로 부터 샘플들이 올바르게 뽑혔다면, $\overline{f}$의 기댓값은 $\mathbb{E}[f(\mathbf{z})]$와 같아진다는 것!

위 과정을 아래 식으로 표현할 수 있다.
$$\mathbb{E}[f(\mathbf{z})] \simeq \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}^{(l)})
$$
이때 연산자 $\simeq$는, 우변이 좌변의 unbiased estimator임을 나타낸다.

estimator $\overline{f}$의 분산은 다음과 같이 계산된다.
$$\operatorname{var}[\overline{f}] = \frac{1}{L} \mathbb{E} \left[ \left( f - \mathbb{E}[f] \right)^2 \right]
$$
이 분산은 확률분포 $p(\mathbf{z})$ 하에서 $f(\mathbf{z})$의 분산이다.

위 식을 통해 다음 사실들을 알 수 있음.
- **샘플이 독립**이라면, 분산의 크기는 $L$의 값이 증가함에 따라 **선형적으로 감소**. 이론적으로 작은 $L$에서도 낮은 분산(높은 정확도)을 만들 수 있음.
- 하지만 일반적으로 샘플은 독립이 아님.
- 따라서 우리는 충분한 정확도를 얻기 위해 많은 샘플이 필요함. 

## 14.1.2 Standard distributions
이제, nonuniform distribution에서 random number를 생성하는 법을 고려한다.
이때 우리는 uniform distribution에서 random number를 뽑을 수 있다고 가정한다.
> uniform distribution -> nonuniform distribution으로의 변환을 원하는 듯?

$z$가 $(0,1)$에서 uniform distribution을 따른다고 가정하자. -> $p(z)=1$
어떤 함수 $g$를 이용하여, $y=g(z)$로 변환하면, $y$의 확률분포 $p(y)$는 다음과 같다.
$$
p(y) = p(z) \left| \frac{dz}{dy} \right| \tag{14.5}
$$
우리의 목표는, $y$가 특정 확률분포 $p(y)$를 따르도록 하는 **적절한 $g(z)$를 선택하는 것**이다.
#### multiple variable Jacobian
$$p(y_1, \dots, y_M) = p(z_1, \dots, z_M) \left| \frac{\partial (z_1, \dots, z_M)}{\partial (y_1, \dots, y_M)} \right|
$$

$(14.5)$을 적분하면, 다음과 같은 결과를 얻을 수 있다.
$$z = \int_{-\infty}^{y} p(\hat{y})\ \mathrm{d} \hat{y}  \equiv h(y) \tag{14.6}
$$
> 왜 계산이 이렇게 되는지 모르겠음

아무튼, $(14.6)$으로 부터 $y = h^{-1}(z)$임을 알 수 있다.

따라서 우리는, 우리가 원하는 특정 확률분포 $p(y)$의 적분의 역함수를 통해,
uniformly distributed된 random number $z$를, $y=h^{-1}(z)$ 변환이 가능하다.

#### Example: Exponential Distribution
$$p(y)=\lambda \exp (-\lambda y) \quad \text{where }0\leq y < \infty. $$
이 경우, $(14.6)$의 적분 구간은 $0$부터 이고, $h(y)=1-\exp (-\lambda y)$가 된다.
따라서, 우리는 uniformly distributed variable $z$를 다음 변환을 통해 exponential distribution을 따르는 변수 $y$로 바꾸어 줄 수 있다.
$$y=h^{-1}(z)=-\lambda^{-1}\ln (1-z)$$

### Box-Muller
> Recall) 수리통계1 기말

1. uniform distribution에서 두 random numbers $z_{1},z_{2}\in (-1,1)$을 뽑는다.
2. $z \rightarrow 2z-1$ 변환으로 $(0,1)$ 사이의 값으로 변환 (*처음부터 $(0,1)$에서 뽑으면 안되나?*)
3. $z^{2}_{1}+z^{2}_{2}\leq 1$을 만족하지 않는 쌍은 다 버린다.
4. 3. 까지 마친다면, 모든 $z_{1},z_{2}$ 조합은 $p(z_{1},z_{2})=1/\pi$ 를 만족한다.
5. 이제 아래 식을 통해 $z_{1},z_{2}$를 서로 독립인 표준 정규분포를 따르는 두 변수 $y_{1},y_{2}$로 변환
	$$\begin{align*}
y_1 &= z_1 \left( \frac{-2 \ln r^2}{r^2} \right)^{1/2} \\
y_2 &= z_2 \left( \frac{-2 \ln r^2}{r^2} \right)^{1/2}
\end{align*}
$$
	이때 $r^{2}=z_{1}^{2}+z_{2}^{2}$

최종적으로 $y_{1},y_{2}$의 joint distribution은 다음과 같이 주어진다. 

$$
\begin{align*}
p(y_1, y_2) &= p(z_1, z_2) \left| \frac{\partial (z_1, z_2)}{\partial (y_1, y_2)} \right| \\
&= \left[ \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y_1^2}{2}\right) \right] \left[ \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y_2^2}{2}\right) \right] \tag{14.12}
\end{align*}

$$

#### Gaussian -> Gaussian
$$y\sim \mathcal{N}(0,1) \rightarrow \sigma y+\mu \sim \mathcal{N}(\mu,\sigma^{2})$$

#### Multivariate Gaussian
평균 $\mathbf{\mu}$, 공분산 $\mathbf{\Sigma}$를 가지는 다변량 정규분포를 따르는 벡터 값을 생성하기 위해,
**Cholesky 분해**($\mathbf{\Sigma}=LL^{T}$)를 활용할 수 있다.

$$\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)\ 
\rightarrow\ 
\mathbf{\mu}+L\mathbf{z}\sim\mathcal{N}(\mathbf{\mu},\mathbf{\Sigma})$$

지금까지 본 변환은, 
목표로 하는 분포의 **부정적분을 계산할 수 있어야**하고, **inverse를 구해야**했다. 하지만 이는 굉장히 일부 분포에서만 가능하기에, 우리는 다른 접근 방식을 고려해야한다.

---
## 14.1.3 Rejection Sampling
**Rejection Sampling**은, 우리가 비교적 복잡한 분포에서 샘플링을 할 수 있도록 해준다.