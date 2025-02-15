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

## 14.1.3 Rejection Sampling
**Rejection Sampling**은, 우리가 비교적 복잡한 분포에서 샘플링을 할 수 있도록 해준다.
univariate를 먼저 생각해보고, 다차원으로 넘어가자.

우리가 $p(\mathbf{z})$에서 샘플링을 하고 싶은데, $p(\mathbf{z})$에서 바로 샘플링을 하기에는 난이도가 있을 수 있다. 우리는 그래서, 아래 식을 만족하는 $\tilde{p}(\mathbf{z})$를 이용하여 샘플링을 할 것이다.
$$p(\mathbf{z})=\frac{1}{Z_{p}}\tilde{p}(\mathbf{z})$$
> ex) Bayesian Inference
> 우리가 원하는 분포가 아래와 같은 경우,
> $$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}$$여기서 우리는 $p(D)$를 구하기 쉽지 않기 때문에, $\tilde{p}(\theta|D)=p(D|\theta)p(\theta)$과 Rejection Sampling을 이용하여 사후 분포에서 샘플링을 할 수 있다.

우리가 Rejection Sampling을 적용하기 위해서는, 간단한 분포 $q(z)$가 필요하다. 그리고 이 간단한 분포를 **proposal distribution**이라고 하고, 우리는 여기서 쉽게 샘플을 뽑을 수 있다. 그리고나서 우리는 모든 $z$에서 $kq(z)\geq\tilde{p}(z)$를 만족하는 상수 $k$를 골라야한다. $kq(z)$는 **comparison function**이라고 한다.

Rejection Sampling에서는 두 개의 random number를 뽑는다.
1. $q(z)$ 분포에서 $z_{0}$를 뽑는다.
2. 그리고 uniform distribution $[0,k(q(z_{0}))]$에서 $u_{0}$를 뽑는다.
이때, $u_{0}>\tilde{p}(z_{0})$이면 샘플은 기각되고, 반대의 경우 $u_{0}$는 선택된다.
![[14.1.1.png]]
위 과정을 거치는 과정에서, 회색 부분에 위치한 샘플들은 전부 기각된다. 그리고 남은 샘플들은, 우리가 의도했던 $\tilde{p}(z)$의 분포에 가까워진다.

샘플링 알고리즘에서는, 원하는 분포에서 빠르게 샘플을 생성해야한다. Rejection sampling에서는 랜덤하게 생성한 샘플을 기각하는 과정이 들어가 있기에, 시간이 조금 걸릴 수도 있다. 이를 채택률과 관련해서 생각해볼 수 있다. 위 과정을 통해 뽑은 샘플이 기각되지 않고 수용될 확률, $p(\text{accept})$는 다음과 같이 계산된다.
$$
p(\text{accept}) = \int \left\{ \frac{\tilde{p}(z)}{kq(z)} \right\} q(z) \, dz
= \frac{1}{k} \int \tilde{p}(z) \, dz.
$$
즉, 수용될 확률을 높이기 위해서는 $k$를 최대한 작은 값으로 골라야 한다.

### 예시 : 감마분포
$$\text{Gam}(z \mid a, b) = \frac{b^a z^{a-1} \exp(-bz)}{\Gamma(a)}$$
위 감마분포에서 $a > 1$이면 bell-shaped form이다.
proposal distribution으로 Cauchy 분포를 채택하여 Rejection Sampling을 할 수 있다.
> Cauchy 분포 sampling
> $y$ : uniform distribution -> $z=b\tan y + c$ : Cauchy distribution
> $$q(z)=\frac{k}{1+(z-c)^{2}/b^{2}}$$

Rejection Rate는 $c = a-1, b^{2}=2a -1$로 채택했을 때 제일 낮아진다.
![[14.1.2.png]]

## 14.1.4 Adaptive rejection sampling
Rejection sampling을 하고 싶지만, 적절한 $q(z)$를 선택하는 것이 어려울 수 있다. 
대안으로, $p(z)$의 측정된 값을 바탕으로 $q(z)$를 만들어나가는 방법이 제시됐다 (Gilks and Wild, 1992). 

특히 $p(z)$가 log-concave(log 씌웠을 때 concave)라면, $q(z)$를 만드는 과정이 매우 간단해진다.
우리가 알고 있는 일부 초기 점들을 통해, $\ln p(z)$의 값과 gradient를 계산해서, 일차함수들로 $q(z)$를 만들어 나간다.
![[14.1.3.png]]
이때, 각 구간의 $q(z)$는 다음과 같이 표현이 가능하다.
$$
q(z) = k_i \lambda_i \exp \left\{ -\lambda_i (z - z_{i-1}) \right\}, \quad z_{i-1} < z \leq z_i.
$$
$-\lambda_{i}$는 $\ln p(z)$의 $z_{i}$에서의 기울기.

초기 점들이 많아질수록, $q(z)$는 $p(z)$에 더더욱 가까워지고, 거부 확률이 점점 감소하게 된다.
14.2.3 에서, 이 알고리즘의 변형된 버전에 대해 나온다.

하지만 Rejection Sampling은, 고차원에서는 그 효율이 매우 감소한다는 단점이 있다.
> 아주 간단하게, 우리는 공분산이 $\sigma^{2}_{p}I$, 평균이 $0$인 다변량 가우시안에서 샘플링을 한다고 해보자. 우리는 이를 위해 공분산이 $\sigma^{2}_{q}I$, 평균이 $0$인 또 다른 다변량 가우시안을 proposal distribution으로 설정하여 rejection sampling을 시행했다고 하자.
> 이 경우 $kq(z)\geq p(z)$를 만족하려면, $\sigma^{2}_{q}\geq\sigma^{2}_{p}$ 가 성립해야하고, $D$차원에서 최적의 $k$는 $k=(\sigma_{q}/\sigma_{p})^{D}$ 이다. 이 $k$에서, 수락율은 $1/k$이다. 즉, 차원이 커짐에 따라 수락율은 지수적으로 감소한다.

## 14.1.5 Importance Sampling
**Importance Sampling**은 기댓값을 근사할 수 있는 좋은 툴을 제공. 하지만 확률 분포에서 샘플을 추출하는 법 자체를 제공하지는 않음.

Importance Sampling에서도, rejection sampling과 마찬가지로 proposal distribution을 이용한다.
$$
\begin{aligned}
\mathbb{E}[f] &= \int f(\mathbf{z}) p(\mathbf{z}) \, d\mathbf{z} \\
&= \int f(\mathbf{z}) \frac{p(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z}) \, d\mathbf{z} \\
&\approx \frac{1}{L} \sum_{l=1}^{L} \frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})} f(\mathbf{z}^{(l)}).
\end{aligned}
$$
여기서 $r_{l}=p(\mathbf{z}^{(l)})/q(\mathbf{z}^{(l)})$은 **importance weights**로, 잘못된 분포에서 샘플링함으로써 발생되는 편향을 보정하는 역할을 한다. 그리고 rejection sampling과 달리, importance sampling에서는 생성된 모든 샘플이 유지된다.

앞서 우리가 rejection sampling에서,
$$p(\mathbf{z})=\frac{1}{Z_{p}}\tilde{p}(\mathbf{z})$$
위와 같은 변환을 이용하였다. 여기서도 똑같이 적용할 수 있다.
$$
\begin{aligned}
\mathbb{E}[f] &= \int f(\mathbf{z}) p(\mathbf{z}) \, d\mathbf{z} \\
&= \frac{Z_q}{Z_p} \int f(\mathbf{z}) \frac{\tilde{p}(\mathbf{z})}{\tilde{q}(\mathbf{z})} q(\mathbf{z}) \, d\mathbf{z} \\
&\approx \frac{Z_q}{Z_p} \frac{1}{L} \sum_{l=1}^{L} \tilde{r}_l f(\mathbf{z}^{(l)}).
\end{aligned}
$$이때 $\tilde{r}_{l}=\tilde{p}(\mathbf{z}^{(l)})/\tilde{q}(\mathbf{z}^{l})$ 이다.
그리고 $Z_{p}/Z_{q}$도 계산이 가능한데,
$$
\begin{aligned}
\frac{Z_p}{Z_q} &= \frac{1}{Z_q} \int \tilde{p}(\mathbf{z}) \, d\mathbf{z} 
= \int \frac{\tilde{p}(\mathbf{z})}{\tilde{q}(\mathbf{z})} q(\mathbf{z}) \, d\mathbf{z} \\
&\approx \frac{1}{L} \sum_{l=1}^{L} \tilde{r}_l.
\end{aligned}
$$
이를 통해 $\mathbb{E}[f]$를 계산할 수 있다.
$$
\mathbb{E}[f] \approx \sum_{l=1}^{L} w_l f(\mathbf{z}^{(l)})
$$
이때 $w_{l}$은 다음과 같다.
$$
w_l = \frac{\tilde{r}_l}{\sum_m \tilde{r}_m} = 
\frac{\tilde{p}(\mathbf{z}^{(l)})/q(\mathbf{z}^{(l)})}
{\sum_m \tilde{p}(\mathbf{z}^{(m)})/q(\mathbf{z}^{(m)})}.
$$


Importance Sampling 역시 $q(z)$가 $p(z)$를 얼마나 잘 맞추냐에 따라 성능이 크게 달라진다. 특히 우리가 관심있는 $p(z)$가, 아주 작은 특정 영역에 몰려있으면 샘플링이 잘못될 위험이 크다.

## 14.1.6 Sampling-importance-resampling
> SIR은, 주어진 확률 분포에서 직접 샘플링이 어려울 때, 더 쉽게 샘플링할 수 있는 분포를 이용해 샘플을 생성하고, 중요도 가중치를 적용하여 재샘플링하는 방식입니다. (from GPT)

SIR은 크게 세 단계로 구성.
#### 1. 샘플링
Proposal Distribution에서 $N$개의 sample을 생성.
$$z^{(1)},z^{(2)},...,z^{(N)}$$

#### 2. Importance Weighting
각 샘플에 대해, $p(x)$와 $q(x)$간의 비율을 기반으로 가중치를 계산.
$$w_{i}=\frac{p(z^{(i)})}{q(z^{(i)})}$$
이 가중치는, 특정 샘플이 실제 목표 분포($p(z)$)에서 얼마나 중요한지를 나타냄.
> 가중치가 곧 리샘플링에서 각 샘플이 뽑힐 확률이다.

#### 3. Resampling
가중치가 큰 샘플은 높은 확률로 유지, 가중치가 작은 샘플은 제거하여 새롭게 샘플링.
> 책에서는 제거한다고 하던데, GPT의 답변으로는 각 샘플들의 가중치를 정규화하여 다시 샘플링 한다고 함.

하지만 이렇게 무수히 리샘플링을 반복하면 일부 샘플만 살아남는 문제가 있을 수 있는데, 이를 **Particle Degeneracy, 입자 붕괴**라 부르며, 이를 해결하기 위해 *1) 샘플 가중치의 분산이 커질 때만 샘플링을 하는 필터 리샘플링, 2) 리샘플링 후 노이즈 추가, 3) $q(z)$를 mixture distribution으로 설정*하는 등의 방법이 있음.

