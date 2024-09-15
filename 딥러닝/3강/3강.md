## CHAPTER 3. Standard Distributions
관측치의 밀도추정 (density estimation)은 모델링 시 중요한 문제
적절한 분포를 고르기 위해 분포들을 배울 필요가 있음
## Discrete Variables
### Bernoulli distribution
$$\mathrm{Bern}(x|\mu)=\mu^x(1-\mu)^{1-x}$$

코인 토스와 같은 상황 ($\text{random variable } x\in\{0,1\}$)

- 파라미터 : $\mu = p(x=1|\mu), \ 0\leq \mu \leq 1$
  
- 평균, 분산 
  $$\begin{array}{rcl}\mathbb{E}[x]&=&\mu\\\operatorname{var}[x]&=&\mu(1-\mu)\end{array}$$
  
- MLE
  $p(\mathcal{D}|\mu)=\prod_{n=1}^Np(x_n|\mu)=\prod_{n=1}^N\mu^{x_n}(1-\mu)^{1-x_n}$
  $\ln p(\mathcal{D}|\mu)=\sum_{n=1}^N\ln p(x_n|\mu)=\sum_{n=1}^N\left\{x_n\ln\mu+(1-x_n)\ln(1-\mu)\right\}$
  $$\mu_{\mathrm{ML}}=\frac1N\sum_{n=1}^Nx_n$$
 
### Binomial distribution
$$\mathrm{Bin}(m|N,\mu)=\binom Nm\mu^m(1-\mu)^{N-m}$$
$N$번 베르누이 시행에서 $x=1$을 $m$번 뽑을 확률

- 평균, 분산
$$\begin{aligned}
\mathbb{E}[m]& \equiv\sum_{m=0}^Nm\operatorname{Bin}(m|N,\mu)=N\mu  \\
\mathrm{var}[m]& \equiv\sum_{m=0}^N\left(m-\mathbb{E}[m]\right)^2\mathrm{Bin}(m|N,\mu)=N\mu(1-\mu) 
\end{aligned}$$

### Multinomial distribution
$$\mathrm{Mult}(m_1,m_2,\ldots,m_K|\boldsymbol{\mu},N)=\binom N{m_1m_2\ldots m_K}\prod_{k=1}^K\mu_k^{m_k}$$
이산분포의 확장. 선택지가 2개가 아닌 $K$개이며, 각각 $\mu_1,\mu_2,\ldots,\mu_K$확률에 따라 $m_1,m_2,\ldots,m_K$번 뽑아 총 $N$번 뽑을 확률

## The Multivariate Gaussian
$$\mathcal{N}(\mathbf{x}|\boldsymbol{\mu},\boldsymbol{\Sigma})=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\boldsymbol{\Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}$$
정규분포의 $D$차원 확장. $\mu$는 $D$차원 평균 벡터, $\Sigma$는 $D \times D$ 공분산 행렬

단변량 실수 변수에 대해 엔트로피를 최대화 하는 분포는 정규분포 >> 다변량 정규분포에도 똑같이 적용됨
### central limit theorem
**i.i.d.** 한 random sample들의 **합**은, 그 수가 **커질수록** **정규분포**에 가까워진다.

### Geometry of the Gaussian
![[스크린샷 2024-09-14 오후 3.59.16.png]]
정규분포는 $\mathbf{x}$에서 $\mathbf{y}$로의 축 변환과 같다.
$y_i=\mathbf{u}_i^\mathrm{T}(\mathbf{x-\mu})$ << $\mathbf{u}_i$는 공분산 행렬에 대한 고유벡터 (공분산 행렬은 대칭행렬로, 고윳값 항상 실수, 고유벡터들은 orthonormal하게 잡을 수 있음)
$\mathbf{y}=\mathbf{U}(\mathbf{x-\mu})$ << 벡터로의 확장
모든 고윳값이 양수라면 위 그림처럼 타원으로 표현 가능 (공분산이 positive definite)

$$|\boldsymbol{\Sigma}|^{1/2}=\prod_{j=1}^D\lambda_j^{1/2}$$
공분산 행렬의 행렬식은 위와 같이 고윳값들의 곱으로 표현 가능
**자세한 유도 내용은 책 71페이지부터 참고**

### Moments
평균과 분산 유도 과정 (**책 참고**)
### Conditional distribution
$$\begin{aligned}&\boldsymbol{\mu}_{a|b}&&=\quad\boldsymbol{\mu}_a+\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b-\boldsymbol{\mu}_b)\\&\boldsymbol{\Sigma}_{a|b}&&=\quad\boldsymbol{\Sigma}_{aa}-\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}.\end{aligned}$$

$\mathbf{x}_b$를 상수 취급하고 $\mathbf{x}_a$에 대해 가우시안 꼴로 만들면 됨

**유도과정 책 참고**

### Marginal distribution
$$p(\mathbf{x}_a)=\int p(\mathbf{x}_a,\mathbf{x}_b)\operatorname{d}\mathbf{x}_b$$
$\mathbf{x}_b$에 대해서 묶고, 적분하면 $\mathbf{x}_a$만 남게 됨

$$\begin{array}{rcl}\mathbb{E}[\mathbf{x}_a]&=&\boldsymbol{\mu}_a\\\operatorname{cov}[\mathbf{x}_a]&=&\boldsymbol{\Sigma}_{aa}\end{array}$$

**유도과정 책 참고**
### Bayes' theorem
조건부확률과 주변확률분포의 수식을 정확히 유도할 수 있기 때문에 **Bayes' thm** 적용 가능
### Maximum likelihood
- log likelihood
$$\ln p(\mathbf{X}|\boldsymbol{\mu},\boldsymbol{\Sigma})=-\frac{ND}2\ln(2\pi)-\frac N2\ln|\boldsymbol{\Sigma}|-\frac12\sum_{n=1}^N(\mathbf{x}_n-\boldsymbol{\mu})^\mathrm{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu})$$
- $\boldsymbol{\mu}_{\mathrm{ML}}$
$$\frac\partial{\partial\boldsymbol{\mu}}\ln p(\mathbf{X}|\boldsymbol{\mu},\boldsymbol{\Sigma})=\sum_{n=1}^N\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu})$$
$$\boldsymbol{\mu}_{\mathrm{ML}}=\frac1N\sum_{n=1}^N\mathbf{x}_n$$
- $\mathbf{\Sigma}_{\mathrm{ML}}$ 
$$\mathbf{\Sigma}_{\mathrm{ML}}=\frac1N\sum_{n=1}^N(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^\mathrm{T}$$
$$\begin{array}{rcl}\mathbb{E}[\boldsymbol{\mu}_\mathrm{ML}]&=&\boldsymbol{\mu}\\\mathbb{E}[\boldsymbol{\Sigma}_\mathrm{ML}]&=&\frac{N-1}{N}\boldsymbol{\Sigma}\end{array}$$
$\mathbf{\Sigma}_{\mathrm{ML}}$의 상수를 $N-1$로 맞출경우 불편추정량이 됨

### Sequential estimation
?
### Mixtures of Gaussians
정규분포를 여러 개 혼합해서 데이터의 복잡한 분포를 잡아낼 수 있음
![[스크린샷 2024-09-15 오전 8.45.16.png]]

- K개의 가우시안 확률밀도함수 (mixture of Gaussians)
$$p(\mathbf{x})=\sum_{k=1}^K\pi_k\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$$
$\pi_k$ 는 mixing coefficients. 0~1
MLE를 통해 파라미터들을 추정. EM으로도 가능
![[스크린샷 2024-09-15 오전 8.49.58.png]]

참고하면 좋은 자료 : https://untitledtblog.tistory.com/133
## Periodic Variables
반복되는 데이터 (calender, 바람의 방향)은 polar coordinate로 표현 가능
### Von Mises distribution
각에 대한 분포를 고려 $p(\theta)$
$$\begin{array}{rcl}p(\theta)&\geqslant&0\\\int_0^{2\pi}p(\theta)\operatorname{d}\theta&=&1\\p(\theta+2\pi)&=&p(\theta)\end{array}$$

정규분포를 polar coordinate로 바꾸어 생각

- von Mises distibution (circular normal)

$$p(\theta|\theta_0,m)=\frac1{2\pi I_0(m)}\exp\left\{m\cos(\theta-\theta_0)\right\}$$
$$I_0(m)=\frac1{2\pi}\int_0^{2\pi}\exp\left\{m\cos\theta\right\}\mathrm{~d}\theta $$
똑같이 각에 대해 MLE 가능


## The Exponential Family
$$p(\mathbf{x}|\boldsymbol{\eta})=h(\mathbf{x})g(\boldsymbol{\eta})\exp\left\{\boldsymbol{\eta}^{\mathrm{T}}\mathbf{u}(\mathbf{x})\right\}$$
파라이터 $\eta$(natural parameters)가 주어졌을 때, $\mathbf{x}$의 exponential family 분포
$\mathbf{u}$는 $\mathbf{x}$에 대한 함수이며, $g(\boldsymbol{\eta})$는 normalize하기 위한 함수(적분시 1 만들기).
위와 같이 표현될 수 있는 분포를 Exponential Family라고 함 (정규분포, 지수분포, 감마분포 등)

- 베르누이 분포
$$p(x|\mu)=\text{Bern}(x|\mu)=\mu^x(1-\mu)^{1-x}$$
$$\begin{aligned}p(x|\mu)&=\quad\exp\left\{x\ln\mu+(1-x)\ln(1-\mu)\right\}\\&=\quad(1-\mu)\exp\left\{\ln\left(\frac\mu{1-\mu}\right)x\right\}.\end{aligned}$$
$\eta=\ln\left(\frac\mu{1-\mu}\right)$이며 $\mu = \sigma(\eta)=\frac1{1+\exp(-\eta)}$로, 시그모이드 함수임을 알 수 있음.
그럼 다음과 같이 표현 가능 $p(x|\eta)=\sigma(-\eta)\exp(\eta x)$

**책에 다항 분포에 대한 예시도 나와있음**
### Sufficient statistics
표본이 가지고 있는 모수에 대한 모든 정보를 포함하고 있는 통계량
데이터를 모두 가지고 있을 필요가 없어서 효율적
ex. 정규분포에서는 T(X) = (∑x, ∑x²)가 모수에 대한 충분통계량 (두개만으로 평균, 분산 구할 수 있으므로)

## Nonparametric Methods
위에 봤던 방법들은 모두 모수적(parametic) 접근법
분포를 잘못 택했을 경우, 나쁜 예측 성능이 나오는 한계가 존재
### Histograms
![[스크린샷 2024-09-15 오전 9.19.37.png]]
$$p_i=\frac{n_i}{N\Delta_i}$$
합하면 1. 넓이(Bin, $\Delta$)를 어떻게 잡느냐에 따라 모양이 달라짐
- 장점
  데이터 가시화에 좋음
- 단점
  적절한 넓이를 택해야함
  밀도가 불연속적
  $D$차원을 $M$ 넓이로 나눌 경우, $M^D$ 만큼의 계산이 필요 (지수연산)
### Kernel densities
이해가 잘 안가서 링크로 대체
https://norman3.github.io/prml/docs/chapter02/5.html
### Nearest-neighbours
https://norman3.github.io/prml/docs/chapter02/5.html