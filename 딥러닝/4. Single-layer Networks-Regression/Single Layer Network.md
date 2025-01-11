---
프로젝트: 딥러닝
날짜: 2024-09-21
완료: false
담당: 
tags:
  - DSHY
---

## Linear Regression
 - 회귀의 목적은 입력 변수의 D차원 벡터 x의 값을 바탕으로 하나 이상의 연속형 대상 변수 t의 값을 예측하는 것
$$y(\mathbf{x},\mathbf{w})=w_0+w_1x_1+\ldots+w_Dx_D$$
- 기본 회귀 모델을 다음과같이 신경망 표현으로 변환할 수 있다.
$$y(\mathbf{x},\mathbf{w})=w_0+\sum_{j=1}^{M-1}w_j\phi_j(\mathbf{x})$$
- 행렬 표기법을 사용하면 다음과 같이 표현할 수 있다.
$$y(\mathbf{x},\mathbf{w})=w^T\phi(\mathbf{x})$$
- 여기서 $\textsf{w}=(w_{0},\ldots,w_{M-1})^{\mathrm{T}}\mathrm{~and~}\phi=(\phi_{0},\ldots,\phi_{M-1})^{\mathrm{T}}$.

- 딥러닝 이전
	- 기계 학습에서 입력 변수를 사전 처리하기 위해 기저 함수들을 사용하여 특징 추출
	-  복잡한 작업을 위해 적절한 기저 함수를 찾는 것이 어려웠습니다.
- Gaussian 기저 함수
	- 가우시안 함수는 다음과 같이 표현$$\phi_j(x)=\exp\left(-\frac{(x-\mu_j)^2}{2s^2}\right)$$
	- $\mu_{j}$는 함수의 위치, s는 함수의 범위를 조절
	- 평균과 분산을 조절하여 함수의 모양을 조절하는 느낌
- 시그모이드 기저 함수
	- 대표적인 비선형 변환 함수
	- 다음과 같은 형태로 표현$$\sigma(a)=\frac1{1+\exp(-a)}$$
	- 출력 범위가 0과 1사이에 있어서 확률을 나타내는데 사용
	- tanh 함수와 같은 다른 비선형화 함수도 사용가능
-  딥러닝은 데이터를 통해 비선형 변환을 자 동으로 학습하여, 기저 함수를 직접 설계할 필요가 없다.

## Likelihood Function
- 일반적인 선형회귀 식을 다음과 같이도 표현 가능 $$t=y(\mathbf{x},\mathbf{w})+\epsilon$$
- 또한 이렇게도 적을수 이씀 $$p(t|(\mathbf{x},\mathbf{w}),\sigma^{2})=\mathcal{N}(t|y(\mathbf{x},\mathbf{w}),\sigma^{2})$$
- $\epsilon$이 정규분포를 따르기 때문에, t도 정규분포를 따른다.
- 여기서 이제 data set을 통해 likelihood function을 구할 수 있다.
- likelihood function은 다음과 같이 표현 가능$$p(\mathbf{t}|\mathbf{X},\mathbf{w},\sigma^{2})=\prod_{n=1}^{N}\mathcal{N}(t_n|\mathbf{w}^T\phi(\mathbf{x}_n),\sigma^{2})$$
- 양변에 log를 취하면 다음과 같다. $$\ln p(\mathbf{t}|\mathbf{w},\sigma^{2})=\sum_{n=1}^{N}\ln\mathcal{N}(t_n|\mathbf{w}^{T}\phi(\mathbf{x}_n),\sigma^{2})= -\frac N2\ln\sigma^2-\frac N2\ln(2\pi)-\frac1{\sigma^2}E_D(\mathbf{w})$$
- 정리하면 다음과 같다. $$E_D(\mathbf{w})=\frac12\sum_{n=1}^{N}\{t_n-\mathbf{w}^T\phi(\mathbf{x}_n)\}^2$$
- 이를 최소화하는 $\mathbf{w}$를 찾는 것이 목표
- 이를 위해 미분을 통해 최소화하는 $\mathbf{w}$를 찾는다.
$$\nabla_\mathbf{w}\ln p(\mathbf{t}|\mathbf{X},\mathbf{w},\sigma^2)=\frac{1}{\sigma^2}\sum_{n=1}^N\left\{t_n-\mathbf{w}^\mathrm{T}\phi(\mathbf{x}_n)\right\}\phi(\mathbf{x}_n)^\mathrm{T}$$
- 0일때 최소값을 찾는다. 
- $$0=\sum_{n=1}^Nt_n\phi(\mathbf{x}_n)^\mathrm{T}-\mathbf{w}^\mathrm{T}\left(\sum_{n=1}^N\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^\mathrm{T}\right)$$
- $$\mathbf{w}_{ML}=(\Phi^\mathrm{T}\Phi)^{-1}\Phi^\mathrm{T}\mathbf{t}$$
- $\mathbf{\Phi}^\dagger\equiv\left(\mathbf{\Phi}^\mathrm{T}\mathbf{\Phi}\right)^{-1}\mathbf{\Phi}^\mathrm{T}$로 표현 가능 하고, 이를 무어-펜로즈 유사 역행렬이라고 한다.
- 무어-펜로즈 유사 역행렬은 $\Phi^{\dagger}$ 은 $\Phi^{-1}$ 와 같은 역할을 하는 것으로 볼 수 있다 (?)
$$E_D(\mathbf{w})=\dfrac{1}{2}\sum_{n=1}^N\{t_n-w_0-\sum_{j=1}^{M-1}w_j\phi_j(\mathbf{x}_n)\}^2$$

$$w_0=\overline{t}-\sum_{j=1}^{M-1}w_j\overline{\phi_j}$$

$$\bar{t}=\frac{1}{N}\sum_{n=1}^{N}t_{n},\quad\overline{\phi_{j}}=\frac{1}{N}\sum_{n=1}^{N}\phi_{j}(\mathbf{x}_{n})$$
- 최종적으로 다음과 같은 결론을 얻을 수 있다.
$$\sigma_{\mathrm{ML}}^{2}=\frac{1}{N}\sum_{n=1}^{N}\{t_{n}-\mathbf{w}_{\mathrm{ML}}^{\mathrm{T}}\phi(\mathbf{x}_{n})\}^{2}
$$

## Geometry of Least Squares
- 솔직히 이해를 못해서 궁금하신분은 읽어보시길
- $\Phi(\Phi^\mathrm{T}\Phi)^{-1}\Phi^\mathrm{T}$ -> projection matrix로 볼수 있다.
- $\Phi\mathbf{w}_{ML}$은 $\mathbf{t}$에 가장 가까운 벡터이다.
## Sequential Learning
- 데이터가 많아지면 계산량이 많음
- 데이터를 나누어 학습하면 계산량을 줄일 수 있음
- 이를 위해 데이터를 나누어 순차적으로 학습하는 방법을 사용
$$\mathbf w^{(\tau+1)}=\mathbf w^{(\tau)}-\eta\nabla E_n$$
$$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta(t_n-\mathbf{w}^{(\tau)\text{T}}\phi_n)\phi_n$$

## Regularized Least Squares
- MLE의 문제점은 overfitting
- 이를 해결하기 위해 정규화(벌점화)를 사용
$$E_D(\mathbf{w})+\lambda E_W(\mathbf{w})$$
- $\lambda$는 정규화의 강도를 조절하는 하이퍼파라미터

$$E_W(\mathbf{w})=\frac{1}{2}\sum_jw_j^2=\frac{1}{2}\mathbf{w}^\mathrm{T}\mathbf{w}$$
결론
$$\mathbf{w}=\begin{pmatrix}\lambda\mathbf{I}+\mathbf{\Phi}^\mathrm{T}\mathbf{\Phi}\end{pmatrix}^{-1}\mathbf{\Phi}^\mathrm{T}\mathbf{t}.$$
- 이를 ridge regression이라고 한다.
## Multiple Outputs
- 다중 출력을 위해 다음과 같이 표현 가능
- $$\mathbf{t}=\mathbf{W}^\mathrm{T}\phi(\mathbf{x})+\mathbf{\epsilon}$$
- 이를 likelihood function에 대입하면 다음과 같다.
- $$p(\mathbf{t}|\mathbf{X},\mathbf{W},\beta)=\prod_{n=1}^{N}\mathcal{N}(\mathbf{t}_n|\mathbf{W}^\mathrm{T}\phi(\mathbf{x}_n),\sigma^{2}\mathbf{I})$$
- 이를 정리하면 multi output에서도 비슷한 결과를얻을 수 있다.
$$\mathbf{W}_{\mathrm{ML}}=\left(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\right)^{-1}\mathbf{\Phi}^{\mathrm{T}}\mathbf{T}$$

## Decision theory



## The Bias-Variance trade-off
- 회귀 모델에서 목표는 입력 데이터 에 대해 출력값을 예측하는 것인데 우리가 만든 모델의 목표가 예측하는 값은 항상 오차를 가진다
- 이 오차는 크게 두가지로 나눌 수 있다.
- Bias : 모델의 예측값과 실제값의 차이
- Variance : 모델의 예측값들의 분산
- 이 두가지는 trade-off 관계에 있다.
- 둘다 작으면 좋겠지만, 둘다 작게 만드는 것은 어렵다.
$$\begin{aligned}&\mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-h(\mathbf{x})\}^{2}\right]\\&=\underbrace{\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}^{2}}_{(\mathbf{bias})^{2}}+\underbrace{\mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x};\mathcal{D})]\}^{2}\right]}_{\text{variance}}.\end{aligned}$$
- 과소적합 (Underfitting) 
	- 상태에서는 Bias가 높고, Variance는 낮다. 모델이 너무 단순해서 패턴을 잡아내지 못한다.
- 과적합 (Overfitting) 
	- 상태에서는 Variance가 높고, Bias는 낮다. 모델이 너무 복잡해서 데이터의 노이즈까지 학습한다.
	- 