생성형 모델의 강력한 방법 중 하나는, latent variable $\mathbf{z}$의 분포 $p(\mathbf{z})$를 학습하고, deep neural network를 이용하여 $\mathbf{z}$를 $\mathbf{x}$로 다시 복원하는 것.
우리는 앞 챕터에서, 이 deep neural network를 정의하고 학습하는 다양한 방법들에 대해서 다룸.
> GAN, VAE, normalizing flows

우리는 이제 네 번째 방법, **diffusion models(Denoising Diffusion Probabilistic Models, DDPMs)** 를 다룸.

diffusion model은 다양한 분야에서 적용이 될 수 있지만, 이 책에서는 이미지 데이터로 내용 설명함.
> 최근에 보니까 diffusion LLM도 있고, 내가 인턴 기간 중 사용한 RFDiffusion도 있음.

### Diffusion Model의 Encoding 과정
![[20.0.1.png]]

# 20.1. Forward Encoder
$\mathbf{x}$ : 훈련 셋에서 가져온 이미지
$\mathbf{x}$의 각 픽셀에 독립적으로 가우시안 노이즈를 추가하여 $\mathbf{z}_{1}$을 만들자.
$$\mathbf{z}_1 = \sqrt{1 - \beta_1} \, \mathbf{x} + \sqrt{\beta_1} \, \boldsymbol{\epsilon}_{1} \quad \text{where } \boldsymbol{\epsilon}_{1}\sim\mathcal{N}(\boldsymbol{\epsilon}_{1}|\mathbf{0}, \mathbf{I}), \beta_{1}<1$$
이때, $\beta_{1}$은 noise distribution의 분산이다. 
이때, $\mathbf{x}$와 $\boldsymbol{\epsilon}_{1}$에 곱해진 $\sqrt{1-\beta_{1}}$과 $\sqrt{\beta_{1}}$은,
1. $\mathbf{z}_{t}$의 평균이 $\mathbf{z}_{t-1}$의 평균보다 0에 더 가깝게 만들어주고,
2. $\mathbf{z}_{t}$의 분산이 $\mathbf{z}_{t-1}$의 분산보다 더 unit matrix에 가깝게 만들어준다.

위 식을 다시 쓰면, 
$$q(\mathbf{z}_1 \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}_1 \mid \sqrt{1 - \beta_1} \, \mathbf{x}, \beta_1 \mathbf{I})
$$
이렇게 쓸 수 있다. $\mathbf{z}_{1}$을 평균이 $\sqrt{1 - \beta_1} \, \mathbf{x}$, 분산이 $\beta_1 \mathbf{I}$인 정규분포에서 $\mathbf{z}_{1}$을 샘플링하는 것으로 이해.

우리는 이렇게 독립적인 가우시안 노이즈를 추가하는 과정을 반복해서, 점점 더 노이즈가 추가 된 이미지들 $\mathbf{z}_{2},...,\mathbf{z}_{T}$ 를 만들어나간다.
> 여기서 노이즈가 추가된 이미지들 $\mathbf{z}_{1},...,\mathbf{z}_{T}$를 *latent variables*라고 하는데,
> 우리가 기존에 이해하고 있던 '데이터의 숨겨진 정보를 담고있다'는 의미와는 약간 다른 듯.
> 여기선 그저 *'노이즈가 추가된 것들'* 정도로 이해하기

그리고 이 경우, $T\rightarrow\infty$이면 $\mathbf{z}_{T}\rightarrow\mathcal{N}(\mathbf{0},\mathbf{I})$

#### Diffusion Model의 도식화
![[20.1.1.png]]

#### Encoding Process의 수식
$$\mathbf{z}_t = \sqrt{1 - \beta_t} \, \mathbf{z}_{t-1} + \sqrt{\beta_t} \, \boldsymbol{\epsilon}_{t}\quad \text{where } \boldsymbol{\epsilon}_{t}\sim\mathcal{N}(\boldsymbol{\epsilon}_{t}\mid\mathbf{0},\mathbf{I})
$$
이렇게 쓸 수도 있고,
$$q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_t \mid \sqrt{1 - \beta_t} \, \mathbf{z}_{t-1}, \beta_t \mathbf{I}).
$$
이렇게 쓸 수도 있다.

이때 $\beta_{t}\in(0,1)$은 사용자가 직접 선택하게 되는데, 
일반적으로 체인 상에서 분산 값이 점점 증가하도록 $\beta_{1}<\beta_{2}<...<\beta_{T}$ 를 만족하도록 선택한다.

## 20.1.1 Diffusion kernel
Forward Process의 전체 확률 모델을 구성하면 아래와 같음.
$$q(\mathbf{z}_1, \ldots, \mathbf{z}_t \mid \mathbf{x}) = q(\mathbf{z}_1 \mid \mathbf{x}) \prod_{\tau=2}^{t} q(\mathbf{z}_\tau \mid \mathbf{z}_{\tau-1}).
$$

우리는 Forward Process에서, 원래는 한 step씩 나아갔지만,
$\mathbf{x}$에서 $\mathbf{z}_t$로 바로 점프 뛸 수 있다면 계산이 훨씬 쉬워지고 빨라짐.

#### diffusion kernel
그래서 $\mathbf{x}$에서 $\mathbf{z}_t$로 바로 점프 뛰는 **diffusion kernel**을 아래와 같이 모델링 할 수 있음.
$$q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}_t \mid \sqrt{\alpha_t} \, \mathbf{x}, (1 - \alpha_t) \mathbf{I})
$$
$\mathbf{z}_{1}$부터 $\mathbf{z}_{t-1}$까지를 marginalize 해버려서 얻어짐.

이때 
$$\alpha_{t}=\prod^{t}_{\tau=1}{1-\beta_\tau}$$
 임.

혹은 
$$\mathbf{z}_{t}=\sqrt{\alpha_{t}}\mathbf{x}+\sqrt{1-\alpha_{t}}\boldsymbol{\epsilon}_{t} \quad \text{where } \boldsymbol{\epsilon}_{t}\sim\mathcal{N}(\boldsymbol{\epsilon}_{t}\mid\mathbf{0},\mathbf{I})$$
처럼 나타낼 수 있음.
## 20.1.2 Conditional distribution
우리의 목표는, noise process를 되돌리는 법을 배우는 것.
즉, $q(\mathbf{z}_{t}\mid\mathbf{z}_{t-1})$의 reverse를 생각할 수 있음.
$$q(\mathbf{z}_{t-1}\mid\mathbf{z}_{t})=\frac{q(\mathbf{z}_{t}\mid\mathbf{z}_{t-1})\ q(\mathbf{z_{t-1}})}{q(\mathbf{z_{t}})}$$
그리고 $q(\mathbf{z}_{t-1})$은, 아래와 같이 marginalize를 통해 쓸 수 있음.
$$q(\mathbf{z}_{t-1})=\int{q(\mathbf{z}_{t-1}\mid\mathbf{x})p(\mathbf{x})}\text{ d}\mathbf{x}$$
하지만 이 적분은 계산할 수 없음. 왜냐하면 $p(\mathbf{x})$를 모르기 때문. 만약 우리가 훈련 데이터셋에서 샘플링을 통해 이 적분을 근사한다면, 분포는 mixture of gaussian이 됨.

대신에, 우리는 다른 버전을 생각할 것임.
바로 $\mathbf{x}$에 의존하는 conditional distribution.
$$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) = \frac{q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x}) \, q(\mathbf{z}_{t-1} \mid \mathbf{x})}{q(\mathbf{z}_t \mid \mathbf{x})}.
$$
#### conditional distribution 하나씩 뜯어보기
그리고 이때, Forward Process의 Markov 성질에 의해,
$$\begin{aligned}
q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x}) &= q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \\
&=\mathcal{N}(\mathbf{z}_t \mid \sqrt{1 - \beta_t} \, \mathbf{z}_{t-1}, \beta_t \mathbf{I})
\end{aligned}$$
이 식은 그리고 $\mathbf{z}_{t-1}$에 대한 이차형식의 지수함수 형태가 된다.
> 이차형식의 지수함수?
> 이차형식 : $\mathbf{x}^{\top}A\mathbf{x}$ 처럼 생긴 벡터-행렬-벡터 곱
> 이차형식의 지수함수 : $\exp(-\mathbf{x}^{\top}A\mathbf{x})$
> 다차원 정규분포는 이차형식의 지수함수를 갖고 있음.

그리고 $q(\mathbf{z}_{t-1} \mid \mathbf{x})$는 diffusion kernel로, 
$$q(\mathbf{z}_{t-1} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}_{t-1} \mid \sqrt{\alpha_{t-1}} \, \mathbf{x}, (1 - \alpha_{t-1}) \mathbf{I})
$$
이므로, 역시 $\mathbf{z}_{t-1}$에 대한 이차형식의 지수함수 형태가 된다.

마지막으로, 분모는 $\mathbf{z}_{t-1}$에 대해 상수이므로, 무시할 수 있다.

따라서 conditional distribution의 전체형태는, **가우시안 분포**가 됨.

#### 완전 제곱식 만들기(Completing the Square),
우리는 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$의 우변을 완전 제곱식을 통해 아래와 같이 작성할 수 있음.
$$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) = \mathcal{N} \left( \mathbf{z}_{t-1} \mid \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t), \, \sigma_t^2 \mathbf{I} \right)
$$

이때,
$$\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) = \frac{(1 - \alpha_{t-1}) \sqrt{1 - \beta_t} \, \mathbf{z}_t + \sqrt{\alpha_{t-1}} \, \beta_t \, \mathbf{x}}{1 - \alpha_t}
$$
그리고,
$$\sigma_t^2 = \frac{\beta_t (1 - \alpha_{t-1})}{1 - \alpha_t}$$
$\alpha_{t}$는 앞서 정의한 것과 같음.
$$\alpha_{t} = \prod^{t}_{\tau=1}{1-\beta_\tau}$$