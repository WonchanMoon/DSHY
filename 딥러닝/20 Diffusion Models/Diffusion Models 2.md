[[Diffusion Models 1]] 정리
우리는 Forward Process, Encoder, $q(\mathbf{z}_{t}\mid\mathbf{z}_{t-1})$에 대해 정의했고,
이거의 reverse인 $q(\mathbf{z}_{t-1}\mid\mathbf{z}_{t})$에 대해서 생각해보려 하였으나, 이건 불가능했음. 왜? $p(\mathbf{x})$를 모르기 때문.
그래서 우리는 무엇을 했다? 
$$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) = \mathcal{N} \left( \mathbf{z}_{t-1} \mid \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t), \, \sigma_t^2 \mathbf{I} \right)
$$
를 구했고,
이때,
$$\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) = \frac{(1 - \alpha_{t-1}) \sqrt{1 - \beta_t} \, \mathbf{z}_t + \sqrt{\alpha_{t-1}} \, \beta_t \, \mathbf{x}}{1 - \alpha_t}
$$
그리고,
$$\sigma_t^2 = \frac{\beta_t (1 - \alpha_{t-1})}{1 - \alpha_t}$$
$\alpha_{t}$는 이렇게 생겼었음. 
$$\alpha_{t} = \prod^{t}_{\tau=1}{1-\beta_\tau}$$

# 20.2 Reverse Decoder
우리가 앞서 정리했듯이, $q(\mathbf{z}_{t-1}\mid\mathbf{z}_{t})$는 알 수가 없음.
그래서 우리는 $q(\mathbf{z}_{t-1}\mid \mathbf{z}_{t}, \mathbf{x})$를 대신 구했음. 이 분포는 알 수 있지만, $\mathbf{x}$를 알아야 알 수 있음.
따라서 샘플을 생성하는 과정, $\mathbf{z}_{T}$로부터 $\mathbf{z}_{0}$까지 되돌아가는 과정은 불가능함. 왜냐하면 $\mathbf{x}$가 없는 분포를 알아야하기 때문.
따라서 우리는 신경망으로 정의된 근사 분포 $p(\mathbf{z}_{t-1}\mid\mathbf{z}_{t}, \mathbf{w})$를 학습하게 됨.
이 분포를 학습하고나면, 우린 $\mathcal{N}(\mathbf{z}_{T}\mid\mathbf{0}, \mathbf{I})$에서 샘플링을 시작해서, $p(\mathbf{x})$로 변환할 수 있게 된다.

### Reverse Process
우리는 Reverse Process를 Gaussian으로 디자인한다.
$$p(\mathbf{z}_{t-1}\mid\mathbf{z}_{t},\mathbf{w})=\mathcal{N}(\mathbf{z}_{t-1}\mid\mathbf{\mu}(\mathbf{z}_{t},\mathbf{w},t),\beta_{t}\mathrm{I})$$

그리고 이때 $\mathbf{\mu}(\mathbf{z}_{t},\mathbf{w},t)$는 신경망으로 예측한다.
그리고 이 DNN은 $t$를 입력으로 받기 때문에, Forward Process에서 Step마다 다르게 주었던 분산 $\beta_{t}$를 컨트롤 할 수 있게 한다. 또한 이는 하나의 네트워크로, 전체 체인의 모든 단계를 학습할 수 있게 해준다.

#### Reverse Process의 전체 확률
$$
p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T \mid \mathbf{w}) = p(\mathbf{z}_T) 
\left\{
\prod_{t=2}^{T} p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})
\right\}
p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}).
$$
여기서 $p(\mathbf{z}_{T})$는 $q(\mathbf{z}_{T})$와 동일하고, $\mathcal{N}(\mathrm{0},\mathrm{I})$으로 주어진다.

#### 새로운 이미지 생성 과정
> 성준이형의 저번 주 궁금증은, *Conditional Diffusion Model* 임.

모델이 학습되고 나서, 
$p(\mathbf{z}_{T})$에서 샘플링을 하고,
$p(\mathbf{z}_{t-1}\mid\mathbf{z}_{t},\mathbf{w})$에서 샘플링을 반복하고,
마지막으로 $p(\mathbf{x}\mid\mathbf{z}_{1},\mathbf{w})$에서 샘플링을 하여 data space의 샘플 $\mathbf{x}$를 뽑는다.

## 20.2.1 Training the decoder
우리는 이제 신경망을 학습시킬 objective function을 만들어야한다.

#### Likelihood function
$$
p(\mathbf{x} \mid \mathbf{w}) = \int \cdots \int p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T \mid \mathbf{w}) \, \mathrm{d}\mathbf{z}_1 \cdots \mathrm{d}\mathbf{z}_T
$$
이때 $p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T \mid \mathbf{w})$는 위에서 정의된 대로,
$$p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T \mid \mathbf{w}) = p(\mathbf{z}_T) 
\left\{
\prod_{t=2}^{T} p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})
\right\}
p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}).$$
이다.

하지만 likelihood function의 적분은 **계산이 불가능**하다.

## 20.2.2 Evidence lower bound
Likelihood function이 계산이 불가능했기에,
VAE에서의 아이디어처럼 *evidence lower bound*  **(ELBO)** 를 이용할 것임.

임의의 분포 $q(\mathbf{z})$에 대해, 아래 식이 성립한다.
$$
\ln p(\mathbf{x} \mid \mathbf{w}) = \mathcal{L}(\mathbf{w}) + \mathrm{KL} \left( q(\mathbf{z}) \,\|\, p(\mathbf{z} \mid \mathbf{x}, \mathbf{w}) \right)
$$
$\mathcal{L}$ : evidence lower bound (ELBO), variational lower bound
$$\mathcal{L}(\mathbf{w}) = \int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{x} \mid \mathbf{z}, \mathbf{w}) p(\mathbf{z})}{q(\mathbf{z})} \right\} \, \mathrm{d}\mathbf{z}$$
$\mathrm{KL}$: Kullback-Leibler divergence	
$$
\mathrm{KL} \left( q(\mathbf{z}) \,\|\, p(\mathbf{z} \mid \mathbf{x}, \mathbf{w}) \right) = - \int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{z} \mid \mathbf{x}, \mathbf{w})}{q(\mathbf{z})} \right\} \, \mathrm{d}\mathbf{z}$$
KL divergence는 항상 0 이상이므로,
$$\ln p(\mathbf{x} \mid \mathbf{w}) \geq \mathcal{L}(\mathbf{w})$$

**log likelihood function이 계산이 불가능하기에,**
**우리는 $\mathcal{L}(\mathbf{w})$를 최대화하여 신경망을 학습할 것이다.**

#### 학습을 하기 위해서는, ...
우리는 먼저 디퓨전 모델의 lower bound의 explicit form을 유도해야한다.

lower bound를 정의할 때, 우리는 $q(\mathbf{z})$를 아무 확률분포나 갖다 쓸 수 있다.
**i.e.) $q(\mathbf{z})$는 음수가 아니고, 적분해서 1이면 된다.**

따라서 우리는 이를 활용하여, 입력 $\mathbf{x}$에 따라 $q(\mathbf{z})$를 선택할 것이다.
그리고 앞서 정의했던 $q(\mathbf{z})$와 $p(\mathbf{z})$를 이용하면,
$$q(\mathbf{z}_1, \ldots, \mathbf{z}_t \mid \mathbf{x}) = q(\mathbf{z}_1 \mid \mathbf{x}) \prod_{\tau=2}^{t} q(\mathbf{z}_\tau \mid \mathbf{z}_{\tau-1}).
$$
$$p(\mathbf{x}, \mathbf{z}_1, \ldots, \mathbf{z}_T \mid \mathbf{w}) = p(\mathbf{z}_T) 
\left\{
\prod_{t=2}^{T} p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})
\right\}
p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}).$$
를 활용하면, ELBO를 다음과 같이 쓸 수 있다.
$$\begin{align*}
\mathcal{L}(\mathbf{w})
& = \int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{x} \mid \mathbf{z}, \mathbf{w}) p(\mathbf{z})}{q(\mathbf{z})} \right\} \, \mathrm{d}\mathbf{z}\\
&= \mathbb{E}_{q} \left[ \ln \left\{ \frac{p(\mathbf{x} \mid \mathbf{z}, \mathbf{w}) p(\mathbf{z})}{q(\mathbf{z})}\right\} \right] \\
&= \mathbb{E}_{q} \left[ \ln \left( 
\frac{
p(\mathbf{z}_T) \left\{ \prod_{t=2}^{T} p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w}) \right\} p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w})
}{
q(\mathbf{z}_1 \mid \mathbf{x}) \prod_{t=2}^{T} q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x})
}
\right) \right] \\
&= \mathbb{E}_{q} \left[ 
\ln p(\mathbf{z}_T) 
+ \sum_{t=2}^{T} \ln \frac{p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x})} 
- \ln q(\mathbf{z}_1 \mid \mathbf{x}) 
+ \ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w})
\right]
\end{align*}
$$

그리고 이때,
$$
\mathbb{E}_{q}[\cdot] \equiv \int \cdots \int q(\mathbf{z}_1 \mid \mathbf{x}) \prod_{t=2}^{T} q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \, [\cdot] \, \mathrm{d}\mathbf{z}_1 \cdots \mathrm{d}\mathbf{z}_T.
$$
이다.

#### ELBO 식 뜯어보기 (1, 3, 4번째 항)
**첫번째 항** $\ln p(\mathbf{z}_{T})$는 그냥 고정된 정규분포 $\mathcal{N}(\mathbf{z}_{T}\mid\mathrm{0},\mathrm{I})$ 이다.
또한 **세번재 항** $\ln q(\mathbf{z}_{1}\mid\mathbf{x})$ 역시 $\mathbf{w}$를 갖고 있지 않으므로, 두 항은 ELBO에서 빠질 수 있다.

**네번째 항** $\ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w})$는 VAE에서처럼 몬테카를로로 근사.
$$
\mathbb{E}_{q} \left[ \ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}) \right] \simeq \sum_{l=1}^{L} \ln p(\mathbf{x} \mid \mathbf{z}_1^{(l)}, \mathbf{w})
$$
이때 $\mathbf{z}_1^{(l)} \sim \mathcal{N} ( \mathbf{z}_1 \mid \sqrt{1 - \beta_1} \, \mathbf{x}, \, \beta_1 \mathbf{I}).$
하지만 Diffusion model에서는, $q$ 분포는 고정하기 때문에 학습할 필요가 없고, 따라서 VAE와 달리 reparametrization trick이 필요하지 않음.

#### ELBO 식 뜯어보기 (2번째 항)
이제 **두번째 항**만 남았음.
하지만 이 두번째 항을 계산하기 위해서는, 
1. $\mathbf{z}_{t-1}\sim q(\mathbf{z}_{t-1}\mid\mathbf{x})$ 에서 샘플을 뽑고,
2. $\mathbf{z}_{t}\sim q(\mathbf{z}_{t}\mid\mathbf{z}_{t-1})$ 에서 또 샘플을 뽑아서
계산을 해야함.

하지만 이 방법은 분산이 크고, **노이즈가 심한 추정치**를 만들기 때문에,
정확한 추정을 위해서는 너무 많은 샘플이 필요함.
-> 하나의 샘플만으로도 추정할 수 있도록 ELBO를 다시 작성.
> 흠 이게 가능한가요?

## 20.2.3 Rewriting the ELBO
> 목표 : ELBO를 KL Divergence를 이용하여 작성

#### ELBO의 두 번째 항
$$\sum_{t=2}^{T} \ln \frac{p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x})} $$
여기서 $q(\mathbf{z}_{t}\mid\mathbf{z}_{t-1},\mathbf{x})$를 다음과 같이 분해하여 식에 대입하면,
$$
q(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x}) = \frac{q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \, q(\mathbf{z}_t \mid \mathbf{x})}{q(\mathbf{z}_{t-1} \mid \mathbf{x})}.
$$
아래처럼 시그마 내부의 항을 바꿔 쓸 수 있고,
$$
\ln \frac{p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_{t} \mid \mathbf{z}_{t-1}, \mathbf{x})} 
= \ln \frac{p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_{t-1} \mid \mathbf{z}_{t}, \mathbf{x})} + \ln \frac{q(\mathbf{z}_{t-1} \mid \mathbf{x})}{q(\mathbf{z}_{t} \mid \mathbf{x})}.
$$

결국 ELBO는 아래와 같이 된다.
$$
\mathcal{L}(\mathbf{w}) = \mathbb{E}_{q} \left[
\sum_{t=2}^{T} \ln \frac{p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})}{q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})}
+ \ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w})
\right].
$$
 마지막으로 다시 쓰면,
 $$
\mathcal{L}(\mathbf{w}) =
\underbrace{
\int q(\mathbf{z}_1 \mid \mathbf{x}) \ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}) \, \mathrm{d}\mathbf{z}_1
}_{\text{reconstruction term}}
-
\underbrace{
\sum_{t=2}^{T} \int \mathrm{KL}\left(
q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \,\|\, p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})
\right) q(\mathbf{z}_t \mid \mathbf{x}) \, \mathrm{d}\mathbf{z}_t
}_{\text{consistency terms}}
$$

위의 ELBO 식에서,
**reconstruction term은 샘플링으로 근사**가 가능하고,
**consistency term은 두 정규분포의 KL Divergence**이다.

#### consistency term
$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$와 $p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w})$를 아래와 같이 풀어서 KL Divergence를 다시 계산할 수 있다.
$$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) = \mathcal{N} \left( \mathbf{z}_{t-1} \mid \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t), \, \sigma_t^2 \mathbf{I} \right)$$
$$p(\mathbf{z}_{t-1}\mid\mathbf{z}_{t},\mathbf{w})=\mathcal{N}(\mathbf{z}_{t-1}\mid\mathbf{\mu}(\mathbf{z}_{t},\mathbf{w},t),\beta_{t}\mathrm{I})$$
> 이 식들은 앞서서 다 유도하였음.

$$
\mathrm{KL}\left( q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \,\|\, p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w}) \right)
= \frac{1}{2 \beta_t} \left\| \mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) - \boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t) \right\|^2 + \text{const}
$$
우리는 ELBO가 최대가 되도록 학습을 하므로,
**KL Divergence는 최소**가 되도록 학습한다.

## 20.2.4 Predicting the noise
[[Diffusion Models 1#diffusion kernel|Diffusion kernel]]에서,
$$\mathbf{z}_{t}=\sqrt{\alpha_{t}}\mathbf{x}+\sqrt{1-\alpha_{t}}\boldsymbol{\epsilon}_{t}$$
임을 알고 있다. 이 식을 살짝 변형하면,
$$
\mathbf{x} = \frac{1}{\sqrt{\alpha_t}} \mathbf{z}_t - \frac{\sqrt{1 - \alpha_t}}{\sqrt{\alpha_t}} \boldsymbol{\epsilon}_t
$$
이다.

이를 이용하여, 우리는 $\mathrm{m}_{t}(\mathbf{x},\mathbf{z}_{t})$를 아래와 같이 다시 쓸 수 있다.
$$
\mathbf{m}_t(\mathbf{x}, \mathbf{z}_t) = \frac{1}{\sqrt{1 - \beta_t}} \left\{ \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \boldsymbol{\epsilon}_t \right\}
$$
마찬가지로, $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$도 다시 쓰면,
$$
\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t) = \frac{1}{\sqrt{1 - \beta_t}} \left\{ \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \mathbf{g}(\mathbf{z}_t, \mathbf{w}, t) \right\}
$$
> 여기서 $\mathbf{g}(\mathbf{z}_t, \mathbf{w}, t)$는, 
> $\mathbf{z}_{t}$를 생성하기 위해 $\mathbf{x}$에 더해진 총 노이즈를 예측하는 신경망.
#### 최종적으로 KL Divergence(Consistency Term)는,
$$\begin{align*}
\mathrm{KL}\left( q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \,\|\, p(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{w}) \right)
&= \frac{\beta_t}{2(1 - \alpha_t)(1 - \beta_t)} \left\| \mathbf{g}(\mathbf{z}_t, \mathbf{w}, t) - \boldsymbol{\epsilon}_t \right\|^2 + \text{const} \\
&= \frac{\beta_t}{2(1 - \alpha_t)(1 - \beta_t)} \left\| \mathbf{g}\left( \sqrt{\alpha_t} \mathbf{x} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_t, \mathbf{w}, t \right) - \boldsymbol{\epsilon}_t \right\|^2 + \text{const}
\end{align*}
$$
이다.
Ho, Jain, and Abbeel (2020)은 ${\beta_t}/{2(1 - \alpha_t)(1 - \beta_t)}$를 제거할 경우 성능이 더 오르는 것을 경험적으로 확인하였다.

#### Reconstruction Term
ELBO의 Reconstruction Term은 아래와 같이 근사가 가능하다.
$$\ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}) = - \frac{1}{2 \beta_1} \left\| \mathbf{x} - \boldsymbol{\mu}(\mathbf{z}_1, \mathbf{w}, 1) \right\|^2 + \text{const.}$$

이때 **(1). $\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t)$를 아래와 같이 바꿔 쓸 수 있다는 것**과,
$$
\boldsymbol{\mu}(\mathbf{z}_t, \mathbf{w}, t) = \frac{1}{\sqrt{1 - \beta_t}} \left\{ \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \mathbf{g}(\mathbf{z}_t, \mathbf{w}, t) \right\}
$$
**(2). $\alpha_{1}=(1-\beta_{1})$임을 이용하면,** 
아래와 같은 사실을 확인할 수 있다.
$$
\ln p(\mathbf{x} \mid \mathbf{z}_1, \mathbf{w}) = -\frac{1}{2(1 - \beta_t)} \left\| \mathbf{g}(\mathbf{z}_1, \mathbf{w}, 1) - \boldsymbol{\epsilon}_1 \right\|^2 + \text{const.}
$$
또한 이 식은 결국 Consistency Term에서 $t=1$인 경우와 같으므로, **Reconstruction Term과 Consistency Term은 하나로 합쳐질 수 있다.**

#### 최종 Objective Function은,
$$
\mathcal{L}(\mathbf{w}) = - \sum_{t=1}^{T} \left\| \mathbf{g}\left( \sqrt{\alpha_t} \mathbf{x} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_t, \mathbf{w}, t \right) - \boldsymbol{\epsilon}_t \right\|^2.
$$
Loss function은, 예측된 노이즈와 실제 노이즈 간의 제곱 차이다.

#### 학습 과정
1. 데이터 셋에서 $\mathbf{x}$ 하나 무작위로 뽑기 (예: 고양이 사진)
2. $t\in\{1,...,T\}$ 하나 무작위로 뽑기 (예: $t$ = 432)
3. 노이즈 $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{\epsilon}\mid\mathrm{0},\mathrm{I})$ 하나 무작위로 뽑기
4. $\mathbf{x}$에 노이즈 추가하여 $\mathbf{z}_t$ 만들기
5. loss 함수 계산
6. 그걸 기반으로 파라미터 업데이트
![[Algorithm 20.1.png]]

## 20.2.5 Generating new samples
네트워크의 학습이 끝나면, 우리는 가우시안 분포 $p(\mathbf{z}_{T})$에서 샘플링을 하고, 노이즈를 벗겨가면서 새로운 샘플을 만들어 낼 수 있다.
#### 학습이 끝난 이후 샘플링 과정
![[Algorithm 20.2.png]]
