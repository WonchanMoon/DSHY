# 14.3 Langevin Sampling
> 랑주뱅 샘플링

Metropolis-Hasting 알고리즘은, 
1. proposal distribution이 비교적 simple하고, 
2. random walk 기반이기에
비효율적임.

우리가 신경망을 학습할 때, log-likelihood의 gradient를 이용하여 학습을 했었음.
따라서, **확률분포의 gradient를 이용하는 샘플링 알고리즘**을 소개함.

## 14.3.1 Energy-based models
대부분의 생성형 모델은 조건부 확률로 작성 가능.
$$p(\mathbf{x}|\mathbf{w})$$
이때 $\mathbf{x}$는 데이터, $\mathbf{w}$는 학습 가능한 파라미터.
위 모델은 likelihood를 최대화하며 학습이 가능함.

#### energy function
$$E(\mathbf{x}, \mathbf{w})\ :\ \text{energy function}$$
real-valued function이고, 그 외에는 아무 조건도 없음.

$$\exp\{-E(\mathbf{x},\mathbf{w})\}$$
이렇게 함수를 약간 변형하면, **이 함수는 non-negative**이기 때문에, **$\mathbf{x}$에 대해 아직 정규화되지 않은 확률분포(un-normalized probability distribution with respect to $\mathbf{x}$)** 로 볼 수 있음.

따라서 우리가 적당한 정규화 상수 $Z(\mathbf{w})$를 도입하여 모델(확률분포)을 작성하면,
$$p(\mathbf{x}|\mathbf{w})=\frac{1}{Z(\mathbf{w})}\exp\{-E(\mathbf{x},\mathbf{w}) \}$$
처럼 쓸 수 있음.

이때 $Z(\mathbf{w})$는 **partition function**으로 불리고,$$Z(\mathbf{w})=\int{\exp\{-E(\mathbf{x},\mathbf{w}) \}} \text{d}\mathbf{x}$$ 로 정의 됨.
> $Z(\mathbf{w})$가 partition function이지만 위에서 정규화 상수라고 한 이유는, $\mathbf{x}$에 대해서는 상수이기 때문

그리고 일반적으로, energy function은 $\mathbf{x}$를 input으로 받는 Deep Neural Network로 모델링 함.

#### energy function의 likelihood
$\mathcal{D} = (\mathbf{x}_{1},...,\mathbf{x}_{N})$의 log-likelihood는 아래와 같음.
$$\begin{align}
\ln p(\mathcal{D} | \mathbf{w}) &= \ln ({\prod^{N}_{n=1} p(\mathbf{x_{n}}|\mathbf{w})}) \\
&= - \sum_{n=1}^{N} E(\mathbf{x}_n, \mathbf{w}) - N \ln Z(\mathbf{w})
\end{align}
$$

우리가 log-likelihood의 gradient를 계산하려면, $Z(\mathbf{w})$를 알아야함.
하지만 $E(\mathbf{x},\mathbf{w})$에서 $Z(\mathbf{w})$를 구하는게 어려움.
## 14.3.2 Maximizing the likelihood
partition function을 직접 구하지 않고, 근사를 할 수 있는 여러 방법이 있음.
여기서는 MCMC 기반의 방법을 소개할 것임.

위의 likelihood 식에서, 우리는 gradient를 계산할 수 있음.
$$\nabla_{\mathbf{w}} \ln p(\mathbf{x} | \mathbf{w}) = - \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) - \nabla_{\mathbf{w}} \ln Z(\mathbf{w})
$$
하지만 이 식은, 하나의 데이터에 대한 식임.
데이터가 i.i.d라고 가정하면, 우리는 likelihood의 gradient의 기댓값을 계산할 수 있음.
$$\mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} \ln p(\mathbf{x} | \mathbf{w}) \right] 
= - \mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right] 
- \nabla_{\mathbf{w}} \ln Z(\mathbf{w})
$$
이때 $p_{\mathcal{D}}(\mathbf{x})$는 데이터의 분포임.

#### 항 변환
이때,
$$- \nabla_{\mathbf{w}} \ln Z(\mathbf{w}) = \int \left\{ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right\} p(\mathbf{x} | \mathbf{w}) \, d\mathbf{x}$$
이고,
$$\int \left\{ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right\} p(\mathbf{x} | \mathbf{w}) \, d\mathbf{x} 
= \mathbb{E}_{\mathbf{x} \sim \mathcal{M}} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right]
$$
임. 이때 $\mathcal{M}$은 모델, $p(\mathbf{x}|\mathbf{w})$임.

따라서 항을 변환한 최종 식은,
$$\nabla_{\mathbf{w}} \mathbb{E}_{\mathbf{x} \sim p_D} \left[ \ln p(\mathbf{x} | \mathbf{w}) \right] 
= - \mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right] 
+ \mathbb{E}_{\mathbf{x} \sim p_M(\mathbf{x})} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right]
$$
임.

#### 우리의 목표
우리는 이제 likelihood가 최대가 되는 $\mathbf{w}$를 찾고 싶음.
즉, gradient가 0이 되는 순간.
즉,
$$\mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right] = \mathbb{E}_{\mathbf{x} \sim p_M(\mathbf{x})} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right]$$

## 14.3.3 Langevin dynamics
이제 우리는,
1. $\mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right]$
2. $\mathbb{E}_{\mathbf{x} \sim p_M(\mathbf{x})} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right]$
를 계산해야함.

$\mathbf{x}$가 주어지면, automatic differentiation을 이용해 $\nabla_{\mathbf{w}} E(\mathbf{x}_n, \mathbf{w})$ 계산 가능.

#### 1.
$$\mathbb{E}_{\mathbf{x} \sim p_D} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right] 
\approx \frac{1}{N} \sum_{n=1}^{N} \nabla_{\mathbf{w}} E(\mathbf{x}_n, \mathbf{w})
$$

#### 2.
얘가 조금 더 어려움.
이걸 계산하기 위해 **stochastic gradient Langevin Sampling**이 필요
![[Algorithm 14.4.png]]

$M$개의 sample을 뽑아서,
$$\mathbb{E}_{\mathbf{x} \sim p_M(\mathbf{x})} \left[ \nabla_{\mathbf{w}} E(\mathbf{x}, \mathbf{w}) \right] 
\approx \frac{1}{M} \sum_{m=1}^{M} \nabla_{\mathbf{w}} E(\mathbf{x}_m, \mathbf{w})
$$
이런 식으로 근사함.