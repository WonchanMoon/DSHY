이전 챕터에서, 함수의 기댓값 $\mathbb{E}[f]$를 계산하기 위한 두 샘플링 기법
[[Sampling 1#14.1.3 Rejection Sampling|Rejection Sampling]]과 [[Sampling 1#14.1.5 Importance Sampling|Importance Sampling]]에 대해서 다루었음.
하지만 이 두 샘플링은 고차원에서 큰 문제가 발생.

따라서 매우 일반적이고 강력한 샘플링 기법인 **Markov Chain Monte Carlo**를 소개.

#### Preliminaries
**Markov Chain Monte Carlo**는, 어떤 샘플링 기법이 아니라, **아이디어**임. 
> (GPT 설명)
> MCMC는 마코프 체인을 설계하여 **주어진 목표 분포(Target Distribution)에서 표본을 생성하는 과정**입니다. MCMC 알고리즘은 다음과 같은 과정으로 진행됩니다.
> 
> 1. 초기 상태 $X_{0}$ 설정: 임의의 초기값 선택
> 2. 새로운 상태 $X_{t}$ 샘플링: 특정한 규칙에 따라 새로운 샘플 $X_{t}$ 를 생성
> 3. 목표 분포에 수렴할 때까지 반복: 충분한 반복 후, 샘플이 목표 분포에 따라 생성되도록 유도
> 4. 샘플을 사용하여 기대값 추정: 충분한 샘플을 얻으면, 이를 이용해 기대값 계산
> 
> 이 과정에서 중요한 점은 적절한 전이(Transition) 규칙을 사용하여 샘플이 목표 분포에 수렴하도록 하는 것입니다.

그리고 이 MCMC를 수행하기 위해서는 여러 알고리즘이 나오고, 책에서 다루는 것은 **Metropolis-Hastings**와 **Gibbs** 샘플링

---
$\mathbf{z}^{*}$ : sample from proposal distribution $q(\mathbf{z}|\mathbf{z}^{(\tau)})$
$p(\mathbf{z})=\tilde{p}(\mathbf{z})/Z_{p}$. $\tilde{p}(\mathbf{z})$는 $\mathbf{z}$가 주어졌을 때, $Z_{p}$를 모름에도 불구하고 쉽게 계산할 수 있다고 가정
## 14.2.1 The Metropolis algorithm
![[Algorithm 14.1.png]]

#### 알고리즘 부연설명
샘플 $\mathbf{z}^{\star}$가 accept될 확률은, 
$$A(\mathbf{z}^{\star}, \mathbf{z}^{(\tau)}) = \min \left( 1, \frac{\tilde{p}(\mathbf{z}^{\star})}{\tilde{p}(\mathbf{z}^{(\tau)})} \right)
$$
으로 정의되고, 임의로 선택된 $u\in\text{Unif}[0,1]$에 대해,
$$A(\mathbf{z}^{\star}, \mathbf{z}^{(\tau)})>u$$
이면 샘플은 accept되고, 그렇지 않으면 기각된다.
>  $\tilde{p}(\mathbf{z}^{\star})/\tilde{p}(\mathbf{z}^{(\tau)})>1$의 의미?
>  새로운 샘플 $\mathbf{z}^{\star}$가, 기존 샘플 $\mathbf{z^{(\tau)}}$보다 target distribution에서 높은 확률을 지님.
>  -> $\mathbf{z}^{\star}$가 더 합리적인 값

#### 추가적으로,
proposal distribution $q$가, 
모든 $\mathbf{z}_{A}, \mathbf{z}_{B}$에 대해 $q(\mathbf{z}_{A}|\mathbf{z}_{B}) > 0$ 이라면 (**즉, $0$이 아니라면 == 다른 $\mathbf{z}$값으로 바뀔 가능성이 있다면**),
$$\text{Distribution of }\mathbf{z}^{(\tau)}\rightarrow p(\mathbf{z})\quad \text{ as }\quad \tau\rightarrow\infty$$
> 즉, $\tau$ (시간 또는 반복 횟수)가 커질수록 $z(\tau)$를 여러 번 모아서 히스토그램을 그리면 $p(z)$와 비슷해짐.
> https://www.youtube.com/watch?v=U561HGMWjcw
> ![[14.2.1.png]]
> ![[14.2.2.png]]
#### 그러나
Metropolis Sampling에는 두 가지 한계점이 있음
1. **대칭인 proposal distribution**밖에 다룰 수 없음(Normal, Uniform)
	(Metropolis의 가정)
	[[Sampling 2#14.2.3 The Metropolis-Hasting Algorithm|14.2.3 The Metropolis-Hasting Algorithm]]에서는, proposal distribution의 비대칭성을 반영하는 알고리즘을 소개. 
2. Random Walk 기반이기에, 탐색할 수 있는 범위가 제한적임.

## 14.2.2 Markov Chain
본격적인 알고리즘 소개에 들어가기 전, Markov Chain의 일반적인 내용들에 대한 학습.
#### First-Order Markov Chain
First-Order Markov Chain은, random variables $\mathbf{z}^{(1)},...\mathbf{z}^{(M)}$의 series로 정의되는데, 이들은 아래 성질을 만족.
$$p(\mathbf{z}^{(m+1)}|\mathbf{z}^{(1)},...,\mathbf{z}^{(m)})=p(\mathbf{z}^{(m+1)}
|\mathbf{z}^{(m)})\quad \text{for } m\in\{1,...,M-1\}$$

#### Markov Chain
우리는
1. **초기 확률 분포** $p(\mathbf{z}^{(0)})$
2. **transition probabilites** $T_m(\mathbf{z}^{(m)}, \mathbf{z}^{(m+1)}) \equiv p(\mathbf{z}^{(m+1)} \mid \mathbf{z}^{(m)})$
이 두 가지로 Markov Chain을 특정지을 수 있음.

특히, transition이 모든 $m$에 대해 같다면, Markov Chain이 **homogeneous** 하다고 함.

#### Marginal, Invariant(stationary)
**Marginal 분포**는 아래 식으로 구해짐. 이때 이산 확률변수에서 적분은 summation으로 대체됨.
$$p(\mathbf{z}^{(m+1)}) = \int p(\mathbf{z}^{(m+1)} \mid \mathbf{z}^{(m)}) p(\mathbf{z}^{(m)}) \,\mathrm{d} \mathbf{z}^{(m)}$$

특정 확률분포 $p^{\star}(\mathbf{z})$가 Markov Chain 이후에도 그대로 유지가 된다면, 그 분포는 Markov Chain에 **invariant**, 혹은 **stationary** 하다고 함.
$$p^{\star}(\mathbf{z}) = \int T(\mathbf{z}', \mathbf{z}) p^{\star}(\mathbf{z}') \,\mathrm{d} \mathbf{z}'
$$

하나의 Markov Chain이, 여러 invariant distribution을 가질 수도 있음.
예를 들어, $T(\mathbf{z}', \mathbf{z})=\delta(\mathbf{z}-\mathbf{z}')$ : identity transform을 선택한다면, 어떤 $p(\mathbf{z})$를 선택하든 invariant.

MCMC에서는 우리의 target distribution을 invariant distribution으로 가지는 chain을 설계하는 것이 매우 중요.

#### Check the Invariant: Detailed Balance
transition이 아래 조건(**detailed balance**)를 만족한다면,
$$p^{\star}(\mathbf{z}) T(\mathbf{z}, \mathbf{z}') = p^{\star}(\mathbf{z}') T(\mathbf{z}', \mathbf{z})$$
확률분포 $p(\mathbf{z})$는 invariant.

> Simple Proof
> $$\begin{aligned}
    \int p^{\star}(\mathbf{z}') T(\mathbf{z}', \mathbf{z}) \,\mathrm{d} \mathbf{z}'
    &= \int p^{\star}(\mathbf{z}) T(\mathbf{z}, \mathbf{z}') \,\mathrm{d} \mathbf{z}' \\[8pt]
    &= p^{\star}(\mathbf{z}) \int p(\mathbf{z}' \mid \mathbf{z}) \,\mathrm{d} \mathbf{z}' \\[8pt]
    &= p^{\star}(\mathbf{z}).
\end{aligned}$$

그리고 Detailed Balance를 만족하는 Markov Chain은 **reversible**하다고 함.

#### Base Transitions
우리는 transition을 설계할 때, '**base**' transitions $B_{1},...,B_{K}$의 집합으로부터 설계할 수도 있음.
$$T(\mathbf{z}', \mathbf{z}) = \sum_{k=1}^{K} \alpha_k B_k(\mathbf{z}', \mathbf{z})$$
이때 $\alpha_{k}$는 모두 0 이상, 다 더해서 1

여러 단계의 중간 상태를 반영해서, 아래와 같이 쓸 수도 있음.
$$T(\mathbf{z}', \mathbf{z}) =
\sum_{\mathbf{z}_1} \dots \sum_{\mathbf{z}_{n-1}}
B_1(\mathbf{z}', \mathbf{z}_1) \dots B_{K-1}(\mathbf{z}_{K-2}, \mathbf{z}_{K-1}) B_K(\mathbf{z}_{K-1}, \mathbf{z})$$
>이 식은 다음과 같이 해석할 수 있습니다:
>$B_k(\mathbf{z}_{k-1}, \mathbf{z}_k)$:  
    각 단계 $k$ 에서 상태 $\mathbf{z}_{k-1}$​ 에서 $\mathbf{z}_k$​ 로 전이하는 확률.
  연속적인 전이 과정:
    - 첫 번째 전이: $B_1(\mathbf{z}', \mathbf{z}_1)$ → 상태 $\mathbf{z}'$ 에서 $\mathbf{z}_1$​ 로 전이.
    - 두 번째 전이: $B_2(\mathbf{z}_1, \mathbf{z}_2)$ → 상태 $\mathbf{z}_1$​ 에서 $\mathbf{z}_2$​ 로 전이.
    - …
    - 마지막 전이: $B_K(\mathbf{z}_{K-1}, z)$ → 상태 $\mathbf{z}_{K-1}$​ 에서 최종 상태 $z$ 로 전이.
  중간 상태 $\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_{K-1}$ 에 대해 합(sum) 연산 수행:
	*why?* 중간 경로가 여러 개 있을 수 있으므로, 모든 가능한 중간 상태에 대해 확률을 합산하여 전체 전이 확률 $T(\mathbf{z}', \mathbf{z})$ 를 계산.
	*중간 경로* : $\mathbf{z}$에서 $\mathbf{z}'$로 가는 경로
	*중간 상태* : 중간 경로를 지나면서 거치는 $\mathbf{z}_{k}$의 값들.

**두 가지 성질**
- base transition들에 invariant한 확률분포는, $T(\mathbf{z}', \mathbf{z})$에도 invariant
- base transition이 detailed balance -> $T(\mathbf{z}', \mathbf{z})$도 detailed balance
	*꼭 detailed balance인 것은 아님!* 
	ex) 중간 경로를 대칭적으로 설정하면 아닐 수도 있음. $$B_1, B_2, \dots, B_K, B_K, \dots, B_2, B_1$$

## 14.2.3 The Metropolis-Hasting Algorithm
![[Algorithm 14.2.png]]

#### 알고리즘 부연설명
샘플 $\mathbf{z}^{\star}$가 accept될 확률은, 
$$A_k(\mathbf{z}^{\star}, \mathbf{z}^{(\tau)}) =
\min \left( 1, \frac{\tilde{p}(\mathbf{z}^{\star}) q_k(\mathbf{z}^{(\tau)} \mid \mathbf{z}^{\star})}
{\tilde{p}(\mathbf{z}^{(\tau)}) q_k(\mathbf{z}^{\star} \mid \mathbf{z}^{(\tau)})} \right)
$$

으로 정의되고, 임의로 선택된 $u\in\text{Unif}[0,1]$에 대해,
$$A_{k}(\mathbf{z}^{\star}, \mathbf{z}^{(\tau)})>u$$
이면 샘플은 accept되고, 그렇지 않으면 기각된다.

이때 $k$는 가능한 여러 transition 집합의 원소들 중 하나를 나타냄.

또한 이때 $p(\mathbf{z})$는 invariant distribution.

#### Proposal Distribution의 선택이 성능에 영향을 미침
일반적으로 연속확률변수에서는, **정규분포**를 proposal distribution으로 사용.
이때, 정규분포의 표준편차를 어떻게 설정하냐에 따라 알고리즘의 성능이 크게 좌우됨.

*표준 편차가 너무 작다면?*
	새로운 샘플이 현재 샘플과 매우 가까운데서 생성 -> 샘플이 accept되는 경우가 너무 많아지고,
	또한 랜덤 워크 현상이 발생하고,
	결국 수렴이 늦어짐.

*표준 편차가 너무 크다면?*
	새로운 샘플이 현재 샘플과 매우 먼 곳에서 생성 -> accept되는 경우가 매우 낮아짐.
	결국 효율이 떨어짐.

#### (Continue) 다변수 분포에서의 문제
**상관관계가 너무 강하다면** 알고리즘의 효율성이 낮아짐.

다변수 분포에서는, proposal distribution의 표준편차를 $\sigma_{\text{min}}$과 비슷하게 설정해야함.
> $\sigma_{\text{min}}$은 다변수 분포에서 가장 짧은 방향의 표준편차 ![[14.2.3.png]]

다변수 정규분포에서, 독립적인 샘플을 얻기 위해서 필요한 반복 횟수는 대략적으로,
$$O\left(\left(\frac{\sigma_{\text{max}}}{\sigma_\text{min}}\right)^{2}\right)$$
분포가 타원형으로 길수록($\sigma_{\text{max}} >> \sigma_{\text{min}}$) 속도가 느려짐.

Neal., 1993에 따르면, 두 번째로 작은 표준편차 $\sigma_{2}$를 사용하면, 반복횟수가
$$O\left(\left(\frac{\sigma_{\text{max}}}{\sigma_{2}}\right)^{2}\right)$$
까지 낮아진다고 함.

#### (Continue) 그래서 어떻게 개선하는데? (by GPT)
만약 우리가 **이차원 정규분포**(타원형 형태)를 샘플링해야 한다면, 다음과 같은 방식으로 개선할 수 있다.
#### ✅ **제안 1: 주축 방향을 정렬하여 샘플링**
- 타겟 분포가 강한 상관관계를 가지면, **좌표 변환(principal component analysis, PCA)** 을 통해 새로운 축으로 정렬한 후 샘플링을 진행하면 효율적이다.
#### ✅ **제안 2: Adaptive Proposal Distribution**
- 초기에는 작은 $\rho$ 를 사용하여 탐색을 시작하고, 샘플링이 진행됨에 따라 $\rho$ 를 조절하는 방법도 가능하다. 이를 **Adaptive Metropolis-Hastings** 기법이라고 한다.
#### ✅ **제안 3: Hamiltonian Monte Carlo (HMC) 사용**
- MH는 단순 랜덤 워크를 기반으로 하므로 느리게 수렴할 수 있음. **Hamiltonian Monte Carlo (HMC)** 는 에너지 기반 탐색을 통해 이 문제를 해결할 수 있다.


## 14.2.4 Gibbs Sampling
#### 아이디어 설명
초기 샘플 설정 : $M$ 개의 sample$(z_{1}^{(0)},...,z_{M}^{(0)})$로 시작
각 샘플의 Conditional Distribution으로 샘플 업데이트
$$z_{i}^{(\tau + 1)}\sim p(z_{i}|z_{-i}^{(\tau)})$$

따라서 **Gibbs Sampling을 하기 위해서는**, Conditional Distribution을 알 수 있어야 함.
만약 Conditional Distribution을 모른다면,
1. 다른 샘플링 기법을 사용하거나
2. Conditional Distribution을 근사해서 Gibbs Sampling을 적용

또한 **Gibbs Sampling이 수렴**하기 위해서는,
1. 목표 분포를 invariant distribution으로 가져야함.
	>  Invariant Distribution : Transition 이후에도 분포가 변하지 않아야함.
2. Conditional Distribution이 0이 되면 안됨.
	> 만약 0이 된다면, 특정 값에 갇혀 버릴 수 있음.

![[Algorithm 14.3.png]]

![[14.2.4.png]]
#### 예시
분포 : $p(z_{1}, z_{2}, z_{3})$
우리가 $\tau$ 번째 step까지 마무리 했고, $z_{1}^{(\tau)}, z_{2}^{(\tau)}, z_{3}^{(\tau)}$ 까지 구함.

$z_{1}^{(\tau + 1)}$은 분포 $p(z_{1}|z_{2}^{(\tau)}, z_{3}^{(\tau)})$에서 샘플링해서 업데이트.
$z_{2}^{(\tau + 1)}$은 분포 $p(z_{2}|z_{1}^{(\tau + 1)}, z_{3}^{(\tau)})$에서 샘플링해서 업데이트.
$z_{3}^{(\tau + 1)}$은 분포 $p(z_{3}|z_{1}^{(\tau + 1)}, z_{2}^{(\tau + 1)})$에서 샘플링해서 업데이트.

#### 하지만 Gibbs Sampling은, 속도가 살짝 느림
Gibbs Sampling을 통해, 위 그림의 빨간색 분포에서 independent한 Sample을 뽑으려면, $(L/l)^{2}$만큼의 step이 필요함.
또한, 빨간색 분포 처럼 두 변수가 높은 상관관계를 가진다면, 속도는 더욱 느려짐.
> 이를 해결하기 위해, PCA같은 방법을 이용해서 축을 서로 상관관계가 낮도록 바꾸기도 함.
> 낮은 상관관계에서는 효율적임. $\mathcal{O(l)}$

#### 이걸 해결하기 위한 기법
**over-relaxation** 기법을 이용하면, 수렴 속도를 높일 수 있음.

기존에는 $z_{i}^{(\tau + 1)}$을 $p(z_{i}|z_{-i}^{(\tau)})$에서 샘플링 하는 것으로 끝냈다면,
over-relaxation에서는
$$z_{i}^{(\tau + 1)}=\mu+\alpha(z_{i}^{(\tau)}-\mu)+\eta$$
이때 $\alpha\in(-1,1),\eta\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$
$\mu$는 Conditional Distribution의 평균
$\eta$는 그저 노이즈를 주기위한 값, 
$\alpha$는 파라미터, $\alpha = 0$이면 그냥 Gibbs Sampling과 다를 게 없음. $\alpha<0$이면, 평균의 반대방향으로 다음 값이 업데이트

#### 하지만,
over-relaxation은 Conditional Distribution이 Gaussian일 때 잘 작동.
non-Gaussian에도 적용할 수 있는 **ordered over-relaxation**이 있음.

## 14.2.5 Ancestral Sampling
다양한 모델에서, 우리는 $p(\mathbf{z})$를 그래프를 이용하여 나타낼 수 있음.
$$p(\mathbf{z})=\prod^{M}_{i=1}{p(\mathbf{z}_{i}|\text{pa}(i))}$$

그리고 우리는 이 경우에, 그래프를 한 번 쭉 돌면 joint distribution, $p(\mathbf{z})$에서 부터 샘플을 얻을 수 있음.

이때 일부 노드가 관측이 되었다고 치자.
관측된 노드들을 모아 evidence set $\mathcal{E}$ 라고 하자.

우리는 이제 **likelihood weighted sampling** 기법을 이용하여 샘플링을 수행할 것임.

#### likelihood weighted sampling
각 변수에 대하여, 
해당 변수가 evidence set 에 존재한다면, 그냥 그 값을 가져다 사용하고,
그렇지 않다면 $p(\mathbf{z}_{i}|\text{pa}(i))$에서 샘플링.
