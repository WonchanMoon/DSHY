Transformer는 다양한 NLP task에 적용이 될 수 있는데,
input과 output에 따라 크게 3개로 나눌 수 있다.

#### Sequence IN, single Variable OUT
예를 들어, 감정분석
> 이 트랜스포머는, 시퀀스의 encoder 역할을 함

#### single Vector IN, Sequence OUT
예를 들어, 사진을 input으로 받아서 그 사진의 caption을 생성하는 것
> 이 트랜스포머는 decoder 역할을 함

#### Sequence IN, Sequence OUT
예를 들어, 한 언어를 다른 언어로 번역
> 이 트랜스포머는 encoder, decoder 역할을 모두 다 함.

## 12.3.1 Decoder Transformers
이 모델은 **생성형 모델**로 사용될 수 있다.
여기서는 **GPT(Generative Pre-trained Transformer)** 라는 모델 클래스를 중점적으로 다룸.
> 이 모델의 목표는, 트랜스포머를 활용해 자기회귀 모델을 구축해 나가는 것.
$$p(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \prod_{n=1}^{N} p(\mathbf{x}_n \mid \mathbf{x}_1, \ldots, \mathbf{x}_{n-1})
$$

여기서의 조건부 분포 $p(\mathbf{x}_n \mid \mathbf{x}_1, \ldots, \mathbf{x}_{n-1})$는 트랜스포머 신경망을 사용하여 데이터로부터 학습됨.

### 모델 구성
Input : $n - 1$개의 토큰을 가진 시퀀스
Output : $n$번째 토큰의 조건부 분포

우리가 이 Output으로 나온 조건부 분포에서, 샘플을 하나 뽑으면 input 시퀀스에 $n$번째 토큰을 추가할 수 있다. 그러면 또 $n+1$번째 토큰의 조건부 분포를 만들 수 있고, 이를 무수히 반복하여 최대 길이까지 시퀀스를 생생성할 수 있다. 
> 최대 길이 : Transformer의 입력 토큰 수에 의해 결정

### 모델의 구조
stack of transformer layers, 
	**Input** : $\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}$ 토큰의 시퀀스, 각 토큰의 차원은 $D$
	**Output** : $\widetilde{\mathbf{x}}_{1}, \ldots, \widetilde{\mathbf{x}}_{n}$ 토큰의 시퀀스, 각 토큰의 차원은 $D$
> 이건 transformer layer의 구조

GPT의 최종 목표는, $K$개의 토큰을 갖고 있는 딕셔너리에서, 현재 위치에서 특정 토큰이 선택될 확률을 나타내고 싶은 것이다.
따라서 우리의 Output은, 각 위치에서 $K$개의 토큰에 대한 확률 분포를 나타내야한다. 즉, 한 위치에서 $K$ 차원의 벡터를 나타내야함. 하지만 우리의 모델의 각 출력은, $D$ 차원의 벡터. 차원이 맞지 않음. 따라서 우리는 $D\times K$의 차원을 가지는 행렬 $\mathbf{W}^{(p)}$를 도입하여 아래 식으로 차원을 맞춰줌. 이때 softmax함수는 row별로 적용.
$$\mathbf{Y} = \text{Softmax} \left( \tilde{\mathbf{X}} \mathbf{W}^{(p)} \right)
$$
이때, $\mathbf{Y}$의 각 row는, $K$ 차원을 가지는 벡터 $\mathbf{y}_{n}^{\top}$. 
> 출력 시퀀스의 $n$번째 위치에서, $K$개의 토큰이 나올 확률들을 가진 벡터

그리고 $\tilde{\mathbf{X}}$의 각 row는, $D$ 차원을 가지는 벡터 $\mathbf{x}^{\top}_{n}$.
> $n$번째 입력 토큰이, 어텐션을 마치고 나온 결과.

### 모델의 훈련

![[12.3.1.png]]

**학습 데이터** : 라벨이 없는 대규모 데이터, Self-Supervised Learning
	**Input** : $\mathbf{x}_{1},...,\mathbf{x}_{n}$
	**Output** : 입력 시퀀스 다음에 오는 단어 $\mathbf{x}_{n+1}$

이때 입력되는 **시퀀스들은 i.i.d.** 로 간주. 모든 시퀀스들을 동시에 병렬처리하여 효율성을 높일 수 있음.
근데 생각해보면, **입력되는 시퀀스들의 길이가 모두 다를 것**임. 
-> 이를 해결하기 위해, **pad** 토큰을 도입. **마스킹 어텐션**을 활용하여, **pad 토큰은 무시하도록** 해줌.

#### Cheating
하지만 모델을 적합하는 과정에서, **Cheating**이 발생할 수 있음.
>**Cheating** : 다음에 나올 단어를 맞추는 모델인데, 모델이 학습과정에서 **미래 단어를 미리 봐버리는 문제**. 그렇게 되면, 모델이 다음 단어를 맞추는게 아니라, 그저 학습 된 것에서 **복사**해버림. 이렇게 되면, 모델이 새로운 시퀀스를 만드는 능력을 잃게 됨.

이를 방지하기 위해 두 가지 방법을 사용함
1. **입력 토큰들을 하나씩 미룸.**
	하나씩 미뤄야 다음 단어를 학습할 수 있음. 그렇지 않으면 그저 복사를 하게 됨.
	>(GPT answer)
	>Shift를 하지 않으면, 모델은 원본 시퀀스를 그대로 입력으로 받고, 출력(target)으로도 동일한 원본 시퀀스를 사용하게 됩니다.
	>예를 들어:    
	    입력(Input): `"I swam across the river"`
	    목표(Target): `"I swam across the river"` 
	이런 경우, **각 토큰이 자기 자신을 그대로 예측**하도록 학습됩니다:    
	    - `"I"` → `"I"`
	    - `"swam"` → `"swam"`
	    - `"across"` → `"across"`
	    - `"the"` → `"the"`
	    - `"river"` → `"river"`
	결과적으로 모델은 **문맥을 학습하지 않고 단순 복사만 학습**하게 됩니다.

	하지만 한 칸을 미루고 학습한다면?
	입력 : `<start>, I, swam, across, the` 
	목표 : `I, swam, across, the, river` 
	이처럼 다음 단어를 맞출 수 있게 되는 것임.

2. **마스킹 어텐션**
	어텐션은 학습하는 과정에서 문장의 모든 단어를 봐버림. 즉, **미래 단어도 같이 보게 됨**.
	**미래 단어의 어텐션 가중치를 $-\infty$으로 만들어서** 미래 단어는 학습에 사용이 안되도록 설정.
	>어텐션 가중치가 $-\infty$여야, softmax를 이용해 확률을 계산했을 때 확률이 0이 됨.

## 12.3.2 Sampling Strategies
>Recall) Decoder Transformer의 output은, 다음에 올 단어의 확률 분포

우리는 시퀀스를 늘려가기 위해, 어떤 단어가 올지 골라줘야함. 어떤 단어가 올지 고르는 데에는 굉장히 다양한 방법이 있다고 함.
> 내가 생각하기에 그냥 확률 제일 큰 거 고르면 되는 줄 알았는데 아닌가 봄.

### Beam Search
내가 생각한 것 처럼, 매번 제일 높은 확률을 고르는 greedy한 방법도 있음.
하지만! 매번 제일 높은 확률의 토큰을 생성하는 것과, 가장 높은 확률의 전체 시퀀스를 선택하는건 다름!
따라서 우리는, **모든 토큰에 대한 결합 분포를 최대화 할 것**임.
$$p(\mathbf{y}_1, \ldots, \mathbf{y}_N) = \prod_{n=1}^{N} p(\mathbf{y}_n \mid \mathbf{y}_1, \ldots, \mathbf{y}_{n-1})
$$
하지만, 우리가 모든 시퀀스 조합을 따지면, $O(N^{K})$라 말이 안됨. 그리디로 찾으면 $O(NK)$.

우리는 **Beam Search**로 시퀀스를 찾을 것임.
> Beam Search
> : 다음 단어를 예측할 때, 여러 가설을 설정하고 가장 확률이 높은 가설 $B$개를 고름.
> [[Additional Resources#Beam Search]]

Beam Search의 시간 복잡도는 $O(BNK)$
하지만 그리디, Beam에는 단점이 두 개 있음.
1. **출력의 다양성을 제한**하고, 
2. **동일한 시퀀스가 무수히 반복**되는 루프에 빠질 수 있다
	>저번에 회사에서 쓰다가 봤는데 좀 무서웠음...

### Top-p Sampling (누클리어스 샘플링)
상위 $K$개의 확률을 가지는 토큰만 고려하여, 이 토큰들의 확률을 정규화하여 샘플링하는 방법도 있음.
이 방법을 변형한 **Top-p Sampling 방법**이 존재.

**Top-p Sampling** : 출력된 토큰들의 누적 확률을 계산하여, 임계값에 도달할 때 까지 상위 토큰들을 집합에 포함하고, 임계값에 도달한 후 이 집합에서 샘플링을 수행

조금 더 부드러운 Top-p Sampling 버전이 있음.
$$y_i = \frac{\exp(a_i / T)}{\sum_j \exp(a_j / T)}
$$
$a_{i}$는 $i$번째 토큰이 선택될 확률(정규화 되기 전)
$y_{i}$는 $i$번째 토큰이 선택될 확률(정규화 된 후)
$T$는 temperature, $a$의 역할 조정
	$T\rightarrow0+$, 확률이 제일 큰 토큰의 확률이 1에 가까워짐. 나머지 상태의 확률은 모두 0. greedy와 같아짐
	> *원래 $T=0$인데, 이론 상 모순이라 내가 바꿈*
	$T=1$, Softmax 함수를 그대로 갖고 옴.
	$T\rightarrow\infty$, 모든 토큰이 뽑힐 확률이 동일해짐.
	$0<T<1$, 확률이 큰 값이 더 커짐.


## 12.3.3 Encoder Transformers
**Input** : 시퀀스
**Output** : 고정된 길이의 벡터

### 인코더 구조

![[12.3.2.png]]

#### 예시 모델 : BERT
대표적인 모델 **BERT (Bidirectional Encoder Representations from Transformers)** 가 있음.

모델의 목표는, **대규모 데이터셋을 통한 pre-training**을 거치고 **transfer learning**을 활용해, 더 작은 크기의 특화 데이터셋에 맞게 **fine-tuning** 하는 것.

모델은 토큰 시퀀스를 입력으로 받아 pre-train됨. 토큰 중의 랜덤한 확률(15%)로 일부 토큰이 **mask** 처리, 이후 모델은 이 mask를 예측하도록 학습. 
예를 들어, **I ⟨mask⟩ across the river to get to the ⟨mask⟩ bank.** 처럼 문장이 주어졌다면, 우리의 인코더 트랜스포머는 출력 노드 2에서 swam을 예측하고, 10에서 other를 예측해야함. 이때 모델 학습에 쓰이는 loss function에는 오직 2, 10번째 노드만이 영향을 줌.
> 하지만 이 mask 방법을 사용하면, 
> pre-training에 사용한 데이터셋에는 mask가  있고,
> fine-tuning에 사용할 데이터 셋에는 mask가 없는 불일치가 생김.
> -> 뽑힌 15% 중, **80%는 mask, 10%는 다른 단어, 10%는 유지**하되 그 단어를 맞추도록 학습
> 이렇게 함으로써 데이터의 다양성을 확보
> 근데 그냥 fine-tuning 데이터셋에 mask 만들면 안됨? [[Additional Resources#BERT-mask|안되는 이유]]

Bidirectional이 붙은 이유는, 네트워크가 마스킹 된 단어 이전과 이후의 모든 단어를 보기 때문.

### 인코더 학습 후
인코더 모델이 학습된 후에는, 목적에 맞게 fine-tuning이 얼마든지 가능.
이를 위해 해결하려는 작업에 따라 **새로운 출력 레이어**를 구성할 수 있음.

예를 들어 분류 문제를 풀기 위해서는 softmax, logistic sigmoid, MLP 를 추가하는 걸 고려할 수 있음.

fine-tuning 과정에서는 gradient-descent로 학습됨.

## 12.3.4 Sequence-to-sequence transformers
예를 들어 우리가 영어 -> 네덜란드어 번역을 한다고 치자.
디코더를 통해, 출력에 해당하는 네덜란드어 문장을 하나씩 생성할 수 있음. 이때 이 생성된 문장은, 입력이었던 영어 문장에 따라 생성되어야함.
>this output needs to be conditioned on the entire input sequence corresponding to the English sentence. (내가 번역하기 어려워서 원문 추가)

인코더는, 입력 시퀀스를 적절한 내부 표현 $\mathbf{Z}$로 매핑하는데 사용할 수 있음. 
우리는 이 $\mathbf{Z}$를, 출력 시퀀스 생성과정에 통합하기 위해, **cross-attention**이라는 어텐션 메커니즘을 사용.

cross-attention은 self-attention과 동일하지만, 
- 쿼리 벡터는 생성중인 시퀀스에서 나오고,
- 키, 값 벡터는 $\mathbf{Z}$로 표현되는 시퀀스로 부터 나옴.

디코더 구조
![[12.3.3.png]]

인코더 + 디코더
![[12.3.4.png]]

좀 더 디테일한 트랜스포머 구조
![[12.3.5.png]]