Transformer는 Attention이라는 처리 개념에 기반을 두고 있음. 

Transformer라고 부르는 이유 : 벡터 집합을 특정 표현 공간에서 동일한 차원을 유지하는 새로운 공간의 대응 벡터 집합으로 변환(transform)하기 때문.
Transformer는 원래 자연어 처리 분야에서 소개되었으나, 다른 많은 도메인에서도 뛰어난 성능을 보임. Vision Transformer는 이미지 처리에서 CNN을 능가, 여러 유형의 데이터를 결합하는 Multimodal Transformer는 매우 강력한 딥러닝 모델.

Transformer는 전이학습이 매우 효과적임. 대규모 데이터에 대해 학습한 후, 약간의 파인튜닝을 통해 여러 후속작업에 적용할 수 있음. 이렇게 다양한 작업을 해결할 수 있도록 적응할 수 있는 대규모 모델을 Foundation model 이라고 함. 

# 12.1 Attention
>🎵 뉴진스 - 어텐션 https://www.youtube.com/watch?v=js1CtxSY38I

Transformer의 근본 개념은 Attention.
Attention은 원래 RNN의 성능을 향상 시키기 위해 개발된 개념(Bahdanau, Cho, and Bengio, 2014)인데, 순환 구조를 제거하고 오직 Attention에만 집중 했을 때 성능이 크게 향상되는 것을 확인(Vaswani et al., 2017)

Attention is All you need

(1) I swan across the river to get to the other **bank**
: 나는 강을 건너 다른 쪽 **강둑**에 가기 위해 수영했다.
(2) I walked across the road to get cash from the **bank**
: 나는 길을 건너 **은행**에서 현금을 인출했다.

두 문장에서 bank의 의미는 다름. 이는 문맥을 파악해야만 알 수 있는 일. 또한, bank의 의미를 결정하는데에 있어 모든 단어가 똑같은 역할을 갖는게 아니라, 일부 단어가 더 큰 영향력을 가짐. 따라서 bank의 적절한 해석을 결정하려면 특정 단어에 더 많은 집중을 해야함(12.1.1 참고). 

12.1.1
![[12.1.1.png]]

또한 중요한 단어의 위치는 입력 시퀀스에 따라 달라짐. (1)에서는 2, 5번째가 중요했고, (2)에서는 8번째 단어가 중요했음. 
그러나 일반적인 신경망에서는 입력값이 출력값에 미치는 영향이 입력값에 곱해지는 가중치에 따라 달라짐. 그리고 신경망의 학습이 끝나면, 가중치와 관련된 입력값들은 전부 고정됨. 
Attention에서는, 입력 데이터에 따라 값이 결정되는 가중치 계수를 사용(12.1.2 참고)

12.1.2
![[12.1.2.png]]

### Word Embedding
자연어 처리를 논할 때, **Word Embedding**을 사용하여 단어를 **임베딩 공간의 벡터**로 매핑. 임베딩은 단어의 의미를 포착하여, 비슷한 의미를 가지는 단어끼리는 임베딩 공간에서 가까운 위치에 있도록 매핑함. 또한 기존의 임베딩은, 주어진 단어가 항상 동일한 임베딩 벡터로 매핑이 됨. 
하지만 Transformer는 더 풍부한 형태의 임베딩을 제공하여, 특정 벡터를 해당 시퀀스 내의 다른 벡터들에 따라 동적으로 매핑하는 특징이 있음. 
(1)의 bank는 water와 가까운 위치, (2)의 bank는 cash와 가까운 위치로 매핑할 수 있음.

트랜스포머는 단백질의 멀리 떨어진 아미노산들이 서로 어텐션을 적용할 수 있도록 하여, 단백질의 3차원 구조를 더욱 정밀하게 모델링할 수 있도록 돕는다(Vig et al., 2020).

## 12.1.1 Transformer processing
Transformer에 입력되는 데이터는 차원이 $D$인 벡터들의 집합 $\{x_{n}\}, n=1,...,N$임. 이러한 데이터 벡터를 **토큰(token)** 이라 부름. 예를 들어 문장 내의 단어가 토큰이 될 수 있음. 토큰의 요소 $x_{ni}$는 **특징(feature)** 이라 불림.
Transformer의 강력한 특성 중 하나는, 서로 다른 데이터를 처리하기 위해 새로운 신경망 구조를 설계할 필요 없이, 데이터를 하나의 토큰 집합으로 결합하여 처리할 수 있다는 것임.

> (나의 이해를 위한) 보충 설명
> **입력 데이터**(예를 들어 문장)는, **토큰**(단어)이 $N$개임. 그리고 각 토큰을 컴퓨터가 이해할 수 있게 **벡터로 임베딩** 해줘야 하는데, 이때 벡터의 차원이 $D$. 그리고 임베딩된 벡터의 각 원소를 **특징**이라고 함.

데이터 벡터를 $N\times D$차원 행렬 $\mathbf{X}$로 결합. 
이 행렬의 $n$번째 행은 하나의 토큰 벡터 $\mathbf{x}^{T}_{n}$임. 
하나의 행렬은, 하나의 입력 토큰 집합을 나타냄.

Transformer의 기본 구조는 입력행렬 $\mathbf{X}$를 받아 변환된 행렬 $\widetilde{\mathbf{X}}$을 출력으로 생성하는 함수임.
$$\widetilde{\mathbf{X}}=\text{TransformerLayer}[\mathbf{X}]$$
이후 여러 Transformer 레이어를 연속적으로 적용하여 풍부한 내부 표현을 학습할 수 있는 깊은 네트워크(**deep network**)를 구축할 수 있음. 각 트랜스포머 레이어는 자체적인 가중치와 편향을 가지고, 이는 적절한 손실함수를 사용한 gradient descent로 학습될 수 있음.

### Transformer Layer의 구조
토큰 벡터 간의 관계를 파악하는 **Attention**, 이후 이를 반영하여 벡터를 업데이트하는 **피드포워드**

## 12.1.2 Attention coefficients
임베딩 공간에서, $N$개의 입력 토큰 $\mathbf{x}_{1},...,\mathbf{x}_{N}$이 주어졌다고 가정하자. 우리는 이보다 풍부한 의미 구조를 포착할 수 있는 새로운 임베딩 공간에서 같은 개수의 출력 토큰 $\mathbf{y}_{1},...,\mathbf{y}_{N}$으로 매핑하고자 한다.

특정 벡터 $\mathbf{y}_{n}$을 고려해보자. 이 벡터는, $\mathbf{x}_{n}$에 영향을 받을 뿐만 아니라, $\mathbf{x}_{1},...,\mathbf{x}_{N}$의 모든 벡터에도 영향을 받아야한다. **어텐션**을 사용하면, $\mathbf{y}_{n}$을 결정하는데 있어 $\mathbf{x}_n$의 중요도를 반영할 수 있다. 이를 나타낼 수 있는 제일 간단한 방법은, 아래와 같이 선형 결합으로 나타내는 것이다.
$$
\mathbf{y}_n = \sum_{m=1}^{N} a_{nm} \mathbf{x}_m
$$
이때 $a_{nm}$은 **어텐션 가중치(Attention Weight)** 라고 한다.
> 어텐션 가중치를 행렬로 나타내면, 이 행렬은 N x N

어텐션 가중치는 다음과 같은 조건을 만족해야한다.
1. **모든 가중치 $a_{nm}$은 0 이상**. 왜냐하면, 이는 한 가중치가 너무 큰 양수일 때, 다른 가중치가 이를 상쇄하기 위해 너무 작은 음수가 되는 것을 방지하기 위해서이다.
2. **정규화 조건**. 특정 input에 더 많이 어텐션을 하면, 다른 input에는 상대적으로 그 어텐션이 줄어들도록 보장. $$\sum\limits^{N}_{m=1} a_{nm} = 1$$
위 두 조건을 만족하면, 모든 어텐션 가중치는 0에서 1 사이의 값을 가지게 된다.
특수한 경우로, $a_{mm}=1$이고 $a_{nm}=0, (n \neq m)$이면, 출력 벡터는 아무 변환 없이 그대로 유지된다. 일반적으로 **출력 $y_{m}$은 여러 입력 벡터의 조합으로 구성되고, 일부 입력이 다른 입력보다 더 큰 가중치를 갖는다.**
> 위의 두 조건은, 출력 토큰 별로 각각 적용된다. 즉, $y_{n}$별로 각각 적용된다는 것이다.

## 12.1.3 Self-attention
이제 우리는, 어텐션 가중치를 어떻게 정할 것인지 알아봐야한다. 그 전에, 일부 용어를 알고 넘어가자.

### 정보 검색 분야(Information retrieval)의 용어
온라인 영화 스트리밍 서비스에서 어떤 영화를 볼지 선택하는 문제가 있다고 생각해보자. 이를 해결하는 하나의 접근 방식은, 각 영화를 **속성**들과 연관짓는 것이다. 

**속성**의 예
- 장르
- 주연 배우의 이름
- 영화의 길이 등

사용자는 이를 통해 원하는 속성에 맞는 영화를 찾을 수 있다. 이 과정을 다음과 같이 자동화 할 수 있다. 각 영화의 속성을 벡터(이를 **키(key)** 라 한다)로 표현한다. 영화의 실제 파일은 **값(value)** 이라 한다. 그리고 사용자가 원하는 속성을 벡터로 입력하면, 이를 **쿼리(query)** 라고 한다. 영화 서비스는 쿼리와 키를 비교하여, 사용자에게 제일 잘 맞는 영화의 값을 제공한다. 우리는 각 유저가 그들의 쿼리와 제일 일치하는 키를 가진 특정 영화에 **attend** 하고 있다고 생각할 수 있다. 이 과정은 **하나의 value vector가 리턴**되는 **하드 어텐션**의 한 형태로 볼 수 있다.

트랜스포머에서는 이를 일반화하여 **소프트 어텐션**을 사용한다. 쿼리와 키 사이의 유사도를 연속적인 값(가중치)으로 측정. 즉, 단 하나의 영화만 선택하는 것이 아니라 **여러 영화의 값을 가중합 하여 최종 출력을 생성**

#### 정보 검색 분야와 트랜스포머의 연결
입력 벡터 $\mathbf{x}_{n}$을, value vector이자 key vector라고 생각한다(**Self-Attention**). 이는 영화 자체를 이용하여 해당 영화의 특징을 요약하는 것과 유사하다고 볼 수 있다. 또한, $\mathbf{y}_{m}$에 대한 query vector도 입력 벡터 $\mathbf{x}_{n}$을 그대로 사용한다. 
> 보충 설명
> 입력 벡터 $\mathbf{x}_{n}$ 자체를 value이자 key로 사용 : 단어 자체의 정보를 사용하여 특징을 요약
> query도 $\mathbf{x}_{n}$을 사용 : 각 단어가 다른 단어와의 관계를 볼 때 자기자신도 포함.
> -> Self-Attention !

이제 한 입력 벡터 $\mathbf{x}_{n}$이, 다른 입력 벡터 $\mathbf{x}_{m}$을 얼마나 참조해야 하는지를 결정하기 위해, 두 벡터의 유사도를 계산한다. 유사도를 측정하기 위해 우리는 내적 $\mathbf{x}_{n}^{T}\mathbf{x}_{m}$을 사용한다. 그리고 앞서 정해둔 어텐션 가중치의 두 조건을 만족하기 위해, 우리는 softmax 함수를 이용하여 내적 값을 변환해준다.
$$
a_{nm} = \frac{\exp\left(\mathbf{x}_n^\top \mathbf{x}_m\right)}{\sum_{m'=1}^{N} \exp\left(\mathbf{x}_n^\top \mathbf{x}_{m'}\right)}
$$
> 이때, softmax 함수를 썼다고 해서 가중치를 확률로 해석하지 않도록 할 것. 단순히 정규화를 위한 용도로 사용하였음.

### 중간 내용 정리
내용을 한 번 정리해보면,
- 입력 벡터 $\mathbf{x}_{n}$은 이제 다른 입력 벡터들의 선형 결합을 통해 출력 벡터 $\mathbf{y}_{n}$로 변환됨.
- 그리고 이 선형결합에 사용되는 어텐션 가중치는 softmax 함수를 통해 계산됨.
- 그리고 이때 가중치 계산은, 두 입력 벡터의 내적을 통해 계산됨.
이때, 입력 벡터가 서로 orthogonal 하다면, 출력 벡터는 입력 벡터와 같아짐.

우리는 matrix 표현을 빌려서, 어텐션을 다음과 같이 나타낼 수 있음.
$$
\mathbf{Y} = \text{Softmax}[\mathbf{X}^{T}\mathbf{X}]\mathbf{X}
$$
이때, $\mathbf{X} : N\times D$ 크기의 입력 행렬, $\mathbf{Y} : N\times D$  크기의 출력 행렬, 그리고 $\text{Softmax}[\mathrm{L}]$은, 행렬 $\mathrm{L}$의 각 원소에 지수함수를 취하고, 각 행을 독립적으로 정규화하여 합이 1이 되도록 조절하는 것이다.

이 과정을 우리는 **self-attention**이라고 한다. 왜냐하면 우리는 동일한 시퀀스를 사용하여 쿼리, 키, 값을 모두 결정하기 때문이다. 또한 쿼리와 키의 유사도를 내적으로 측정하므로, **dot-product self-attention**이라고 하기도 한다.

> 쿼리, 키, 값이 이해가 잘 안 가서 내용 추가

트랜스포머의 입력 데이터가 다음과 같이 주어졌다고 가정합시다.
입력 문장:  "The cat sat on the mat"

이 문장을 트랜스포머에 넣으면, 각 단어(토큰)가 벡터로 변환됩니다. 즉, 각 단어가 하나의 벡터(임베딩)로 표현됩니다.

이제, **각 단어 벡터가 다음과 같은 역할을 하게 됩니다.**
- **쿼리(Query) 벡터:** "이 단어가 다른 단어와 어떤 관계를 맺어야 할까?"
- **키(Key) 벡터:** "나는 이런 특성을 가진 단어야!"
- **값(Value) 벡터:** "이 단어의 실제 정보(최종적으로 반영될 벡터)"

**각 단어는 문장 내에서 자신이 다른 단어를 얼마나 중요하게 여겨야 하는지를 판단**하기 위해,  
쿼리(Query)와 키(Key) 사이의 유사도를 계산합니다.

유사도가 높은 단어일수록, 해당 단어의 **값(Value) 벡터가 최종 출력에 더 많이 반영**됩니다.

## 12.1.4 Network parameters
현재 셀프-어텐션 연산은 두 가지 문제점이 있음
1. 데이터가 주어지면 연산이 고정적이므로, 학습 가능한 가중치가 없어서 데이터를 통해 조절이 불가능함. 
2. 각 토큰 벡터 $\mathbf{x}_{n}$이 동일한 중요도로 어텐션 가중치를 결정하는데에 기여함
	(왜냐하면 기존 방법에서는 그저 벡터 내적으로 어텐션 가중치를 결정하기 때문)

이 두 가지를 해결하기 위해, 기존 벡터를 선형 변환하여 새로운 벡터를 정의할 수 있음.
$$
\tilde{\mathbf{X}} = \mathbf{X} \mathbf{U}
$$
이때 $\mathbf{U}$는 $D\times D$ 크기의 **학습 가능한 가중치 행렬**, 이는 **일반적인 신경망에서 하나의 레이어에 해당**

따라서 기존 변환 과정을, 다음과 같이 수정할 수 있음.
$$
\mathbf{Y} = \text{Softmax} \left[ \mathbf{X} \mathbf{U} \mathbf{U}^{\top} \mathbf{X}^{\top} \right] \mathbf{X} \mathbf{U}.
$$
하지만 이 과정 역시 아직 부족함. 왜냐? 여기서는 $\mathbf{X} \mathbf{U} \mathbf{U}^{\top} \mathbf{X}^{\top}$가 symmetric하기 때문,...
> 왜 문제?
> 예를 들어, chisel과 tool은 강한 연관이 있지만, 반대로 tool은 chisel과 약한 연관을 가짐.
> 그러나 위의 식대로 계산을 한다면, chisel -> tool, tool -> chisel의 중요도가 같아지는 문제가 발생.

따라서 어텐션에서는 **비대칭성**이 굉장히 중요함. 우리는 더 유연한 모델을 만들기 위해, 쿼리와 키를 독립적으로 학습할 수 있도록 설계할 수 있음. 
또한, 위의 식에서는 value vector와 어텐션 가중치를 결정하는데에 있어 $\mathrm{U}$를 동시에 사용하고 있음. 이는 바람직하지 못함. 

우리는 위 과정을 극복하기 위해, 쿼리와 키, 값 행렬을 별도로 정의할 수 있음
$$\begin{aligned}
\mathrm{Q} &= \mathrm{X} \mathrm{W}_{q}\\
\mathrm{K} &= \mathrm{X} \mathrm{W}_{k}\\
\mathrm{V} &= \mathrm{X} \mathrm{W}_{v}\\
\end{aligned}$$
여기서 가중치 행렬 $\mathrm{W}_{q}, \mathrm{W}_{k}, \mathrm{W}_{v}$은 최종 트랜스포머 아키텍쳐를 학습하는 동안 학습될 매개변수.
$\mathrm{W}_{k}$ 는 $D\times D_{k}$의 크기를 가지며, $D_{k}$는 키 벡터의 길이를 의미. 또한 $\mathrm{W}_{q}$ 역시 $D\times D_{k}$의 크기를 가져야한다. $\mathrm{W}_{v}$ 는 $D\times D_{v}$ 크기를 행렬. $D_{v}$는 출력 벡터의 차원을 결정, $D_{v}=D$로 설정하면, 입력 차원과 출력 차원이 동일하게 되어, 잔차 연결에 유리하다.
> Recall) 잔차 연결

이러한 설정을 통해, 각 레이어가 동일한 차원을 유지하면 여러 개의 트랜스포머 레이어를 서로 쌓을 수 있다. 이를 통해 어텐션 연산을 다음과 같이 나타낼 수 있음.

$$\mathrm{Y}=\text{Softmax}[\mathrm{Q}\mathrm{K}^{\text{T}}]\mathrm{V}$$
이때, $\mathrm{Y}$는 $N\times D_{v}$, $\mathrm{Q}\mathrm{K}^{\text{T}}$는 $N\times N$ 의 크기이다. 그리고 이 계산과정을 시각화 하면 아래 두 그림 처럼 나타낼 수 있다.

12.1.3
![[12.1.3.png]]

12.1.4
![[12.1.4.png]]

실제 구현에서는, 이러한 변환에 bias를 포함. 앞으로는 bias가 암묵적으로 포함된 것으로 간주.

## 12.1.5 Scaled Self-Attention
softmax함수의 gradient는, 입력 값의 크기가 커질 수록 지수적으로 작아지는 경향이 있음.
따라서 아래처럼 스케일링을 하여 어텐션을 정의
$$\mathbf{Y} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \equiv \text{Softmax} \left[ \frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{D_k}} \right] \mathbf{V}.$$
이때, $D_{k}$는 $\mathrm{Q}\mathrm{K}^{\top}$의 분산.
#### $D_{k}$에 대한 설명 보충
$D_{k}$는 $\mathrm{Q}\mathrm{K}^{\top}$의 분산? → 솔직히 무슨 말인지 와닿지 않았음
$D_{k}$는 key 벡터 차원의 크기 ; $\mathbf{K} : N\times D_{k}$, $N$개의 입력 토큰, 각 토큰이 $D_{k}$의 차원을 가지는 벡터
근데 이렇게 $D_{k}$를 정의하고 보면, 이게 $\mathbf{Q}\mathbf{K}^{\top}$의 분산과 같아진다고 함.

![[12.1.5.png]]
이 구조가 하나의 Attention Head

## 12.1.6 Multi-head Attention
위 구조를 우리는 하나의 **Attention Head**라 한다. 어텐션 헤드는, 입력 토큰 간의 의존성 패턴에 주목(Attend)할 수 있도록 한다. 하나의 어텐션 헤드는, 하나의 의존성 패턴에 집중하는 경향이 있다.

하지만 우리는, 동시에 여러 개의 어텐션 패턴이 중요할 수 있다. 예를 들어서, 한 패턴은 말의 시제(tense)와 관련된 것이고, 다른 패턴은 어휘에 관련된 것일 수 있다. 이때 하나의 어텐션 헤드만을 사용한다면, 이러한 효과들이 평균으로 퉁쳐져서 중요한 정보를 놓칠 수 있다.

우리가 $H$ 개의 헤드를 가지고 있다고 생각하면, 각 헤드는 다음과 같다.
$$\mathbf{H}_h = \text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h)$$
여기서의 어텐션 함수는, 다음 Scaled self-attention을 의미한다.
$$\mathbf{Y} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \equiv \text{Softmax} \left[ \frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{D_k}} \right] \mathbf{V}.$$
각 헤드에 대해, 별도로 쿼리, 키, 값 행렬을 정의한다.
$$\begin{aligned}
    \mathbf{Q}_h &= \mathbf{X} \mathbf{W}_h^{(q)} \\
    \mathbf{K}_h &= \mathbf{X} \mathbf{W}_h^{(k)} \\
    \mathbf{V}_h &= \mathbf{X} \mathbf{W}_h^{(v)}
\end{aligned}
$$
이렇게 계산된 각 헤드들을 하나의 행렬로 연결한 후, 가중치 행렬 $\mathbf{W}^{(\text{o})}$를 사용하여 선형변환 해준다.
$$\mathbf{Y}(\mathbf{X}) = \text{Concat} \left[ \mathbf{H}_1, \dots, \mathbf{H}_H \right] \mathbf{W}^{(\text{o})}$$
이 계산을 그림으로 나타내면, 아래 그림 12.1.6과 같다.

12.1.6
![[12.1.6.png]]

각 행렬 $\mathbf{H}_{h}$는 $N\times D_{\text{v}}$의 크기를 가지며, 이를 Concat한 매트릭스는 $N\times H D_{\text{v}}$의 크기를 가진다. 이 Concat된 매트릭스는, $H D_{\text{v}}\times D$의 크기를 가지는 $\mathbf{W}^{(\text{o})}$에 의해 $N\times D$의 크기를 가지는 매트릭스 $\mathbf{Y}$로 선형변환된다. 이는 입력 행렬 $\mathbf{X}$와 동일한 차원이 된다.

행렬 $\mathbf{W}^{(\text{o})}$의 원소들은 훈련 과정에서 쿼리, 키, 값 행렬 ($\mathbf{Q}_{h}, \mathbf{K}_{h}, \mathbf{V}_{h}$)과 함께 학습된다.

또한 일반적으로, $D_{\text{v}}$는 $D/H$로 설정하여, concat된 매트릭스의 크기가 $N\times D$가 되도록한다.

![[Algorithm 12.1.png]]
![[Algorithm 12.2.png]]

## 12.1.7 Transformer layers
멀티 헤드 셀프 어텐션은, 트랜스포머 네트워크의 핵심 구조.
우리는 신경망이 깊어질수록 성능이 좋다는 것을 알고 있음. 
따라서 **깊게깊게 쌓고** 싶고, 훈련 효율성을 위해 **잔차 연결**을 수행하고 싶음.

따라서 우리는 깊게 쌓기 위해, 어텐션 레이어의 출력 차원을 입력 차원과 동일한 $N\times D$로 맞춰주어야 함.
그리고 이때 레이어의 출력 값을 안정화 하는 레이어 정규화를 수행.
$$\mathbf{Z} = \text{LayerNorm}\left[\mathbf{Y}(\mathbf{X}) + \mathbf{X}\right]
$$
이때 $\mathbf{Y}(\mathbf{X})$는, 멀티 헤드 셀프 어텐션
$$\mathbf{Y}(\mathbf{X}) = \text{Concat} \left[ \mathbf{H}_1, \dots, \mathbf{H}_H \right] \mathbf{W}^{(\text{o})}$$

또는, 레이어 정규화가 멀티 헤드 셀프 어텐션 이전에 적용되는 Pre-norm으로 대체 가능.
$$\mathbf{Z} = \mathbf{Y}(\mathbf{X}') + \mathbf{X}, \quad \text{where} \quad \mathbf{X}' = \text{LayerNorm}[\mathbf{X}].
$$
위 두 경우 모두, $\mathbf{Z}$의 차원은 입력 차원과 같은 $N\times D$ 임.

> 레이어 정규화?
> 각 row 별(각 토큰 별)로, 평균을 0, 분산을 1로 맞추어 주는 과정

우리는 여기까지 봤을 때, 어텐션 메커니즘은 선형결합 이후 softmax 함수를 통과시켜 비선형성을 만들어주는 것을 볼 수 있다. 하지만 이래도 결국 선형결합을 조금 변형시킨 것에 불과하다.

모델을 더 유연하게 만들기 위해, 레이어의 출력을 $D$개의 입력과 $D$개의 출력을 가진 표준 비선형 신경망으로 처리할 수 있음. 이를 $\text{MLP}[ . ]$로 나타냄.
$$\widetilde{\mathbf{X}} = \text{LayerNorm}[\text{MLP}(\mathbf{Z}) + \mathbf{Z}]
$$

여기서도 마찬가지로, pre-norm을 사용할 수도 있겠죠?
$$\widetilde{\mathbf{X}} = \text{MLP}(\mathbf{Z}') + \mathbf{Z}, \quad \text{where} \quad \mathbf{Z}' = \text{LayerNorm}[\mathbf{Z}].
$$

이 과정을 그림으로 나타내면,![[12.1.7.png]]
이게 하나의 트랜스포머 레이어고, 어텐션 레이어와 피드포워드 레이어로 구성된다.

![[Algorithm 12.3.png]]

