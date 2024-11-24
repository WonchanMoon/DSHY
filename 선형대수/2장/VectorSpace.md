# CHAPTER 2. Vector Spaces
## 2.1 Vector Spaces and Subspaces
더 깊은 선형대수의 세계로 초대합니다...
여전히 $Ax = b$ 를 풀고 이에 대해 이해하기 위함.

$\mathbf{R}^{n}$은 $n$개의 component를 가지는 모든 column vector를 포함하고 있다.

### Definition of Vector Space
**real vector space는 vector들과, vector의 합과 스칼라 곱이 함께 정의된 것.**

예를 들어, $\mathbb{R}^{\infty}$, $The\ space\ of\ 3\ by\ 2\ matrices$, $The\ space\ of\ functions\ f(x)$ 는 vector space.

벡터끼리의 합과 스칼라곱은 vector space에 속하는 또 다른 vector를 생성하고, 이 vector 역시 아래 eight properties를 만족
### Eight Properties
1. $x + y = y + x$
2. $x+(y+z)=(x+y)+z$
3. $\text{There is a unique "zero vector" s.t. }x+0 = x\ \text{for all }x.$
4. $\text{For each }x\ \text{there is a unique vector }-x\ \text{such that }x+(-x)=0.$
5. $1x=x$
6. $(c_{1}c_{2})x=c_{1}(c_{2}x)$
7. $c(x+y)=cx+cy$
8. $(c_{1}+c_{2})x=c_{1}x+c_{2}x$
### Subspace
정의 : Vector space의 **subspace**는, vector space의 **nonempty subset**으로서 vector space의 조건(subspace의 vector 끼리의 **linear combination**이 subspace에 존재)을 만족하는 것을 말한다.
1) subspace의 임의의 두 벡터 $x, y$가 있을 때, $x+y$도 subspace에 존재
2) subspace의 임의의 한 벡터 $x$가 있을 때, 임의의 상수 $c$에 대해 $cx$가 subspace에 존재

Subspace는 addition과 scalar multiplication에 닫혀있다.(closed)
-> 여덟개의 properties도 subspace에서도 유지됨.

zero vector는 모든 subspace에 속함. 즉, 어떤 subspace든지 간에 origin을 무조건 지나야함.

가장 작은 subspace는 zero vector 하나 만을 갖고 있는 subspace
가장 큰 subspace는 origin space 그 자체. ex) $\mathbb{R}^{3}$의 subspace 중 제일 큰 subspace는 $\mathbb{R}^{3}$ 그 자체

주의 : subset과 subspace는 다름.
예를 들어, 우리가 흔히 하는 2차원 평면에서 1사분면은, 2차원 평면($\mathbb{R}^{2}$)의 subset. 그러나 subspace는 아님.
$The\ space\ of\ 3\ by\ 3\ matrices$는 vector space. 그리고 lower triangular matrix를 모아둔 subspace가 있을 수 있음. 마찬가지로 symmetric matrix만 모아둔 subspace가 있을 수 있음.

### The Column space of A : $C(A)$
$$The\ column\ space\ contains\ all\ linear\ combinations\ of\ the\ columns\ of\ A.\ : Subspace\ of\ \mathbb{R}^{m}$$
**한 matrix 의 column 들의 모든 linear combination을 갖고 있는 것을 column space라 함.**
#### 2A : $Ax=b$ 를 풀어낼 수 있는지 확인 할 수 있는 방법
The system $Ax=b$ is solvable if and only if the vector $b$ can be expressed as a combination of the columns of $A$. Then $b$ is in the column space

$Ax = b$ can be solved if and only if $b$ lies in the plane that is spanned by two column vectors.

Column Space도 Vector Space의 조건을 만족.
위에서 언급했듯이, $\mathbb{R}^{m}$의 subspace. matrix의 각 column들은 $m$개의 components를 가짐. 따라서 이 $n$개의 column들의 linear combination도 $m$개의 components를 가짐 : $\mathbb{R}^{m}$ 

#### 깜짝 퀴즈~
제일 작은 Column space와 제일 큰 Column space는 각각 어느 matrix로 부터 만들어질까요?
[[정답]]

### The Nullspace of A : $N(A)$
$$The\ solutions\ to\ Ax=0\ form\ a\ vector\ space\ :\ the\ null\ space\ of\ A\ : Subspace\ of\ \mathbb{R}^{n}$$
한 matrix의 null space라는 것은, $Ax=0$을 만족하는 모든 $x$들을 모아놓은 것을 말함.
위에서 언급했듯이, $\mathbb{R}^{n}$의 subspace. $x$ 는 $n$ 개의 component를 가짐.

A의 column들이 independent 하면, Nullspace는 오직 0벡터만을 가질 수 밖에 없나?
그렇다네요 ~
$$\begin{bmatrix}1&0\\5&4\\2&4\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix}$$

A의 column들이 independent 하지 않으면, 비교적 큰 NullSpace가 될 수 있음.
$$B=\begin{bmatrix}1&0&1\\5&4&9\\2&4&6\end{bmatrix}$$

첫번째 column과 두번째 column의 합이 세번째 column과 같음.
$$\begin{bmatrix}1&0&1\\5&4&9\\2&4&6\end{bmatrix}\begin{bmatrix}c&c&-c\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix}$$
위 경우 이처럼 Nullspace가 line이 될 수 있음.

따라서, Nullspace와 Column space를 다음과 같이 이해할 수 있음.
Nullspace : All solutions of $Ax = 0$
Column space : $Ax=b$ 에서 attainable 한 모든 $b$

## 2.2 Solving $Ax = 0$ and $Ax=b$
Invertible matrix에서 Nullspace는 오로지 $x=0$ 만을 가지고, Column space는 전체가 됨.(아마 $\mathbb{R}^{m}$ ?)
그렇다면, 언제 Null space는 여러 벡터를 가지고, Column space는 언제 모든 벡터가 아닌 일부의 벡터만을 갖고 있을까?

### Echelon Form $U$ and Row Reduced Form $R$
$$A=\begin{bmatrix}1&3&3&2\\2&6&9&7\\-1&-3&3&4\end{bmatrix}$$
$$A\to\begin{bmatrix}1&3&3&2\\0&\mathbf{0}&3&3\\0&\mathbf{0}&6&6\end{bmatrix}$$
만약 여기서 두번째 row가 아닌 다른 row에서 두번째 column에 0이 있었다면, row의 순서를 바꿀 수 있지만 지금은 그럴 수 없다. 하지만 그렇다고 해서 멈추지 말자. 두번째 row에 들어있는 3을 pivot으로 보자.
$$U=\begin{bmatrix}\mathbf{1}&3&3&2\\0&0&\mathbf{3}&3\\0&0&0&0\end{bmatrix}$$
그럼 위와 같이 $U$를 만들어 낼 수 있다. 
이처럼 upper triangle 이지만, pivot이 main diagonal에 위치하지 않은 경우, $U$는 staircase pattern이 있다고 하거나, **echelon form** 이라고 한다.

$U$ 어디서 많이 보지 않았나요? 네~ 맞습니다. 바로 $A = LU$에서 봤지요. 그렇다면 여기서도 못할게 뭐가 있겠습니까?
$$L=\begin{bmatrix}1&0&0\\2&1&0\\-1&2&1\end{bmatrix}\quad\mathrm{and}\quad A=LU.$$
이렇게 $L$을 만들어 낼 수 있음을 알 수 있습니다.
$L$ 은 square matrix. 그 사이즈는 $A, U$의 row개수와 같다.

#### 2B
For any $m$ by $n$ matrix $A$ there is a permutation $P$, a lower triangular $L$ with unit diagonal, and an $m$ by $n$ echelon matrix $U$, such that $PA=LU$

위에서 만든 $U$에서, 아래처럼 reduced row echelon form $R$을 만들어 낼 수 있다.
$$\begin{bmatrix}1&3&3&2\\0&0&3&3\\0&0&0&0\end{bmatrix}\longrightarrow\begin{bmatrix}1&3&3&2\\0&0&\mathbf{1}&\mathbf{1}\\0&0&0&0\end{bmatrix}\longrightarrow\begin{bmatrix}\mathbf{1}&\mathbf{3}&\mathbf{0}&\mathbf{-1}\\\mathbf{0}&\mathbf{0}&\mathbf{1}&\mathbf{1}\\\mathbf{0}&\mathbf{0}&\mathbf{0}&\mathbf{0}\end{bmatrix}=R.$$
pivot을 1로 만들어주고, pivot 위로도 전부 0이 되도록...
$R$은 $A$ 에서 elimination이 끝난 matrix이다.
$Rx=0$의 solution은 $Ux=0,\ Ax=0$ 과 같은 solution을 가지기 때문에, 우리는 이 $R$로 $A$의 Null Space를 빠르게 찾을 수 있다.
- 왜죠?

### Pivot Variables and Free Variables
$Rx=0$ 의 solution을 읽어내는 것이 목표! pivot이 중요하다.

예시로 pivot variable과 free variable을 이해해보자.
$$Rx=\begin{bmatrix}\mathbf{1}&3&\mathbf{0}&-1\\\mathbf{0}&0&\mathbf{1}&1\\\mathbf{0}&0&\mathbf{0}&0\end{bmatrix}\begin{bmatrix}\boldsymbol{u}\\\boldsymbol{v}\\\boldsymbol{w}\\\boldsymbol{y}\end{bmatrix}=\begin{bmatrix}0\\0\\0\end{bmatrix}$$

$u, v, w, y$가 두 그룹으로 나누어 질건데, **pivot variables**와 **free variables**로 나누어진다. corresponding column이 pivot을 가진 colunm이면 pivot variable. 아니라면 free variable. 
이 경우, $u,\ w$가 pivot variables. $v,\ y$가 free variables

pivot variable들은, free variable을 통해 표현이 가능하다. 
$$\begin{aligned}u+3v-y&=0\quad\text{yields}\quad u=-3v+y\\w+y&=0\quad\text{yields}\quad w=\quad-y\end{aligned}$$
즉, 최종 $x$는 아래와 같이 표현이 가능하다.
$$x=\begin{bmatrix}-3v+y\\v\\-y\\y\end{bmatrix}=v\begin{bmatrix}-3\\1\\0\\0\end{bmatrix}+y\begin{bmatrix}1\\0\\-1\\1\end{bmatrix}$$
여기서 각 vector를 **speical solution**이라고 부른다. 그리고 $Rx=0$의 모든 해는 speical solution들의 linear combination으로 만들어진다.
얘네가 Nullspace의 basis가 되겠습니다 ~

아래 방법으로 $Ax=0$을 빠르게 풀어낼 수 있다.
1. $Rx=0$ 을 만들고, pivot variable과 free variable을 찾는다.
2. free variable 중 하나를 1, 나머지를 0으로 해서 speical solution들을 모두 찾는다.
3.  2.에서 찾은 special solution 들의 linear combination이 A의 Nullspace를 만든다.

또한 쉽게 알 수 있듯이, matrix가 column의 개수가 더 많다면($n>m$), 
이 matrix는 최소한 $n-m$ 개의 free variable을 갖는다. 
#### 2C
$Ax=0$에서 변수의 개수가 식의 개수보다 많다면($n>m$), 이는 최소한 하나의 special solution을 갖고 있다.

$$\text{If\ there\ are} \ r\ \text{pivots,\ there\ are}\ r\ \text{pivot\ variables\ and}\ n-r\ \text{free\ variables}.$$
그리고 여기서 $r$ 이 바로 이 matrix의 **rank**
rank는 column space에서 pivot column의 수, row space에서 pivot row의 수를 나타냄.
### Solving $Ax=b,Ux=c,$ and $Rx=d$
- $b \neq 0$ 인 케이스
$Ax = b$에서, $A, b$가 아래처럼 생겼다고 해보자.
$$A=\begin{bmatrix}1&3&3&2\\2&6&9&7\\-1&-3&3&4\end{bmatrix}\quad b=\begin{bmatrix}b_{1}\\b_{2}\\b_{3}\end{bmatrix}$$
$A$를 $U$로 바꾸고 나면(elimination 진행) 다음과 같이 $Ux=c$를 얻을 수 있다.
$$\begin{bmatrix}1&3&3&2\\0&0&3&3\\0&0&0&0\end{bmatrix}\begin{bmatrix}u\\v\\w\\y\end{bmatrix}=\begin{bmatrix}b_1\\b_2-2b_1\\b_3-2b_2+5b_1\end{bmatrix}$$
위 system은 해가 있을지 없을지 모른다. 특히 $b_3-2b_2+5b_1=0$ 이 아니면, 위 system의 해는 존재하지 않는다.

column space의 관점에서 접근해보자.
Recall) $Ax=b$ can be solved if and only if $b$ lies in the column space of $A$
위의 $A$에서, column들을 가져와보면 아래와 같다.
$$\begin{bmatrix}1\\2\\-1\end{bmatrix},\quad\begin{bmatrix}3\\6\\-3\end{bmatrix},\quad\begin{bmatrix}3\\9\\3\end{bmatrix},\quad\begin{bmatrix}2\\7\\4\end{bmatrix}$$
쉽게 볼 수 있듯이, 2번째 column은 첫번째 column의 2배, 네번째 column은 세번째 column과 첫번째 column의 차이이다. 
따라서 $C(A)$ 는 그저 첫번째, 세번째 column으로 만들어지는 plane이라고 볼 수 있다. 
$C(A)$ 를 $b_{3}-2b_{2}+5b_{1}=0$ 을 만족하는 vector $b$로 만들어지는 plane이라고 볼 수도 있다.

만약 $b$가 $C(A)$에 속한다면, $Ax=b$는 쉽게 풀린다. 

- 예를 들어, $b=(1,5,5)$라 하자. ($b_{3}-2b_{2}+5b_{1}=0$ 이어야 하기 때문)
$$Ax=b\quad\begin{bmatrix}1&3&3&2\\2&6&9&7\\-1&-3&3&4\end{bmatrix}\begin{bmatrix}u\\v\\w\\y\end{bmatrix}=\begin{bmatrix}1\\5\\5\end{bmatrix}$$
$$Ux=c\quad\begin{bmatrix}1&3&3&2\\0&0&3&3\\0&0&0&0\end{bmatrix}\begin{bmatrix}u\\v\\w\\y\end{bmatrix}=\begin{bmatrix}1\\3\\0\end{bmatrix}$$
그리고 Back-substitution을 통해, 
$$\begin{aligned}3w+3y=3\quad &\mathrm{~or~}\quad w=1-y \\ u+3v+3w+2y=1\quad &\mathrm{~or~}\quad u=-2-3v+y.\end{aligned}$$
임을 알아낼 수 있다.

-따라서 최종 solution은 아래와 같다.
$$\begin{aligned}\textbf{Complete solution}\ x\\x=x_{p}+x_{n}\\ \end{aligned}
\quad 
x=\begin{bmatrix}u\\v\\w\\y\end{bmatrix}=\begin{bmatrix}-2\\0\\1\\0\end{bmatrix}+\nu\begin{bmatrix}-3\\1\\0\\0\end{bmatrix}+y\begin{bmatrix}1\\0\\-1\\1\end{bmatrix}$$
이때, free variable이 달려있지 않은 vector를 **particular solution** 이라고 한다.
그리고 particular solution은 $Ax_{p}=b$를 만족하는 $x_{p}$ 이다.
particular solution은 모든 free variable을 0으로 두고 쉽게 구할 수 있다.

- 이처럼 $Ax=b$의 모든 solution은 하나의 particular solution과 $Ax=0$을 만족하는 solution들(special solution)의 합으로 나타낼 수 있다.
$$x_{\mathbf{complete}}=x_{\mathbf{particular}}+x_{\mathbf{nullspace}}$$
- 앞서 요약했던 $Ax=0$의 풀이 방법에 이어, $Ax=b$를 풀어내는 방법을 요약할 수 있다.
1. $Ax=b \rightarrow Ux=c$
2. free variable을 0으로 두고 particular solution $x_{p}$를 찾는다. ($Ax_{p}=b$ and $Ux_{p}=c$)
3. $Ax=0$을 풀어 speical solution을 찾는다.
4. $x=x_{p}+\text{any combination of special solution}$


## 2.3 Linear Independences, Basis, and Dimension
목표
1. Linear independence 와 dependence 이해하기
2. Spanning a Subspace 이해하기
3. Basis 이해하기
4. Dimension 이해하기

### linearly independent and dependent
#### 2E : linearly independent and dependent
$c_{1}v_{1} + \dots + c_{k}v_{k}=0$ 이 $c_1=\cdots=c_{k}=0$ 일 때만 성립한다면, $v_{1}, \dots , v_{k}$는 **linearly independent**. $c_{1},\dots,c_{k}$의 다른 combination이 가능하다면, $v_{1}, \dots , v_{k}$는 **linearly dependent**
linearly dependent인 경우에는, 임의의 한 벡터를 다른 벡터들의 linear combination으로 표현할 수 있다.

아래 matrix의 column들은 linearly dependent.
$$A=\begin{bmatrix}1&3&3&2\\2&6&9&5\\-1&-3&3&0\end{bmatrix}$$
아래 matrix의 column들은 linearly independent.
$$A=\begin{bmatrix}3&4&2\\0&1&5\\0&0&2\end{bmatrix}$$

- $N(A) = \{ \text{zero vector} \}$ 이면 $A$의 column들은 **linearly independent.**
이 아이디어를 활용하여 여러 벡터의 linearly independence를 체크할 수 있다.
그 vector들을 column으로 갖는 matrix $A$로 만들고, $Ac=0$을 풀어보면 알 수 있다. $c=0$이 아닌 solution이 나온다면, vector들은 dependent.

- Echelon matrix $U$에서, nonzero row들은 서로 linearly independent.
또한, pivot을 갖고 있는 column들 끼리도 linearly independent.

#### 2F
echelon matrix $U$와 reduced matrix $R$에 있는 $r$개의 nonzero row들은 서로 linearly independent. 
즉, $r$개의 column이 pivot을 갖고 있음을 알 수 있다.

#### 2G
$\mathbb{R}^{m}$에 속하는 $n$개의 vector는, $n>m$ 이라면 무조건 linearly dependent 하다.

### Spanning a Subspace
vector들의 집합이 space를 span 한다는게 뭔지 정의한다.
앞서 배웠던 column space $C(A)$를 떠올려 보자. 
$A$의 column space는, $A$의 column들에 의해 **spanned** 된다.
비슷하게, row space는 row들에 의해 **spanned** 된다.

#### 2H
vector space $V$가 $w_{1},\dots,w_{l}$의 모든 linear combination으로 구성된다면, 이 vector 들이 그 space를 **span** 한다고 한다. $V$에 속하는 임의의 vector $v$는 $w_{1},\dots,w_{l}$의 어떤 linear combination으로 만들어 진다.

예를 들어, 세 vector $w_{1}=(1,0,0),w_{2}=(0,1,0),w_{3}=(-2,0,0)$는, $\mathbb{R}^{3}$의 $xy$평면을 span한다.

- Spanning은 column space와 관련이 있고, independence는 null space와 관련이 있다.

### Basis for a Vector Space
#### 2I
vector space $V$의 **basis**는 아래 두 성질을 동시에 만족시키는 vector의 sequence이다.
1.  vector들이 linearly independent.
2. 그 vector들이 $V$를 span 해야함.

이 성질들이 linear algebra에서 absolutely fundamental 하대요

vector space $V$의 임의의 vector $v$를, $V$의 basis vector들의 combination으로 작성하는 방법은 유일하다.

- vector space는 무수히 많은 다른 basis를 가지고 있다. 
예를 들어, invertible한 n by n matrix가 있으면, 이 column들은 independent하고, $\mathbb{R}^{n}$의 basis 이다.
아래 nonsingular matrix의 두 개의 column도 $\mathbb{R}^{2}$의 basis이다.
$$A=\begin{bmatrix}1&1\\2&3\end{bmatrix}$$

- Echelon matrix $U$
pivot을 가지고 있는 column들이, column space의 basis가 된다. 
Echelon matrix의 column space와, elimination이 이루어지기 전 matrix의 column space는 다르다.
하지만 independent column의 개수는 보존이 됩니다~
$$A=\begin{bmatrix}1&3&3&2\\2&6&9&7\\-1&-3&3&4\end{bmatrix} \quad U=\begin{bmatrix}1&3&3&2\\0&0&3&1\\0&0&0&0\end{bmatrix}$$
$A$의 column space는 첫번재, 세번째 column으로 span 되는 plane.
$U$의 column space 역시 첫번째, 세번째 column으로 span 되는 plane 이지만, 이는 $xy$평면
둘이 다르쥬?

### Dimension of a Vector Space
알고 계셨나요? basis의 복수형은 bases 랍니다.

#### 2J
vector space $V$의 임의의 두 basis는 같은 개수의 vector를 가진다. 이 개수는 모든 basis가 동일하며, space의 "degree of freedom"이고 $V$의 **dimension** 이라 한다.

$\mathbb{R}^{n}$의 dimension은 $n$

#### 2K : First Big Thm in Linear Algebra (공식적인 이름은 아님)
$$\begin{aligned}&\mathrm{lf~}\nu_1,\ldots,\nu_m\mathrm{~and~}w_1,\ldots,w_n\text{ are both bases for the same vector space, then } m=n.\\
&\text{The number of vectors is the same.}\end{aligned}$$
증명은 교재 참고 : $n>m$이라고 가정하고 모순 발생.

- dimension이 $k$인 subspace에서는, 
1. $k$개 보다 많은 vector들은 서로 independent 할 수 없다. 
2. 그리고 $k$개 보다 많은 vector는 space를 span하는데에 불필요하다. $k$개면 충분하다.

#### 2L
$V$의 임의의 linearly independent set은, (필요하다면 몇 개의 vector를 추가해서) basis가 될 수 있다.
$V$의 임의의 spanning set은, (필요하다면 몇 개의 vector를 제외해서) basis가 될 수 있다.

basis가 가능한 최대의 independent set이다. (basis is maximal independent set)
independence를 유지하면서 더 확장할 수 없다.

basis는 가능한 최소한의 spanning set이다. (basis is minimal spanning set)
이보다 작은 set으로는 space를 span할 수 없다.

- 이제 우리가 dimension에 대해 크게 두 가지의 뜻을 알고 있는데요, 예시를 통해 비교해봅시다.
1. four-dimensional vector : 4차원 벡터, vector in $\mathbb{R}^{4}$, 4개의 component를 가짐
2. four-dimensional subspace : ex) 첫번째와 마지막 component가 0인 6차원 벡터들의 모임.

#### Theorem
column space의 dimension은 matrix의 rank와 같다.
