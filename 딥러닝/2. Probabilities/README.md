# Deep Learning 교재 공부
## CHAPTER 2. Probabilities
### uncertainty
In almost of every application of machine learning we have to deal with uncertainty
Uncertainty는 두 종류가 있음.
#### epistemic uncertainty (systematic uncertainty)
이는 우리가 finite 크기의 데이터를 관찰 하기에 생기는 문제.
데이터의 수를 늘리거나(observe more data), 더 많은 사례를 학습하면 문제를 줄일 수 있음.
#### aleatoric uncertainty (intrinsic uncertainty, stochastic uncertainty, noise)
데이터를 무수히 많이 모아도 문제는 uncertainty는 여전히 남아있음.
우리가 데이터의 모든 것을 보지 못하고 일부만 관측할 수 있기 때문. 따라서 이를 줄이기 위해서는 여러 종류의 데이터(different kinds of data)를 모으는 것을 생각할 수 있음.
### Frequentist's view
예를 들어, 구부러진 동전이 있어서 이를 매우 많이(large number of times) 던졌더니 전체 시행에서 60%는 concave side up, 40%는 convex side up. 이 경우 우리는 동전을 던졌을 때 이 동전이 concave side up일 **확률**은 0.6 = 60%라고 이야기 할 수 있다. 
이렇듯 확률은 제한된 횟수의 시행에 의해 정의된다. 그리고 이 경우 확률의 합은 1이 된다.

### Bayesian view
위의 코인 토스 문제에서, 우리가 코인을 던졌을 때 이 동전이 concave side up일 확률이 0.6임을 알고 있어도, 우리가 동전을 관찰할 수 없으면 우리는 동전의 어느 면이 head이고 어느면이 tail일지 모른다. 따라서 이 경우 우리는 동전이 H로 떨어질 확률이 0.5라고 추측할 것이고, 이는 rational하다. 
이처럼 확률을 불확실함의 정도(quantification of uncertainty)로 사용하는 것이 Bayesian perspective.

### Rules of probabilities.
확률 계산에 사용되는 몇 가지 룰을 확인
#### Sum Rule
$$
\begin{aligned}
p(X) &= \sum\limits_{Y}{p(X,Y)} \newline
p(x) &= \int {p(x,y)} dy
\end{aligned}
$$
#### Product Rule
$$
\begin{aligned}
p(X,Y) &= p(Y|X)p(X) \newline
p(x) &= p(y|x)p(x)
\end{aligned}
$$
#### Bayes' Theorem
$$
\begin{aligned}
p(X,Y) &= { {p(X|Y)p(Y)} \over {p(X)}} \\
p(y|x) &= {{p(x|y)p(y)} \over {p(x)}}
\end{aligned}
$$

#### independent
$p(X,Y) = p(X)p(Y)$ then $X, Y$ are said to be **independent**

Also, if $X, Y$ are independent, then $p(X|Y) = p(X)$

### Expectation, Variance
#### Expectation
The weighted average of some function $f(x)$ under a probability distribution $p(x)$ is called the *expectation* of $f(x)$ and will denoted by $\mathbb{E}[f]$

$$
\begin{aligned}
\mathbb{E}[f] &= \sum\limits_{x} {p(x)f(x)} \\
&= \int p(x)f(x)dx
\end{aligned}
$$
우리에게 $N$개의 데이터가 있다면 expectation은 다음과 같이 approximation이 가능함.
$$\mathbb{E}[f] \approx {{1}\over{N}}\sum\limits_{n=1}^{N}f(x_{n}) $$
expectation에 아래 첨자를 이용해서 어떤 변수에 대한 평균을 구할 것인지 나타내기도 함.
$$\mathbb{E}_{x}[f(x,y)]$$
위 경우, expectation은 $y$에 대한 함수가 될 것임.

#### Conditional Expectation
$$
\begin{aligned}
\mathbb{E}_{x}[f|y] &= \sum\limits_{x} p(x|y)f(x) \\
&= \int p(x|y)f(x)dx
\end{aligned}
$$

#### Variance
우리는 중학교 때부터 분산은 편차 제곱의 평균이라고 익히 알고 있음
$$
\begin{aligned}
var[f] &= \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^{2}] \\
&= \mathbb{E}[f(x)^{2}] - \mathbb{E}[f(x)]^2
\end{aligned}
$$

$f$의 분산이라 함은, $f(x)$가 그 평균인 $\mathbb{E}[f(x)]$로 부터 얼마나 떨어져있냐를 나타냄.

#### Covariance
$$
\begin{aligned}
cov[x,y] &= \mathbb{E}_{x,y}[(x-\mathbb{E}(x))(y-\mathbb{E}(y))]\\
&= \mathbb{E}_{x,y}[xy] - \mathbb{E}[x]\mathbb{E}[y]
\end{aligned}
$$
만약 $x,y$가 벡터라면 covariance matrix는 아래와 같이 계산됨.

### 2.3 Gaussian Distribution
$$
N(x|\mu, \sigma^{2})= {1 \over {\sqrt{2 \pi \sigma^{2}}}} \exp \{ {- {1 \over {2\sigma^2}}(x-\mu)^2\}}
$$
우리가 잘 알고 있듯이,
$$ \mathbb{E}[x] = \mu, \quad var[x] = \sigma^2$$
그리고 $N$개의 관측치가 있고, 각 데이터가 $N(x|\mu, \sigma^2)$ 에서 i.i.d 되었다면, 데이터의 likelihood function은 다음과 같다.
$$p(x|\mu, \sigma^{2} )= \prod^{N}_{n=1} N(x_{n}|\mu, \sigma^2)$$
위 likelihood function을 통해 파라미터를 추정할 수 있다. likelihood function을 최대화하는 maximum likelihood. 우리가 잘 아는 계산을 통해 추정된 파라미터는
$$
\mu_{ML} = {1 \over {N}}\sum\limits^{N}_{n=1} x_{n}, \quad \sigma^2_{ML}={1 \over {N}}\sum\limits^{N}_{n=1}(x_{n}-\mu_{ML})^{2} 
$$
그리고 $\mu_{ML}$은 unbiased, 그러나 $\sigma^2_{ML}$은 biased.
$$
\mathbb{E}[\mu_{ML}] = \mu, \quad \mathbb{E}[\sigma^{2}_{ML}] = ({{N-1}\over{N}})\sigma^2
$$
그리고 여기서 알 수 있듯이, $N$이 커짐에 따라 이 bias는 무시할만 해진다.
Maximum Likelihood로 추정된 파라미터는 overfitting과 연관된다.
#### Linear Regression 과의 연결
본 교재에서는 설명변수가 하나인 다항회귀를 대상으로 설명.
반응변수를 여기서는 target variable 이라고 함. 우리는 이 target variable에 대한 uncertainty를 확률 분포로서 표현. 이 확률분포는 Gaussian 분포라고 가정.
그러면 데이터의 likelihood function은 다음과 같음.
$$
p(t|x, w, \sigma^{2} )= \prod^{N}_{n=1} N(t_{n}|y(x_{n}, w), \sigma^{2})
$$
$w_{ML}$을 추정하기위해 자연로그를 씌우고, $w$를 갖고 있지 않은 항들을 정리하면 이는 LSE와 같아짐을 알 수 있음. 이렇게 추정한 $w_{ML}$을 가지고 $\sigma^{2}_{ML}$도 추정할 수 있음.

### 2.5 information theory
사실 아직 이게 왜 나왔고, 어디에 사용되는 지는 모르겠음.

Entropy는 간단하게 얘기해서, 데이터를 봤을 때 '놀람의 정도'이다.
데이터의 확률이 낮으면 크게 놀랄 것이고, 반대로 확률이 높았다면 조금 놀랐을 것이다.
(낮은 $p(x)$ 높은 $h(x)$, 높은 $p(x)$ 낮은 $h(x)$)
그리고 서로 관련 없는 정보(확률에서는 독립, $p(x,y) = p(x)p(y)$)가 들어왔을 때 내가 놀라는 정도는 각각의 정보에 놀라는 정도의 합으로 표현이 되어야함.
($h(x,y) = h(x) + h(y)$)
이를 연결 짓기 위해 로그를 사용. $h(x) = - \log_{2}x$
$$
Entropy = E[\ln p(x)]
$$
discrete variable에서 entropy가 최대화 되는 데이터의 분포는 uniform distribution,
continuous variable에서 entropy가 최대화 되는 데이터의 분포는 gaussian distribution.

Entropy는 the average amount of information needed to specify the state of a random variable.


KL-Divergence
간단하게 두 분포의 차이 정도로 이해하면 됨(딥러닝 학교 1일차 강의 중)
교재에서는, 'the average additional amount of information required to specify the value of x as a result of using $q(x)$ instead of the true distribution $p(x)$'
$$
\operatorname{KL}\,(p||q) = - \mathbb{E}_{p} [\ln {q(x) \over p(x)}]
$$

Conditional Entropy
$$
\mathrm{H}[y|x] = - \int\int p(y,x) \ln p(y|x) dy dx
$$

$$
\mathrm{H}[x,y] = \mathrm{H}[y|x] + \mathrm{H}[x]
$$

### 2.6 Bayesian
$$
\begin{aligned}
posterior &\propto P(E|H) \times P(H) = likelihood \times prior \\
\log posterior &= \log likelihood + \log prior + C \\
\theta_{MAP} &= \arg max_{\theta} (\log posterior) \\
\end{aligned}
$$

Bayesian 과 Frequentist의 차이
두 패러다임에서 모두 likelihood가 중요한 역할을 함.
frequentist는 파라미터를 그저 고정된 값으로 보고 그 고정된 값을 추정.
하지만 Bayesian은 파라미터를 확률변수로 취급함. 그래서 이 추정된 파라미터에 대한 uncertainty를 분포로서 나타냄.

MAP : Maximum a posterior 
posterior를 최대화하기.

