# 20.4 Guided Diffusion
많은 분야에서는, 조건부 분포 $p(\mathbf{x}\mid\mathbf{c})$에서 샘플링을 하고 싶어함.
> 특정 조건을 만족하는 이미지 생성

이를 달성하는 가장 쉬운 방법은, 디퓨전 모델의 신경망 $\mathbf{g}$에 클래스 값을 추가하여, $\mathbf{g}(\mathbf{z}, \mathbf{w}, t, \mathbf{c})$ 를 학습하는 것. (이때 정답은 $\{\mathbf{x}_{n},\mathbf{c}_{n}\}$)

하지만 이 방법의 가장 큰 한계점은, 신경망이 조건 변수 $\mathbf{c}$에 큰 중요도를 부여하지 않거나, 이 정보를 무시하고 학습을 해버릴 수 있다는 것이다.

## 20.4.1 Classifier Guidance
이미 훈련된 분류기 $p(\mathbf{c}\mid\mathbf{x})$가 있다고 가정해보자.

이때, 디퓨전 모델을 스코어 함수 관점에서 바라보면 다음과 같다.
$$
\nabla_{\mathbf{x}} \ln p(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \ln \left\{ \frac{p(\mathbf{c} \mid \mathbf{x}) p(\mathbf{x})}{p(\mathbf{c})} \right\}
= \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})
$$
#### 항 뜯어보기
분모 $\nabla_{\mathbf{x}} \ln p (\mathbf{c})=0$임을 이용하여 분모를 제거하였다.
$\nabla_{\mathbf{x}} \ln p(\mathbf{x})$는 디퓨전 모델의 스코어 함수
$\nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})$ 는 분류기 모델 하에서 주어진 레이블의 확률을 극대화 하는 방향으로 디노이징 과정을 유도(Dhariwal and Nichol, 2021)
> 샘플 $\mathbf{x}$가 주어진 클래스 $\mathbf{c}$에 속할 확률을 높이는 방향으로 노이즈 제거

#### Guidance Scale
분류기의 영향을 조절하기 위해, **Guidance Scale** $\lambda$ 를 도입할 수 있다.
$$
\text{score}(\mathbf{x}, \mathbf{c}, \lambda) = \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \lambda \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})
$$
- $\lambda = 0$ 이면, 그냥 디퓨전 모델과 같음.
- $\lambda = 1$ 이면, 조건도 반영
- $\lambda > 1$ 이면, 조건에 더 집착. 하지만 너무 크면 생성되는 샘플의 다양성이 감소.

#### 이 방법의 문제점
이 방법은, 별도의 분류기를 따로 학습시켜야 한다는 문제점이 있음.
심지어 이 분류기는, 노이즈가 섞인 샘플도 잘 분류해야함.

## 20.4.2 Classifier-free guidance
$$
\text{score}(\mathbf{x}, \mathbf{c}, \lambda) = \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \lambda \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})
$$
위 식에서, $$\nabla_{\mathbf{x}} \ln p(\mathbf{x} \mid \mathbf{c})
= \nabla_{\mathbf{x}} \ln p(\mathbf{x}) + \nabla_{\mathbf{x}} \ln p(\mathbf{c} \mid \mathbf{x})$$
임을 이용하여 식을 변형하면,
$$
\text{score}(\mathbf{x}, \mathbf{c}, \lambda) = \lambda \nabla_{\mathbf{x}} \ln p(\mathbf{x} \mid \mathbf{c}) + (1 - \lambda) \nabla_{\mathbf{x}} \ln p(\mathbf{x}),
$$
이다.
- $0<\lambda<1$ 이면, $\ln p(\mathbf{x} \mid \mathbf{c})$와 $\ln p(\mathbf{x})$ 의 convex combination.
- $\lambda > 1$ 이면, 모델이 조건 정보를 잘 반영하는 샘플의 생성을 선호하게 됨.

또한, 위 모델을 식을 훈련하기 위해서는, 각각 네트워크를 구성해줘야 함.
하지만, $p(\mathbf{x}\mid\mathbf{c})$ 네트워크 하나만 훈련 시키고, $\mathbf{c}$가 Null 값(예를 들어 $0$)인 경우 $p(\mathbf{x})=p(\mathbf{x}\mid\mathbf{c}=0)$으로 표현 할 수 있음. 따라서 $p(\mathbf{x}\mid\mathbf{c})$를 훈련하면서 일부(10 ~20%)를 $p(\mathbf{x})$ 훈련을 위해 $\mathbf{c}$를 Null로 설정.

Nichol et al., 2021, Saharia et al., 2022에 따르면, Classifier-free Guidance가 Classifier Guidance보다 훨씬 더 높은 품질의 결과를 낸다고 함.
> 왜냐하면,
> 분류기 $p(\mathbf{c}\mid\mathbf{x})$는, $\mathbf{c}$만 잘 예측할 수 있다면, $\mathbf{x}$ 대부분을 무시해도 됨. (일부 특징만 확인해도 잘 맞출 수 있음.)
> 하지만 Classifier-Free Guidance는 $p(\mathbf{x}\mid\mathbf{c})$기반이기 때문에, 이미지 전체($\mathbf{x}$)가 조건에 잘 맞아야 높은 확률이 부여됨.
> $p(\mathbf{c}\mid\mathbf{x})$ : 이 이미지가 고양이일 확률은?
> $p(\mathbf{x}\mid\mathbf{c})$ : 고양이 조건 하에, 가장 그럴싸한 이미지는?


### Text-guided diffusion model
![[20.4.1.png]]
LLM의 기법을 활용하여, 조건 입력을 프롬프트로 사용할 수 있도록 하는 모델.

텍스트 입력은 디노이징 과정에 다음과 같은 방법으로 영향을 줌.
1. 트랜스포머 기반 언어 모델의 내부 표현과 디노이징 네트워크의 입력을 concat.
> 프롬프트 -> *트랜스포머 기반 언어 모델(ex. BERT)에 넣음*
> 이 모델이 각 단어에 대한 *임베딩*을 만들어줌. 
> 이 임베딩을, 디퓨전 모델의 *디노이징 네트워크의 입력에 concat*
2. 디노이징 네트워크의 cross-attention 레이어가 텍스트 토큰 시퀀스를 직접 attend 하도록 함.
> cross-attention을 통해, 이미지의 각 영역이 텍스트의 어떤 부분을 참고해야하는지 결정할 수 있도록 해줌.
> 🐶 예시 프롬프트: “dog riding a skateboard”

| **이미지 위치**           | **주로 집중하는 텍스트 토큰**     | **설명**                   |
| -------------------- | ---------------------- | ------------------------ |
| **중앙 (main object)** | `dog`, `riding`        | 개의 몸통, 얼굴, 포즈 등          |
| **중앙 하단**            | `skateboard`, `riding` | 개가 타고 있는 스케이트보드          |
| **중앙 상단**            | `dog`                  | 개의 머리, 귀, 눈 등            |
| **하단 (바닥 근처)**       | `skateboard`           | 바닥 위 스케이트보드 바퀴, 그림자      |
| **배경 왼쪽**            | `riding`               | 동작감 강조 – 역동적인 배경 흐림 표현 등 |
| **배경 오른쪽**           | (거의 없음) 또는 약간의 `dog`   | 배경 처리, 텍스트의 직접적 연결성은 적음  |

### Image Super-Resolution
저해상도 이미지 -> 고해상도 이미지

**고해상도 샘플을 가우시안 분포에서 샘플링 한 후, 저해상도 이미지를 조건으로 삼아 디노이징 프로세스를 거침.**
![[20.4.2.png]]

Sahaira, Ho, et al., 2021에 따르면, **해상도를 계단식으로 높여가면** 매우 높은 해상도를 얻을 수 있다고 함.

#### 이미지 생성에서의 응용
Nichol et al., 2021; Saharia et al., 2022에 따르면, 계단식 방식은 이미지 생성에서도 사용될 수 있음. 
**낮은 해상도에서 이미지 디노이징을 수행하고, 이후 고해상도로 변환.**
이는 처음부터 고해상도에서 작업을 하는 것보다 계산 비용을 크게 줄일 수 있음.

#### Latent Diffusion Model
고해상도 이미지 공간에서, 디퓨전 모델은 계산 비용이 높음.
**Latent Diffusion Model** 접근법을 활용하면, 계산 비용을 줄일 수 있음.

1. 노이즈가 없는 이미지에 대해 **오토인코더를 학습**. -> 이미지의 latent 표현을 얻음
	이렇게 학습된 오토인코더는 고정.
2. 저차원 공간에서 디노이징을 수행.
3. 고정된 오토인코더의 디코더 부분을 활용해, 저차원 표현을 고해상도 이미지 공간으로 되돌림.

이 방법은, **이미지의 의미에 집중**할 수 있다는 장점이 있음.

이 접근법은, 
- 이미지 인페인팅(inpainting)
- 잘린 이미지 복원(un-cropping)
- 복원(restoration)
- 이미지 변형(image morphing)
- 스타일 변환(style transfer)
- 채색(colourization)
- 블러 제거(de-blurring)
- 비디오 생성(video generation)
과 같은 분야에서 사용.
![[20.4.3.png]]