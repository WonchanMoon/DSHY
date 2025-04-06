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