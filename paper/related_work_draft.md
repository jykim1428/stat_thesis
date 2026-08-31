## 2. Related Work

### 2.1 포트폴리오 최적화와 강화학습

전통적 포트폴리오 이론은 기대수익과 위험의 균형을 통해 자산 비중을 결정하는 평균-분산 최적화에서 출발하였다 [1]. 그러나 이 접근은 수익률 분포와 공분산 추정의 안정성에 의존하며, 시장 상태 변화와 거래비용을 동적인 의사결정 과정으로 직접 표현하기 어렵다. 강화학습(reinforcement learning)은 상태 관측, 행동 선택, 보상이라는 순차적 의사결정 구조로 포트폴리오 리밸런싱을 모델링할 수 있으므로, 이러한 한계를 보완하는 방법으로 연구되어 왔다.

암호자산 시장을 대상으로 한 초기 연구에서 Jiang and Liang [2]은 과거 가격을 입력으로 받아 자산 비중을 직접 출력하는 심층 강화학습 기반 포트폴리오 관리 방식을 제안하였다. 이후 Jiang, Xu, and Liang [3]은 CNN, RNN, LSTM 기반 정책을 포함하는 model-free 프레임워크를 제시하고, 거래비용이 포함된 암호자산 백테스트에서 여러 포트폴리오 선택 전략과 비교하였다. 이러한 연구는 가격 예측과 포트폴리오 의사결정을 분리하지 않고, 누적 포트폴리오 성과를 목표로 비중을 학습할 수 있음을 보였다.

한편 FinRL은 데이터, 거래 환경, 강화학습 알고리즘, 백테스트를 모듈화한 오픈소스 프레임워크를 제공하며, 거래비용과 시장 환경 등을 포함한 금융 거래 실험을 구성할 수 있는 기반을 제공한다 [4]. 본 연구는 FinRL의 `PortfolioOptimizationEnv`를 연구 데이터와 공통 출력 형식에 맞게 적용하여, 동일한 환경에서 서로 다른 정책망을 비교한다.

### 2.2 PPO 기반 연속 비중 결정

포트폴리오 비중은 연속적인 행동 공간으로 표현할 수 있다. Proximal Policy Optimization (PPO)은 정책 업데이트의 크기를 clipping 목적함수로 제한하는 on-policy actor--critic 알고리즘으로, 하나의 수집된 rollout에 대해 여러 epoch의 minibatch 최적화를 수행할 수 있다 [5]. Schulman et al. [5]은 PPO가 trust-region 계열 방법의 장점을 일부 유지하면서도 구현과 최적화를 단순화하는 것을 목표로 제안하였다.

본 연구에서는 PPO 자체의 알고리즘적 개선을 제안하지 않는다. 대신 동일한 PPO 설정, 행동 공간, 보상 구조 및 백테스트 절차를 유지하고, 최종 평가 단계에서 동일한 거래비용 가정을 적용한 상태에서 정책의 feature extractor만 MLP와 Transformer로 달리한다. 따라서 관측 시계열을 표현하는 방식의 차이가 포트폴리오 성과와 위험 지표에 미치는 영향을 비교할 수 있다.

### 2.3 Transformer를 이용한 시계열 표현

Transformer는 self-attention을 이용해 순서열 내 원소 간의 관계를 모델링하는 구조로, 순환 또는 합성곱만으로 구성된 구조와 달리 모든 시점 쌍 사이의 의존성을 직접 참조할 수 있다 [6]. 이후 시계열 예측 분야에서는 장기 의존성과 여러 입력 변수의 상호작용을 다루기 위해 Transformer 계열 구조가 활용되어 왔다. 예를 들어 Temporal Fusion Transformer는 지역적 처리와 self-attention을 결합해 다중 시계열의 시간적 관계를 학습하는 구조를 제안하였다 [7].

다만 본 연구의 Transformer는 미래 가격을 직접 예측하는 예측기가 아니라, 최근 `TIME_WINDOW` 구간의 가격 feature와 자산 차원을 PPO 정책이 사용할 표현 벡터로 변환하는 feature extractor다. 따라서 본 연구는 Transformer가 모든 금융 시계열 문제에서 우월하다고 가정하지 않는다. 동일한 관측 창과 PPO 학습 조건에서 MLP 기준선 대비 Transformer 표현이 서로 다른 시장 국면에 얼마나 안정적으로 일반화하는지를 검증한다.

### 2.4 암호자산 백테스트와 시간 순서 기반 검증

암호자산 시장은 높은 변동성과 비정상성으로 인해 특정 기간의 백테스트 성과를 일반화하기 어렵다. Gort et al. [8]은 암호자산 심층 강화학습에서 백테스트 과적합이 false positive로 이어질 수 있음을 지적하고, 과적합을 고려한 검증의 필요성을 제시하였다. 따라서 시간 순서를 무시하는 무작위 셔플 검증은 실제 운용 상황에서의 미래 예측 문제를 평가하는 데 적합하지 않다.

본 연구는 이러한 문제를 고려하여 시장 국면의 시간적 순서를 보존하는 expanding-window 방식의 forward evaluation을 사용한다. 하이퍼파라미터 선택은 초기 학습 구간과 이후 validation 구간을 이용하여 수행하며, 선택된 설정은 이후 평가 구간에 대해 고정한다. 또한 학습 데이터에 미래의 평가 구간이 포함되지 않도록 각 fold에서 시간 순서를 엄격하게 유지한다.

이와 같은 평가 설계를 통해 단일 기간의 백테스트 성과에 의존하지 않고, 서로 다른 시점의 out-of-sample 구간에서 정책의 일반화 성능을 확인하는 것을 목표로 한다.

### 2.5 본 연구의 위치

기존 연구는 심층 강화학습을 이용한 암호자산 포트폴리오 관리의 가능성을 제시했고 [2, 3], PPO는 연속적인 포트폴리오 비중을 학습하기 위한 실용적인 정책 최적화 방법을 제공한다 [5]. Transformer 계열 연구는 복잡한 시계열 의존성을 표현할 수 있는 구조적 근거를 제공하지만 [6, 7], 이러한 표현력 증가가 금융 시계열의 out-of-sample 성과 향상으로 직접 이어지는지는 별도의 실증적 검증이 필요하다.

이에 본 연구는 (1) 동일한 암호자산 포트폴리오 환경과 공통 평가 스펙에서 MLP 기반 PPO와 Transformer 기반 PPO를 비교하고, (2) Buy & Hold 및 전통적 포트폴리오 전략을 포함한 비교 기준을 구성하며, (3) 거래비용과 위험조정 성과지표를 함께 보고하고, (4) 국면의 시간적 순서를 보존한 expanding-window OOS 평가를 통해 정책의 시간적 일반화 가능성을 검증한다.

본 연구의 기여는 새로운 PPO 알고리즘이나 Transformer 구조 자체를 제안하는 데 있지 않다. 대신 동일한 강화학습 환경과 평가 절차에서 정책의 feature extraction 방식에 따른 성능 차이를 비교하고, 전통적 포트폴리오 전략 및 단순 벤치마크와의 상대적 성과를 함께 평가함으로써 Transformer 기반 포트폴리오 정책의 실증적 위치를 검토하는 데 있다.

## 참고문헌

1. Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77--91. https://doi.org/10.2307/2975974

2. Jiang, Z., & Liang, J. (2017). *Cryptocurrency Portfolio Management with Deep Reinforcement Learning*. arXiv:1612.01277. https://arxiv.org/abs/1612.01277

3. Jiang, Z., Xu, D., & Liang, J. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*. arXiv:1706.10059. https://arxiv.org/abs/1706.10059

4. Liu, X.-Y., Yang, H., Gao, J., Wang, C. D., et al. (2021). *FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance*. arXiv:2111.09395. https://arxiv.org/abs/2111.09395

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/abs/1707.06347

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 30. https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

7. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748--1764. https://arxiv.org/abs/1912.09363

8. Gort, B. J. D., Liu, X.-Y., Sun, X., Gao, J., Chen, S., & Wang, C. D. (2022). *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting*. arXiv:2209.05559. https://arxiv.org/abs/2209.05559