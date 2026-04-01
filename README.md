# 🤖 Multi-Environment RL Benchmark: 3-Environment Benchmark & Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-lightgrey.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-blue.svg)

## 📌 프로젝트 요약 (Executive Summary)
본 프로젝트는 강화학습 에이전트가 단일 환경의 튜토리얼 수준을 넘어, 점진적으로 고도화되는 3단계 도장깨기(Blackjack -> CliffWalking -> Taxi)를 하나의 범용 알고리즘으로 정복한 포트폴리오입니다. 각 환경이 가진 고유한 병목 지점(튜플 상태, 극단적 패널티, 대규모 상태 공간)을 엔지니어링적으로 정의하고 해결하는 과정을 데이터로 증명합니다.

## 🎯 핵심 목표 (Motivation)
1. **Universal Hash-based Q-Table:** 정수형 상태(Taxi, Cliff)와 튜플형 상태(Blackjack)를 별도의 전처리 없이 수용하는 해시 기반 에이전트 설계.
2. **Domain-Specific Reward Shaping:** 각 도메인의 병목 지점을 타겟팅하여 보상을 재설계함으로써 학습 안정성 확보.
3. **Hyperparameter Sensitivity Analysis:** 학습률(Alpha) 변화에 따른 수렴 속도와 안정성을 모든 환경에서 시각적으로 교차 검증.

## 1. 실험 환경 및 단계별 도전 과제 (Dojang-Breaking Stages)
| 도장깨기 순서 | 환경 (Environment) | 도전 과제 (Challenge) | 상태 공간 특성 | 학습 목표 |
| :---: | :--- | :--- | :--- | :--- |
| **Stage 1** | **Blackjack-v1** | 확률적 불확실성 (Stochastic) | (Sum, Card, Ace) 튜플 | 최적의 카드 드로우 임계점 도출 |
| **Stage 2** | **CliffWalking-v1** | 극단적 절벽 추락 패널티 | 48 (Integer) | 위험 회피 및 최단 경로 학습 |
| **Stage 3** | **Taxi-v3** | 복잡한 계층적 행동 순서 | 500 (Integer) | 효율적인 승객 운송 시퀀스 학습 |

## 2. 프로젝트 구조
    ├── src/             
    │   └── main.py      # 범용 에이전트 및 3단계 환경 통합 자동화 루프
    ├── notebooks/       
    │   ├── 1. 김도윤_도장깨기_Blackjack_실습.ipynb
    │   ├── 2. 김도윤_도장깨기_CliffWalking_실습.ipynb
    │   └── 3. 김도윤_도장깨기_Taxi_실습.ipynb
    ├── results/         # 6개의 환경/파라미터별 학습 결과 시각화 그래프
    ├── requirements.txt 
    └── README.md        

## 3. 실행 방법
    pip install -r requirements.txt
    python src/main.py

## 4. 실험 결과 및 하이퍼파라미터 민감도 분석 (Results & Sensitivity Analysis)
각 환경에서 동일한 모델에 대해 학습률(`Alpha=0.1` vs `Alpha=0.5`)을 다르게 적용하여, 모델의 수렴 안정성을 교차 검증했습니다.

### 📈 Stage 1. Blackjack-v1: 확률적 승부의 최적화
| Alpha = 0.1 (안정적 수렴) | Alpha = 0.5 (높은 변동성) |
| :---: | :---: |
| ![Blackjack 0.1](results/Blackjack-v1_alpha_0.1.png) | ![Blackjack 0.5](results/Blackjack-v1_alpha_0.5.png) |

* **엔지니어링 인사이트:** 딜러의 오픈 카드와 플레이어의 합계를 비교하는 튜플 상태 공간을 성공적으로 제어했습니다. 특히 확률적 변동성이 극심한 환경 특성상, 학습률이 높은 0.5 모델은 정책이 크게 요동치는 반면, 0.1 모델은 안정적인 우상향 수렴 곡선(약 +0.2 도달)을 그려내는 것을 데이터로 증명했습니다.

### 📈 Stage 2. CliffWalking-v1: 절벽 패널티 조정 (Reward Shaping)
| Alpha = 0.1 (최적 경로 도출) | Alpha = 0.5 (초기 불안정) |
| :---: | :---: |
| ![CliffWalking 0.1](results/CliffWalking-v1_alpha_0.1.png) | ![CliffWalking 0.5](results/CliffWalking-v1_alpha_0.5.png) |

* **엔지니어링 인사이트:** 가혹한 추락 패널티(-100)를 -10으로 스케일링(Reward Shaping)하여 절벽 옆을 과감하게 타고 넘는 최단 경로를 유도했습니다. 두 파라미터 모두 결국 -13 부근의 최적 값에 도달했으나, 0.1 모델이 학습 초기 구간에서 낭비되는 에피소드를 줄이고 훨씬 견고한 'L자형' 수렴 속도를 보였습니다.

### 📈 Stage 3. Taxi-v3: 대규모 상태 공간 제어 (Scale-up)
| Alpha = 0.1 (강력한 우상향) | Alpha = 0.5 (발산 위험 발견) |
| :---: | :---: |
| ![Taxi 0.1](results/Taxi-v3_alpha_0.1.png) | ![Taxi 0.5](results/Taxi-v3_alpha_0.5.png) |

* **엔지니어링 인사이트:** 이동 패널티를 강화(-1.1)하여 500개의 복잡한 상태 공간에서도 배회 시간을 억제했습니다. 0.5 모델은 복잡한 상태 간 가치 전파 과정에서 Q-값이 튕겨 나가는 발산 위험을 보여주었으나, 0.1 모델을 적용함으로써 초기의 실패 구간을 빠르게 극복하고 안정적인 고점(+8 이상)에 안착시키는 튜닝 능력을 입증했습니다.

## 5. 향후 과제 (Future Work)
현재의 Table 기반 접근법은 500개의 상태(Taxi)까지는 무리 없이 소화하지만, 영상 픽셀 등을 입력받는 연속적 상태 공간에서는 필연적으로 **상태 폭발(State Explosion)**을 겪게 됩니다. 향후 프로젝트는 PyTorch를 활용한 **DQN(Deep Q-Network)** 알고리즘으로 모델을 확장하여 딥러닝 기반의 가치 근사(Approximation) 능력을 검증할 계획입니다.

## 6. 💡 회고록 (Retrospective)
이번 3단계 도장깨기 프로젝트는 제 엔지니어링 사고방식을 절차적 스크립터에서 시스템 아키텍트로 진화시켜 준 결정적 계기였습니다.

* **튜플부터 500 차원까지, 데이터의 본질을 다루다:** 첫 단계인 Blackjack의 튜플 상태를 마주했을 때, 하드코딩으로 예외 처리를 하는 대신 `defaultdict` 기반의 범용 해시 Q-Table을 고안했습니다. 이 기초 공사 덕분에 이어지는 CliffWalking과 Taxi 환경에서도 코드 수정 없이 에이전트를 즉시 이식할 수 있었습니다. 이는 확장성을 고려한 아키텍처 설계가 실무에서 얼마나 강력한 생산성을 내는지 깨닫는 순간이었습니다.

* **통제된 패널티와 하이퍼파라미터가 만들어낸 지능:** CliffWalking의 절벽과 Taxi의 배차 문제. 겉보기엔 전혀 다른 두 난제를 관통하는 핵심은 '수치적 균형'이었습니다. 수백 번의 파라미터 튜닝(Alpha 0.1 vs 0.5)과 패널티 조정을 반복하며 모델의 수렴 곡선을 비교 분석했습니다. 인공지능은 맹목적인 코딩의 산물이 아니라, 엔지니어가 섬세하게 통제한 환경과 데이터 생태계 위에서 피어나는 결과물임을 6개의 비교 그래프를 통해 몸소 체득했습니다.

* **준비된 실무형 인재의 마인드셋:** 저는 이제 단일 튜토리얼을 따라 치는 수준을 넘어섰습니다. 데이터의 형태에 구애받지 않고, 환경의 병목을 찾아내 보상을 설계하며, 하이퍼파라미터 벤치마킹으로 시스템을 최적화하는 전 과정을 완전히 내재화했습니다. 6~7월 실무 현장에서도 이 집요한 데이터 중심의 엔지니어링 마인드를 바탕으로 어떤 복잡한 난제라도 뚫어낼 준비가 되었습니다. 제 닉네임 AD처럼, 어떤 조건에서도 가장 통찰력 있는 분석과 최적화된 경로(Optimal Path)를 찾아내는 가치를 증명하겠습니다.
