# 🤖 RL Dojo: 3-Environment Benchmark & Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-lightgrey.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-blue.svg)

## 1. 프로젝트 요약 (Executive Summary)
본 프로젝트는 강화학습 에이전트가 단일 환경의 튜토리얼 수준을 넘어, 점진적으로 고도화되는 3단계 도장깨기(Blackjack -> CliffWalking -> Taxi)를 하나의 범용 알고리즘으로 정복한 포트폴리오입니다. 각 환경이 가진 고유한 병목 지점(튜플 상태, 극단적 패널티, 대규모 상태 공간)을 엔지니어링적으로 정의하고 해결하는 과정을 데이터로 증명합니다.

## 2. 핵심 엔지니어링 마일스톤
1. **Universal Hash-based Q-Table:** 정수형 상태(Taxi, Cliff)와 튜플형 상태(Blackjack)를 별도의 전처리 없이 수용하는 해시 기반 에이전트 설계.
2. **Domain-Specific Reward Shaping:** 각 도메인의 병목 지점을 타겟팅하여 보상을 재설계함으로써 학습 안정성 확보.
3. **Progressive Benchmarking:** 기초 확률 모델부터 복잡한 계층적 행동 모델까지 점진적인 벤치마킹을 수행하여 알고리즘의 범용성(Generalization) 입증.

## 3. 실험 환경 및 단계별 도전 과제 (Dojang-Breaking Stages)
| 도장깨기 순서 | 환경 (Environment) | 도전 과제 (Challenge) | 상태 공간 특성 | 학습 목표 |
| :---: | :--- | :--- | :--- | :--- |
| **Stage 1** | **Blackjack-v1** | 확률적 불확실성 (Stochastic) | (Sum, Card, Ace) 튜플 | 최적의 카드 드로우 임계점 도출 |
| **Stage 2** | **CliffWalking-v0** | 극단적 절벽 추락 패널티 | 48 (Integer) | 위험 회피 및 최단 경로 학습 |
| **Stage 3** | **Taxi-v3** | 복잡한 계층적 행동 순서 | 500 (Integer) | 효율적인 승객 운송 시퀀스 학습 |

## 4. 프로젝트 구조
    ├── src/             
    │   └── main.py      # 범용 에이전트 및 3단계 환경 통합 자동화 루프
    ├── notebooks/       
    │   ├── 1. 김도윤_도장깨기_Blackjack_실습.ipynb
    │   ├── 2. 김도윤_도장깨기_CliffWalking_실습.ipynb
    │   └── 3. 김도윤_도장깨기_Taxi_실습.ipynb
    ├── results/         # 3개 환경별 자동 생성된 학습 결과 그래프
    ├── requirements.txt 
    └── README.md        

## 5. 실행 방법
    pip install -r requirements.txt
    python src/main.py

## 6. 실험 결과 및 데이터 기반 분석

### 📈 Stage 1. Blackjack-v1: 확률적 승부의 최적화
![Blackjack Result](results/Blackjack-v1_alpha_0.1.png)
* **인사이트:** 다차원 튜플 상태 공간을 유연하게 수용하는 범용 에이전트를 성공적으로 안착시켰습니다. 딜러의 오픈 카드와 플레이어의 합계를 비교하여 'Stay'와 'Hit'의 수학적 경계선을 학습하며, 확률적 불확실성 속에서도 보상이 안정적인 우상향을 그리는 것을 확인했습니다.

### 📈 Stage 2. CliffWalking-v0: 절벽 패널티 조정 (Reward Shaping)
![CliffWalking Result](results/CliffWalking-v0_alpha_0.1.png)
* **인사이트:** 기존의 가혹한 추락 패널티(-100)를 -10으로 스케일링하여, 에이전트가 과도한 공포를 느끼지 않고 위험한 경계면(절벽 옆)을 타고 넘는 최단 경로 정책(Optimal Path)을 수렴시켰습니다.

### 📈 Stage 3. Taxi-v3: 대규모 상태 공간 제어 (Scale-up)
![Taxi Result](results/Taxi-v3_alpha_0.1.png)
* **인사이트:** 이동 패널티를 -1에서 -1.1로 미세 강화하여 승객을 태우지 않고 배회하는 시간을 억제했습니다. 이를 통해 상태 공간이 500개로 팽창한 복잡한 환경에서도 빠르고 안정적으로 최종 목표에 도달하는 엔지니어링 튜닝의 힘을 입증했습니다.

## 7. 향후 과제 (Future Work)
현재의 Table 기반 접근법은 500개의 상태(Taxi)까지는 무리 없이 소화하지만, 영상 픽셀 등을 입력받는 연속적 상태 공간에서는 필연적으로 **상태 폭발(State Explosion)**을 겪게 됩니다. 향후 프로젝트는 PyTorch를 활용한 **DQN(Deep Q-Network)** 알고리즘으로 모델을 확장하여 딥러닝 기반의 가치 근사(Approximation) 능력을 검증할 계획입니다.

## 8. 회고 및 성장 포인트 (Retrospective)
이번 3단계 도장깨기 프로젝트는 제 엔지니어링 사고방식을 절차적 스크립터에서 시스템 아키텍트로 진화시켜 준 결정적 계기였습니다.

* **튜플부터 500 차원까지, 데이터의 본질을 다루다:** 첫 단계인 Blackjack의 튜플 상태를 마주했을 때, 하드코딩으로 예외 처리를 하는 대신 `defaultdict` 기반의 범용 해시 Q-Table을 고안했습니다. 이 기초 공사 덕분에 이어지는 CliffWalking과 Taxi 환경에서도 코드 수정 없이 에이전트를 즉시 이식할 수 있었습니다. 이는 확장성을 고려한 시니어급 아키텍처 설계가 실무에서 얼마나 강력한 생산성을 내는지 깨닫는 순간이었습니다.

* **통제된 패널티가 만들어낸 지능:** CliffWalking의 절벽과 Taxi의 배차 문제. 겉보기엔 전혀 다른 두 난제를 관통하는 핵심은 '보상의 균형'이었습니다. 에이전트에게 너무 가혹한 패널티(-100)는 학습을 마비시키고, 너무 관대한 보상은 방황을 유발했습니다. 수백 번의 실험 끝에 최적의 패널티 수치를 찾아내어 모델의 수렴 곡선을 띄워 올렸을 때의 전율은 잊을 수 없습니다. 인공지능은 맹목적인 수식의 나열이 아니라, 엔지니어가 섬세하게 통제한 보상 생태계 위에서 피어나는 결과물임을 몸소 체득했습니다.

* **준비된 실무형 인재의 마인드셋:** 저는 이제 단일 튜토리얼을 따라 치는 주니어가 아닙니다. 데이터의 형태에 구애받지 않고, 환경의 병목을 찾아내 보상을 설계하며, 하이퍼파라미터로 시스템 리소스를 최적화하는 과정을 완전히 내재화했습니다. 곧 다가올 6~7월 실무 현장에서도, 이 집요한 실험 정신을 바탕으로 어떤 복잡한 비즈니스 환경의 난제라도 뚫어낼 준비가 되었습니다. 제 닉네임 AD처럼, 어떤 조건에서도 가장 최적화된 경로(Optimal Path)를 찾아내는 엔지니어로서의 가치를 증명하겠습니다.
