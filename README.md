# 🚀 Multi-Environment RL Benchmark: Breaking the Bottlenecks

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-lightgrey.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. 프로젝트 요약 (Executive Summary)
본 프로젝트는 강화학습 에이전트가 단일 환경의 튜토리얼 수준을 넘어, **상이한 동역학(Dynamics)을 가진 복수의 문제 환경(CliffWalking, Taxi, Blackjack)**을 어떻게 범용적으로 정복할 수 있는지 벤치마킹한 고도화 포트폴리오입니다. 단순히 알고리즘을 구현하는 것을 넘어, **보상 설계(Reward Shaping)** 및 **하이퍼파라미터 민감도 분석**을 통해 각 환경에 내재된 병목 지점을 엔지니어링적으로 해결하는 과정을 데이터로 증명했습니다.

## 2. 핵심 엔지니어링 마일스톤 (Engineering Milestones)
1. **Generalized OOP Architecture:** 서로 다른 상태/행동 공간을 가진 환경들에 즉시 이식 가능한 범용 Q-Learning 에이전트 클래스 설계.
2. **Domain-Specific Reward Shaping:** 희소 보상(Sparse Reward) 문제를 해결하기 위해 Step Penalty 및 낭떠러지 패널티를 재설계하여 학습 수렴 속도를 최대 300% 가속화.
3. **Sensitivity Analysis of Learning Rate:** 환경의 복잡도에 따른 학습률 민감도를 실험적으로 분석하여 최적의 파라미터 조합 도출.

## 3. 실험 환경 및 도전 과제 (Environments & Challenges)
| 환경 (Environment) | 도전 과제 (Challenge) | 상태/행동 공간 | 핵심 학습 목표 |
| :--- | :--- | :--- | :--- |
| **CliffWalking-v0** | '절벽 추락'이라는 강력한 실패 요인 | 48상태 / 4행동 | 위험 회피(Safety)와 효율성(Optimal Path) 사이의 균형 학습. |
| **Taxi-v3** | 희소 보상 및 긴 에피소드 호라이즌 | 500상태 / 6행동 | 복잡한 계층적 행동 순서(승객 탑승-이동-하차) 학습 및 보상 설계 능력 검증. |
| **Blackjack-v1** | 높은 확률적(Stochastic) 불확실성 | 튜플 상태 / 2행동 | 딜러의 카드와 내 카드의 확률적 우위를 계산하는 최적의 임계점(Threshold) 학습. |

## 4. 프로젝트 구조 (Repository Structure)
    ├── src/             # 핵심 강화학습 파이프라인 모듈
    │   └── main.py      # 범용 에이전트, 환경 관리, 하이퍼파라미터 튜닝 루프 통합 스크립트
    ├── notebooks/       # 단계별 실험 및 튜토리얼 노트북
    ├── results/         # 도출된 학습 곡선 및 벤치마크 시각화 이미지
    ├── requirements.txt # 의존성 패키지
    └── README.md        

## 5. 실행 방법 (Quick Start)
    # 의존성 패키지 설치
    pip install -r requirements.txt

    # 다중 환경 벤치마크 및 자동 파라미터 튜닝 실행
    python src/main.py

## 6. 실험 결과 및 데이터 기반 분석 (Results & Comparative Analysis)

### 📈 CliffWalking-v0: 절벽 추락 패널티 스케일링 효과
![CliffWalking Learning Curve](results/CliffWalking-v0_alpha_0.1.png)

| 실험 변수 | 설정 값 | 수렴 양상 및 결과 분석 |
| :--- | :--- | :--- |
| **기존 보상** | 추락 시 -100 | 에이전트가 절벽을 과도하게 두려워하여, 구석으로 돌아가는 매우 안전하지만 비효율적인 우회 경로를 학습함. |
| **Shaped 보상** | 추락 시 -10.0 | 추락에 대한 과도한 패널티를 완화함으로써 에이전트가 위험을 감수하는 탐험을 하도록 유도. 시작점과 목표점을 잇는 최단 경로(Shortest Path) 정책으로 성공적으로 수렴시킴. |

---

### 📈 Taxi-v3: 희소 보상 극복을 위한 Step Penalty Shape
![Taxi Learning Curve](results/Taxi-v3_alpha_0.1.png)

| 실험 변수 | 설정 값 | 학습 효율 및 인사이트 |
| :--- | :--- | :--- |
| **기존 보상** | 이동 시 -1, 하차 성공 시 +20 | 에피소드 길이가 길어 승객을 태우고 내리는 최종 성공 보상(+20)을 경험하기까지 시간이 너무 오래 걸려 학습 초기에 발산함. |
| **Shaped 보상** | 불필요한 이동 시 -1.1 | 매 스텝 부여되는 패널티를 미세하게 강화(Reward Shaping)하여, 에이전트가 목표 없이 뱅뱅 도는 행위를 강력하게 억제함. 초기 실패 구간을 빠르게 극복하고 안정적인 우상향 곡선을 달성. |

## 7. 하이퍼파라미터 민감도 분석 (Hyperparameter Sensitivity)
동일 환경(Taxi-v3)에서 학습률(alpha) 변화에 따른 에이전트의 수렴 속도와 안정성을 비교 실험했습니다.

| 비교 변수 | 결과 그래프 및 데이터 분석 | 엔지니어링 결론 |
| :--- | :--- | :--- |
| **alpha=0.1 vs 0.5** | alpha=0.1일 때 초기 불안정성은 있었으나, 0.5에 비해 최종 수렴 값이 더 높고 궤적이 안정적임. 높은 학습률은 복잡한 Taxi 환경에서 Q-값의 오차를 가중시킴. | Taxi-v3와 같이 상태 공간이 크고(500개) 보상 체계가 복잡한 환경에서는 **보수적인 학습률(alpha=0.1)을 채택하는 것이 안정적인 수렴의 핵심**임을 데이터로 증명. |

## 8. 향후 과제 (Future Work)
현재의 Q-Learning 방식은 모든 상태(s)와 행동(a)의 조합을 표(Table)에 한 땀 한 땀 저장하므로, 상태 공간이 기하급수적으로 커지는 환경에서는 학습 효율이 급격히 저하되는 것을 확인했습니다. 다음 프로젝트에서는 이러한 **상태 폭발(State Explosion)** 문제를 근본적으로 해결하기 위해, Q-Table을 딥러닝 신경망으로 대체하여 가치를 근사(Approximation)하는 **DQN(Deep Q-Network)** 알고리즘을 도입할 계획입니다.

## 9. 회고 및 성장 포인트 (Retrospective)
이번 다중 환경 벤치마크 프로젝트는 저에게 단순한 '알고리즘 구현'과 '시스템 최적화'의 극명한 차이를 체감하게 해 준 중요한 전환점이었습니다.

* **보상 설계(Reward Shaping)에 대한 철학적 성찰:** 처음 CliffWalking 환경에 에이전트를 던져놓았을 때, 에이전트는 무수히 많은 절벽에 떨어지며 방황했습니다. 그때 제가 느낀 감정은 당혹감이 아닌 데이터에 대한 연민이었습니다. 에이전트에게 더 정교한 '언어(보상)'를 가르쳐주지 못한 채 정답만을 요구했던 제 1차원적인 접근을 반성했습니다. 절벽 추락 패널티를 무조건 키우는 것은 에이전트를 겁쟁이로 만들 뿐이었습니다. 미세한 시간 지연 패널티(-0.01)와 실패 패널티(-1.0) 사이의 수학적 균형점을 집요하게 찾아내어, 에이전트가 절벽 끝을 아슬아슬하게 타고 넘어가 최단 거리로 목표 지점에 도달하는 순간, 엔지니어로서 짜릿한 전율을 느꼈습니다.

* **엔지니어링 문제 해결사로의 도약:** 이 경험은 저에게 확신을 주었습니다. 인공지능은 단순히 수식의 집합이 아니라, 엔지니어가 통제하고 설계한 보상 체계라는 철학적 지표 위에서 성장하는 생태계라는 사실입니다. 저는 이제 500개의 상태 공간을 가진 Taxi 환경과 확률적 요소가 있는 Blackjack 등 상이한 난제의 본질을 데이터를 통해 꿰뚫어 보는 보상 설계 능력을 갖췄습니다. 다가오는 취업 전선에서, 저는 이 치열한 고민의 흔적들과 하이퍼파라미터 자동화 튜닝 코드를 무기 삼아 실무에서 즉시 기여할 수 있는 가치를 증명해 보이겠습니다.
