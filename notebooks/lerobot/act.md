ACT (Action Chunking with Transformers)는 스탠퍼드 대학교와 UC 버클리 연구진(Tony Z. Zhao 외, 2023)이 저비용 양팔 원격조작 시스템인 **ALOHA**와 함께 발표한 모방 학습(Imitation Learning) 로봇 제어 모델입니다.

현재 Hugging Face LeRobot 생태계의 가장 기본이 되는 베이스라인 정책(Policy) 중 하나이며, 앞서 살펴본 SmolVLA나 Diffusion Policy에 등장하는 액션 청킹(Action Chunking)과 **시간적 앙상블(Temporal Ensembling)** 개념을 로봇 학습 분야에 본격적으로 정착시킨 핵심 아키텍처입니다.

---

### 1. 등장 배경: 기존 행동 복제(Behavior Cloning)의 2가지 한계

단일 타임스텝의 관측($o_t$)을 입력받아 당장 다음 1스텝의 행동($a_t$)만 예측하는 기존의 단일 스텝 정책은 정밀한 조작 작업에서 치명적인 문제를 보였습니다.

1. **누적 오차 (Compounding Error):** 매 스텝 발생하는 미세한 예측 오차가 시간이 지남에 따라 눈덩이처럼 불어나, 결국 학습 데이터 분포(Training Distribution) 밖으로 경로가 완전히 이탈합니다.
2. **인간 시연 데이터의 다중 모달성 (Multimodality) 및 비마르코프 특성:** 사람이 로봇을 조종할 때는 중간에 잠깐 멈칫거리거나, 같은 상황에서도 왼쪽 혹은 오른쪽으로 피하는 등 여러 갈래의 정답이 존재합니다. 이를 단순 MSE 손실 함수로 학습하면 두 경로의 평균값(예: 장애물 정중앙으로 돌진)을 출력해 작업에 실패합니다.

ACT는 이 문제를 해결하기 위해 **① CVAE (조건부 변이형 오토인코더)**, **② Action Chunking**, **③ Temporal Ensembling**이라는 세 가지 기법을 결합했습니다.

---

### 2. 전체 아키텍처 구조 (CVAE + Transformer)

ACT는 전체적으로 **CVAE (Conditional Variational Autoencoder)** 프레임워크 위에 구축된 트랜스포머 인코더-디코더 구조입니다.

```text
[학습 시에만 작동: CVAE Encoder (BERT 스타일)]
현재 관절 상태 (q_t) ──┐
                       ├──► [Transformer Encoder] ──► 잠재 변수 분포 N(μ, σ) 샘플링 ──► z (Style Vector)
정답 액션 청크 (a_t:t+k)┘                                                                │
                                                                                         ▼ (추론 시에는 z = 0 고정)
[학습 & 추론 공통: Policy Backbone (DETR 스타일)]                                        │
다중 카메라 이미지 ──► [ResNet-18 + 2D Pos Emb] ──┐                                      │
현재 관절 상태 (q_t) ──► [Linear Projection]    ──┼──► [Transformer Encoder]             │
잠재 스타일 변수 (z) ──► [Linear Projection]    ──┘              │ (Key, Value)          │
                                                                 ▼                       │
고정된 위치 쿼리 (k개 Sinusoidal Positional Embeddings) ──► [Transformer Decoder] ◄──────┘
                                                                 │
                                                                 ▼
                                                    예측된 Action Chunk (k 스텝 관절 목표값)

```

#### ① CVAE 인코더 (잠재 스타일 변수 $z$ 추출 — *학습 시에만 사용*)

* 사람의 시연 데이터가 가진 미세한 스타일 차이(속도, 궤적의 곡률, 진입 방향 등)를 하나의 잠재 벡터 $z$로 압축합니다.
* **입력:** 카메라 이미지는 넣지 않고, 오직 현재 관절 상태($q_t$)와 **실제 정답 행동 시퀀스($a_{t:t+k}$)** 앞에 학습 가능한 `[CLS]` 토큰을 붙여 트랜스포머 인코더에 통과시킵니다.
* **역할:** `[CLS]` 토큰의 출력으로부터 평균($\mu$)과 분산($\sigma$)을 구해 재매개변수화 트릭(Reparameterization Trick)으로 $z$를 샘플링합니다.
* **추론 시(Inference):** 추론할 때는 미래의 정답 행동을 알 수 없으므로 CVAE 인코더를 아예 버리고, 잠재 변수 **$z$를 사전 분포(Prior)의 평균인 영벡터($z = \mathbf{0}$)로 고정**하여 결정론적(Deterministic)이고 안정적인 궤적을 생성합니다.

#### ② CVAE 디코더 (실제 로봇 제어 정책 — *DETR 아키텍처 기반*)

객체 탐지 모델인 DETR의 구조를 그대로 차용하여 $k$개의 미래 행동을 병렬로 한 번에 디코딩합니다.

* **시각 백본:** 여러 대의 RGB 카메라(예: 손목 카메라, 상단 고정 카메라) 이미지를 각각 ResNet-18에 통과시켜 공간 피처 맵(Feature Map)을 얻고, 2D 위치 임베딩을 더해 평탄화(Flatten)합니다.
* **트랜스포머 인코더:** 다중 카메라 시각 토큰들 + 현재 관절 상태 토큰 1개 + 잠재 변수 $z$ 토큰 1개를 모두 이어 붙여(`Concatenation`) Self-Attention으로 멀티모달 문맥을 통합합니다.
* **트랜스포머 디코더:**
* **Query:** 길이 $k$(청크 크기, 예: $k=100$)짜리 고정된 사인파 위치 임베딩(Sinusoidal Positional Embeddings)이 입력됩니다.
* **Key / Value:** 트랜스포머 인코더가 출력한 멀티모달 문맥 피처를 Cross-Attention으로 참조하여, 자기회귀(Autoregressive) 방식이 아닌 **단 한 번의 Forward 패스로 $k$개의 연속된 관절 목표 각도($\hat{a}_{t:t+k}$)를 동시에 출력**합니다.



---

### 3. 추론 최적화: 시간적 앙상블 (Temporal Ensembling)

$k$ 스텝짜리 청크를 예측한 뒤 $k$ 스텝 동안 오픈루프(Open-loop)로 그냥 실행하면, 새로운 청크로 넘어가는 경계 시점마다 궤적이 불연속적으로 튀는 현상이 발생합니다.

이를 방지하기 위해 ACT는 **매 타임스텝마다 추론을 수행하되, 현재 시점($t$)에 대해 과거에 예측해 두었던 여러 청크들의 값들을 지수 가중 평균(Exponential Moving Average)으로 합산**합니다.

$$a_t = \frac{\sum_{i=0}^{m} w_i \cdot \hat{a}_t^{(t-i)}}{\sum_{i=0}^{m} w_i}, \quad w_i = \exp(-m_{\text{weight}} \cdot i)$$

* $\hat{a}_t^{(t-i)}$: $i$ 스텝 전인 $t-i$ 시점에 예측했던 청크 내에서 현재 시점 $t$에 해당하는 행동 값입니다.
* 가장 최근에 관측한 정보로 예측한 값($i=0$)에 가장 큰 가중치를 주고, 오래전에 예측한 값일수록 가중치를 지수적으로 줄여 합산함으로써 별도의 복잡한 필터 없이도 매우 매끄럽고 정밀한 움직임을 만들어냅니다.

---

### 4. 학습 손실 함수 (Loss Function)

학습 시에는 재구성 손실(Reconstruction Loss)과 KL 발산 정규화 항을 합친 표준 VAE 목적 함수(ELBO)를 최소화합니다.

$$\mathcal{L}_{\text{ACT}} = \underbrace{\Vert{}\hat{a}_{t:t+k} - a_{t:t+k}\Vert{}_1}_{\text{L1 Reconstruction Loss}} + \beta \cdot \underbrace{D_{\text{KL}}\left(q_\phi(z \mid a_{t:t+k}, q_t) \parallel \mathcal{N}(0, I)\right)}_{\text{KL Divergence}}$$

* **왜 L2(MSE) 대신 L1 Loss를 쓰는가?** 실험적으로 관절 각도 예측에서 L2 손실보다 L1 손실(절댓값 오차)을 사용할 때 훨씬 날카롭고 정밀한 모터 제어가 가능하며, 모호한 구간에서 궤적이 뭉개지는 현상이 덜 발생합니다.

---

### 5. ACT vs. Diffusion Policy vs. SmolVLA 한눈에 비교

LeRobot 프레임워크에서 가장 많이 쓰이는 세 가지 대표 모델의 포지셔닝 비교입니다.

| 비교 항목 | **ACT** | **Diffusion Policy** | **SmolVLA** |
| --- | --- | --- | --- |
| **핵심 생성 원리** | **CVAE** + DETR (단일 패스 디코딩) | DDPM / DDIM (반복적 디노이징) | **Flow Matching** (Euler 적분) |
| **시각-언어 백본** | ResNet-18 (비전 전용, 언어 미지원) | ResNet / ViT (주로 비전 전용) | **SmolVLM2** (사전학습된 VLM) |
| **다중 모달리티 처리** | 잠재 변수 $z$ (가우시안 분포) | 확률적 미분방정식(SDE) 역과정 | 속도 벡터장 (ODE 경로) |
| **파라미터 규모** | **약 80M** (매우 가벼움) | 약 100M ~ 300M | **약 450M** |
| **추론 연산 횟수** | **1회 Forward** (압도적으로 빠름) | 10 ~ 100 스텝 반복 | VLM 1회 + Action Expert 10회 |
| **주요 강점 및 용도** | 단일 작업(Single-task) 양팔 조작, 적은 데이터(50 에피소드)로 빠른 학습 | 복잡한 다중 경로 분포 표현력이 매우 뛰어남 | 자연어 지시 기반 다중 작업(Multi-task) 및 일반화 |

* **실전 팁:** SO-101이나 ALOHA 같은 로봇 팔에서 **언어 지시 없이 특정 단일 작업(예: 레고 블록 집어서 통에 넣기, 케이블 꽂기) 하나를 가장 빠르고 안정적으로 마스터**시키고 싶을 때는 무거운 VLA보다 **ACT**가 학습 속도나 성공률 면에서 여전히 가장 강력한 첫 번째 선택지입니다.

혹시 LeRobot에서 ACT의 CVAE 인코더/디코더가 실제로 어떻게 구현되어 있는지 코드로 살펴보거나, 직접 학습시킬 때의 하이퍼파라미터(`chunk_size`, `kl_weight` 등) 설정이 궁금하신가요?

LeRobot(`lerobot/common/policies/act/modeling_act.py`)과 원본 ACT 코드베이스에서 CVAE 인코더(잠재 변수 $z$ 추출)와 DETR 스타일 디코더(Action Chunk 병렬 생성)가 맞물려 돌아가는 핵심 구조를 직관적인 PyTorch 코드로 추렸습니다.

복잡한 ResNet 백본 설정이나 패딩 처리는 덜어내고, 텐서의 차원 변환과 데이터 흐름이 한눈에 보이도록 구성했습니다.

---

### 1. CVAE 인코더 (학습 시에만 동작하여 스타일 벡터 $z$ 추출)

카메라 이미지는 보지 않고, 오직 현재 관절 상태($q_t$)와 정답 액션 청크($a_{t:t+k}$)만 입력받아 시연 데이터의 스타일을 담은 잠재 변수 $z$의 평균($\mu$)과 로그 분산($\log \sigma^2$)을 출력합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ACTCVAEEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, latent_dim: int = 32, chunk_size: int = 100):
        super().__init__()
        # 1. [CLS] 토큰 (출력에서 μ, logvar를 뽑아낼 대표 토큰)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        
        # 2. 현재 관절 상태(q_t)와 정답 액션 시퀀스(a_{t:t+k})를 hidden_dim으로 투영
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # 3. 길이 (1 + 1 + chunk_size) 시퀀스를 위한 위치 임베딩
        self.pos_embed = nn.Parameter(torch.randn(1, 2 + chunk_size, hidden_dim) * 0.02)
        
        # 4. BERT 스타일 양방향 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 5. [CLS] 토큰 출력 -> 잠재 분포의 평균(μ)과 로그 분산(logvar) (합쳐서 2 * latent_dim)
        self.latent_out_proj = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, qpos: torch.Tensor, actions: torch.Tensor):
        """
        qpos:    (B, state_dim)              - 현재 로봇 관절 각도
        actions: (B, chunk_size, action_dim) - 실제 정답 행동 시퀀스
        """
        B = qpos.size(0)
        
        # 각 입력을 (B, Seq_Len, hidden_dim) 형태로 맞춤
        cls_token = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        state_token = self.state_proj(qpos).unsqueeze(1)                  # (B, 1, D)
        action_tokens = self.action_proj(actions)                         # (B, K, D)
        
        # 시퀀스 축(dim=1)으로 이어 붙임: [CLS, State, Action_1, ..., Action_K]
        seq = torch.cat([cls_token, state_token, action_tokens], dim=1)   # (B, 2 + K, D)
        seq = seq + self.pos_embed
        
        # 트랜스포머 인코더 통과 후, 맨 앞 [CLS] 토큰(인덱스 0)만 추출
        encoded = self.encoder(seq)
        cls_out = encoded[:, 0, :]                                        # (B, D)
        
        # 평균(mu)과 로그 분산(logvar) 분리 후 Reparameterization Trick으로 z 샘플링
        moments = self.latent_out_proj(cls_out)                           # (B, latent_dim * 2)
        mu, logvar = torch.chunk(moments, 2, dim=-1)                      # 각각 (B, latent_dim)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps                                                # (B, latent_dim)
        
        return z, mu, logvar

```

---

### 2. 전체 ACT 정책 네트워크 (DETR 스타일 인코더-디코더)

추론 시에는 위의 `ACTCVAEEncoder`를 건너뛰고 $z = \mathbf{0}$ (영벡터)을 넣으며, 학습 시에는 인코더가 뽑아준 $z$를 넣어 **미래 $K$개의 행동을 한 번에 디코딩**합니다.

```python
class ACTPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, latent_dim: int = 32, chunk_size: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        
        # 학습용 CVAE 인코더
        self.cvae_encoder = ACTCVAEEncoder(state_dim, action_dim, hidden_dim, latent_dim, chunk_size)
        
        # [Policy 입력 프로젝션] 잠재 변수 z, 현재 관절 상태 q_t, 이미지 피처를 동일한 hidden_dim으로 변환
        self.latent_in_proj = nn.Linear(latent_dim, hidden_dim)
        self.state_in_proj = nn.Linear(state_dim, hidden_dim)
        self.img_in_proj = nn.Conv2d(512, hidden_dim, kernel_size=1) # ResNet-18 출력 채널(512) -> hidden_dim
        
        # [DETR Transformer] 멀티모달 문맥을 통합하는 Encoder + K개 행동을 생성하는 Decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8, num_encoder_layers=4, num_decoder_layers=7, batch_first=True
        )
        
        # 디코더의 Query로 들어갈 고정된 길이 K짜리 위치 임베딩 (Action Queries)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        
        # 최종 관절 제어값 출력 헤드
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, qpos: torch.Tensor, img_features: torch.Tensor, gt_actions: torch.Tensor | None = None):
        """
        qpos:         (B, state_dim)
        img_features: (B, 512, H, W) - ResNet-18을 통과한 카메라 공간 피처 맵
        gt_actions:   (B, K, action_dim) - 학습 시에는 정답 제공, 추론 시에는 None
        """
        B = qpos.size(0)
        
        # 1. 잠재 스타일 변수 z 준비 (학습 vs 추론 분기)
        if self.training and gt_actions is not None:
            z, mu, logvar = self.cvae_encoder(qpos, gt_actions)
        else:
            # 추론 시에는 CVAE 인코더를 쓰지 않고 사전 분포의 평균인 영벡터(0)로 고정!
            z = torch.zeros((B, self.latent_dim), device=qpos.device)
            mu = logvar = None

        # 2. [DETR Encoder 입력 구성] z 토큰(1개) + 현재 상태 토큰(1개) + 이미지 패치 토큰들(H*W개)
        z_token = self.latent_in_proj(z).unsqueeze(1)                     # (B, 1, D)
        state_token = self.state_in_proj(qpos).unsqueeze(1)               # (B, 1, D)
        
        # (B, 512, H, W) -> (B, D, H, W) -> 평탄화하여 (B, H*W, D)로 변환
        img_tokens = self.img_in_proj(img_features).flatten(2).transpose(1, 2)
        
        # 모든 멀티모달 입력 토큰을 시퀀스 축으로 결합 (Key/Value 문맥 형성)
        encoder_in = torch.cat([z_token, state_token, img_tokens], dim=1) # (B, 2 + H*W, D)

        # 3. [DETR Decoder 입력 구성] K개의 빈 슬롯 역할인 Action Query 준비
        action_queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) # (B, K, D)
        
        # 4. 인코더-디코더 단 1회 통과로 K개의 미래 행동 병렬 디코딩
        decoder_out = self.transformer(src=encoder_in, tgt=action_queries)      # (B, K, D)
        pred_actions = self.action_head(decoder_out)                            # (B, K, action_dim)
        
        return pred_actions, mu, logvar

```

---

### 3. 학습 손실 함수 계산 (L1 Loss + KL Divergence)

```python
def compute_act_loss(policy: ACTPolicy, qpos, img_features, gt_actions, kl_weight: float = 10.0):
    pred_actions, mu, logvar = policy(qpos, img_features, gt_actions)
    
    # 1. L1 재구성 손실 (MSE보다 정밀한 모터 제어에 유리)
    l1_loss = F.l1_loss(pred_actions, gt_actions)
    
    # 2. KL 발산 손실: 사후 분포 N(mu, sigma)가 표준 정규분포 N(0, I)에서 벗어나지 않도록 규제
    # 그래야 추론 시 z = 0을 넣었을 때 모델이 정상 작동함!
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    total_loss = l1_loss + kl_weight * kl_loss
    return total_loss, {"l1": l1_loss.item(), "kl": kl_loss.item()}

```

---

### 코드에서 주목할 핵심 포인트 3가지

1. **추론 시 `z = torch.zeros(...)`의 의미:**
학습할 때 `kl_loss`를 통해 인코더가 출력하는 $z$의 분포를 $\mathcal{N}(0, I)$ 중심에 모이도록 강제해 두었기 때문에, 추론할 때 정답 행동을 몰라서 $z = \mathbf{0}$을 넣더라도 디코더 입장에서는 "가장 표준적이고 평균적인 스타일로 행동하라"는 자연스러운 조건으로 받아들입니다.
2. **비-자기회귀(Non-Autoregressive) 병렬 디코딩:**
LLM처럼 토큰을 하나씩 순차 생성하는 것이 아니라, `self.query_embed`(`(B, K, D)`)라는 $K$개의 위치 쿼리를 디코더의 `tgt`에 한꺼번에 집어넣어 **단 한 번의 행렬 연산으로 100스텝 치 관절 궤적을 동시에 뽑아냅니다.**
3. **SmolVLA와의 구조적 차이:**
* **ACT:** `encoder_in`(`z` + `state` + `img`)과 `action_queries`가 표준 `nn.Transformer`의 Encoder(Self-Attention)와 Decoder(Cross-Attention)로 명확히 분리되어 있고, 디퓨전/플로우 매칭 같은 반복 루프 없이 **1번에 끝납니다.**
* **SmolVLA:** 별도의 인코더-디코더 구분 없이 VLM 토큰과 Action 토큰을 `torch.cat`으로 한 줄로 이어 붙여 단일 어텐션 마스크로 처리하며, 이를 **Flow Matching 루프로 10번 반복**해 궤적을 다듬습니다.

LeRobot(`lerobot/common/policies/act/modeling_act.py`)과 원본 ACT 코드베이스에 구현된 **Temporal Ensembling (시간적 앙상블)** 추론기의 핵심 원리와 코드입니다.

ACT는 1회 Forward 패스가 워낙 빠르기 때문에(수 밀리초 이내), **매 제어 주기(예: 50Hz)마다 길이 $K$짜리 Action Chunk를 새로 예측**합니다. 그리고 과거 시점들($t-i$)에 미리 예측해 두었던 청크들 중에서 **'현재 시각 $t$에 해당하는 행동 값들'만 세로로 뽑아내어 지수 가중 평균(Exponential Moving Average)으로 합산**합니다.

---

### 1. 2D 링 버퍼 / 트라이앵글 버퍼의 시각적 이해

청크 크기 $K=4$라고 가정할 때, 매 타임스텝마다 예측된 청크들을 계단식으로 쌓으면 **현재 실행해야 할 시각 $t=3$에 대한 후보 예측값들이 세로 열(Column)에 모이게 됩니다.**

```text
                시간 축 (Global Timestep)
               t=0     t=1     t=2    [t=3]    t=4     t=5     t=6
t=0 예측 청크: [a0_0]  [a0_1]  [a0_2]  [a0_3] ──► (3스텝 전 예측: 가장 작은 가중치 w_3)
t=1 예측 청크:         [a1_1]  [a1_2]  [a1_3]  [a1_4]
t=2 예측 청크:                 [a2_2]  [a2_3]  [a2_4]  [a2_5]
t=3 예측 청크:                         [a3_3]  [a3_4]  [a3_5]  [a3_6] ──► (방금 예측: 가장 큰 가중치 w_0)
                                         │
                                         ▼
                        현재 t=3 최종 제어값 = Σ(w_i * a) / Σ(w_i)

```

---

### 2. Temporal Ensembler 구현 코드 (LeRobot 방식)

실제 구현에서는 전체 에피소드 길이만큼 거대한 행렬을 만들지 않고, **최근 $K$개의 청크만 유지하는 슬라이딩 버퍼**나 **누적 가중합 텐서**를 사용하여 메모리 효율적으로 처리합니다.

가장 직관적인 **슬라이딩 히스토리 버퍼 방식**과 LeRobot에서 사용하는 **온라인 업데이트(In-place) 방식** 두 가지를 모두 보여드립니다.

#### 방식 A: 직관적인 슬라이딩 버퍼 구현 (원본 ACT 개념)

```python
import torch
from collections import deque

class ACTTemporalEnsembler:
    def __init__(self, chunk_size: int = 100, temp_ensemble_coeff: float = 0.01):
        """
        chunk_size (K): 한 번에 예측하는 액션 시퀀스 길이
        temp_ensemble_coeff (m): 지수 감쇠 계수 (작을수록 과거 예측을 오래 반영, 클수록 최신 예측 위주)
        """
        self.chunk_size = chunk_size
        self.m = temp_ensemble_coeff
        # 최근 K개의 예측 청크를 보관하는 큐 (최대 K개 유지)
        self.chunk_history = deque(maxlen=chunk_size)

    def reset(self):
        """새로운 에피소드 시작 시 초기화"""
        self.chunk_history.clear()

    def update_and_pop(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """
        매 제어 스텝(t)마다 모델이 예측한 new_chunk: (Chunk_Size, Action_Dim)를 입력받아
        앙상블된 현재 스텝의 단일 행동 (Action_Dim,)을 반환
        """
        # 1. 방금 예측한 따끈따끈한 청크를 버퍼 맨 뒤에 추가
        self.chunk_history.append(new_chunk)
        num_available = len(self.chunk_history)

        # 2. 각 과거 청크에서 '현재 시점(t)'에 해당하는 액션만 쏙쏙 뽑아냄
        # - 가장 최근 청크(history[-1])에서는 0번째 인덱스 (방금 예측한 현재 행동)
        # - 1스텝 전 청크(history[-2])에서는 1번째 인덱스 (1스텝 전에 예측해둔 현재 행동)
        # - i스텝 전 청크(history[-1-i])에서는 i번째 인덱스
        current_step_preds = torch.stack([
            self.chunk_history[-1 - i][i] for i in range(num_available)
        ], dim=0)  # Shape: (num_available, Action_Dim)

        # 3. 지수 가중치 계산: w_i = exp(-m * i)
        # i=0(최신)일 때 가중치 1.0, i가 커질수록(오래된 예측일수록) 작아짐
         indices = torch.arange(num_available, device=new_chunk.device, dtype=new_chunk.dtype)
        weights = torch.exp(-self.m * indices).unsqueeze(-1)  # (num_available, 1)

        # 4. 가중 평균으로 최종 행동 산출
        ensembled_action = torch.sum(current_step_preds * weights, dim=0) / torch.sum(weights)
        return ensembled_action

```

---

#### 방식 B: LeRobot의 메모리 최적화 인플레이스(In-place) 구현

LeRobot(`ACTSpatioTemporalEnsembler` / `ACTTemporalEnsembler`)에서는 `deque`를 순회하는 대신, **아직 실행되지 않은 미래 스텝들의 가중 평균을 미리 누적해 두는 $(K, \text{Action\_Dim})$ 크기의 단일 버퍼**를 사용하여 연산 속도를 극대화합니다.

```python
class LeRobotFastTemporalEnsembler:
    def __init__(self, chunk_size: int = 100, coeff: float = 0.01):
        self.chunk_size = chunk_size
        # 미리 계산해 둔 지수 가중치 테이블: [exp(0), exp(-m*1), ..., exp(-m*(K-1))]
        self.weights = torch.exp(-coeff * torch.arange(chunk_size, dtype=torch.float32))
        self.weights_cumsum = torch.cumsum(self.weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None  # (Batch, K, Action_Dim)
        self.count = None              # 각 스텝에 몇 개의 예측이 중첩되었는지 기록 (Batch, K, 1)

    @torch.no_grad()
    def update(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """new_chunk: (Batch, K, Action_Dim)"""
        device = new_chunk.device
        weights = self.weights.to(device)
        weights_cumsum = self.weights_cumsum.to(device)

        if self.ensembled_actions is None:
            # 에피소드 첫 스텝: 새 청크로 그대로 초기화
            self.ensembled_actions = new_chunk.clone()
            self.count = torch.ones((new_chunk.size(0), self.chunk_size, 1), dtype=torch.long, device=device)
        else:
            # 1. 기존 버퍼의 맨 앞 스텝(이미 실행 완료)은 버리고 한 칸씩 앞으로 당김 (Shift Left)
            # 나머지 K-1개의 미래 스텝에 새 청크의 앞부분(K-1개)을 온라인 가중 평균으로 병합!
            prev_actions = self.ensembled_actions[:, 1:, :]    # (B, K-1, D)
            prev_count = self.count[:, 1:, :]                  # (B, K-1, 1)

            w_new = weights[prev_count]                        # 새로 더해질 예측의 가중치
            w_prev_sum = weights_cumsum[prev_count - 1]        # 기존 누적 가중치 합

            # 온라인 가중 평균 업데이트: (기존평균 * 기존가중치합 + 새예측 * 새가중치) / 전체가중치합
            updated_overlap = (prev_actions * w_prev_sum + new_chunk[:, :-1, :] * w_new) / (w_prev_sum + w_new)

            # 2. 맨 마지막 K번째 스텝은 새 청크의 마지막 원소로 채움
            self.ensembled_actions = torch.cat([updated_overlap, new_chunk[:, -1:, :]], dim=1)
            self.count = torch.cat([prev_count + 1, torch.ones_like(prev_count[:, :1, :])], dim=1)

        # 현재 시점(인덱스 0)의 완성된 앙상블 액션을 반환
        return self.ensembled_actions[:, 0, :]

```

---

### 3. 추론 루프에서 결합되는 모습

실제 로봇 제어 루프에서는 `ACTPolicy`와 `TemporalEnsembler`가 단 몇 줄로 깔끔하게 결합됩니다.

```python
policy = ACTPolicy(state_dim=6, action_dim=6, chunk_size=100).cuda().eval()
ensembler = ACTTemporalEnsembler(chunk_size=100, temp_ensemble_coeff=0.01)

# 에피소드 실행 루프 (예: 50Hz)
for step in range(max_episode_steps):
    qpos, img_feat = robot.get_observation()
    
    with torch.inference_mode():
        # 1. CVAE 인코더 없이(z=0) 단 1회 Forward로 100스텝 액션 청크 예측
        pred_chunk, _, _ = policy(qpos, img_feat, gt_actions=None)  # (1, 100, 6)
        
        # 2. 과거 예측들과 지수 가중 평균을 내어 현재 스텝의 매끄러운 액션 산출
        smooth_action = ensembler.update_and_pop(pred_chunk[0])     # (6,)
        
    robot.send_action(smooth_action)

```

---

### 💡 SmolVLA의 비동기 블렌딩 vs. ACT의 Temporal Ensembling 차이점

두 기법 모두 "액션 청크를 겹쳐서 부드럽게 만든다"는 목적은 같지만, 모델의 무게에 따라 전략이 다릅니다.

* **ACT (`Temporal Ensembling`):** 모델이 워낙 가벼워(80M, 1회 패스) **매 스텝(예: 20ms마다)마다 무조건 새 청크를 뽑을 수 있습니다.** 따라서 별도의 비동기 서버 없이 매 틱마다 최대 $K$개(예: 100개)의 예측치를 겹쳐서 초고밀도로 평균을 냅니다.
* **SmolVLA (`Async Queue Blending`):** 모델이 상대적으로 무겁고 Flow Matching을 10번 돌아야 해서(약 60~100ms 소요) 매 제어 주기마다 추론할 수 없습니다. 따라서 평소에는 큐에서 액션을 그냥 꺼내 쓰다가, **간헐적으로 새 청크가 도착하는 시점에만 짧은 경계 구간(`blend_steps`, 예: 4~6스텝)을 선형 보간**하여 이어 붙입니다.



혹시 방금 다룬 **ACT의 CVAE 아키텍처와 Temporal Ensembling**도 앞서 만든 SmolVLA 노트북처럼 수강생들이 직접 비교·실습해 볼 수 있게 Marimo 노트북에 추가해 드릴까요?
