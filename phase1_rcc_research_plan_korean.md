# 1단계 연구 계획: 재귀적 캐스케이드 압축(RCC) 구현

## 개요

본 연구 계획은 Vision-Language Model에 대한 재귀적 캐스케이드 압축(RCC) 구현을 상세히 다룹니다. >99.5% 파라미터 압축을 달성하면서 95% 성능을 유지하는 것을 목표로 합니다. 7주간의 일정을 통해 DARE, Nullu, AlphaEdit 압축 기법의 캐스케이딩을 통한 시너지 효과를 체계적으로 탐구하며, 각 단계마다 엄격한 실험 프로토콜과 통계적 검증을 수행합니다.

## 연구 가설

### 주요 가설
DARE→Nullu→AlphaEdit의 순차적 적용은 상호 보완적인 null space 활용을 통해 곱셈적 압축 효과를 생성하여, 원래 VLM 능력의 95%를 보존하면서 >99.5% 총 압축을 달성한다.

### 세부 가설
1. DARE의 비구조적 가지치기는 Nullu의 랭크 감소 효율성을 향상시키는 희소 활성화 패턴을 생성한다
2. 압축 방법 간 Null space 중첩은 캐스케이드 시너지를 가능하게 하는 멱법칙 분포를 따른다
3. AlphaEdit의 적응형 가중치는 이전 캐스케이드 단계의 정보 손실을 보상한다
4. 최적 캐스케이드 순서는 작업별 가중치 분포 패턴에 따라 달라진다

## 주차별 상세 계획

### 1주차: 기반 구축
**월-화요일**: 환경 구성
- PyTorch 2.0, transformers, CLIP 설치
- CUDA 11.8, 혼합 정밀도 학습 구성
- 실험 추적 설정 (W&B/MLflow)

**수-목요일**: DARE 구현
```python
class DARECompressor:
    """
    구조적/비구조적 변형을 포함한 DARE 구현
    참조: [Zhang2023, Algorithm 1]
    """
    def __init__(self, drop_rate_schedule, rescale_method='uniform'):
        self.schedule = drop_rate_schedule  # [0.9, 0.95, 0.99]
        self.rescale = rescale_method

    def compress(self, weights, structure='unstructured'):
        # 크기 기반 가지치기 및 재조정
        # 가중치 분포 통계 유지
        mask = self._magnitude_pruning(weights)
        pruned = weights * mask
        rescaled = self._rescale(pruned, mask)
        return rescaled
```

**금요일**: 데이터 파이프라인
- MS-COCO 데이터로더 및 캐싱 구현
- 전처리 파이프라인 최적화
- 검증 분할 생성

### 2주차: 기준선 방법
**월-화요일**: Nullu 구현
```python
class NulluProjector:
    """
    SVD/QR 분해를 통한 랭크 감소
    참조: [Wang2024, Section 3.3]
    """
    def __init__(self, target_rank_ratio=0.95):
        self.rank_ratio = target_rank_ratio

    def project(self, weight_matrix, method='svd'):
        if method == 'svd':
            U, S, V = torch.svd(weight_matrix)
            k = int(len(S) * (1 - self.rank_ratio))
            S_reduced = S.clone()
            S_reduced[-k:] = 0
            return U @ torch.diag(S_reduced) @ V.T
        elif method == 'qr':
            Q, R = torch.qr(weight_matrix)
            # QR 기반 랭크 감소 로직
            return self._qr_reduction(Q, R)
```

**수-목요일**: AlphaEdit 통합
- Task vector 추출 구현
- 가중치 보간 프레임워크
- 다중 작업 산술 검증

**금요일**: 통합 테스트
- 엔드투엔드 파이프라인 검증
- 성능 벤치마킹
- 메모리 프로파일링

### 3주차: 기준선 검증
**월-화요일**: DARE 검증
- 발표된 결과와 비교 (99% 압축률 재현)
- Drop rate에 대한 민감도 분석
- 성능 곡선 문서화

**수-목요일**: Nullu 검증
- 랭크 감소 효과성 측정
- 재구성 오류 분석
- 원본 논문과 교차 검증

**금요일**: 통합 기준선
- 성능 지표 집계
- 통계 분석
- 기준선 보고서 생성

### 4주차: 캐스케이드 아키텍처
**월-화요일**: 파이프라인 프레임워크
```python
class RCCPipeline:
    """
    체크포인팅을 포함한 캐스케이드 압축 조율
    """
    def __init__(self, compressors, checkpoint_dir):
        self.stages = compressors
        self.checkpoints = []
        self.null_space_analyzer = NullSpaceAnalyzer()

    def cascade_compress(self, model, validate_intermediate=True):
        compressed = model
        for i, compressor in enumerate(self.stages):
            # 각 단계에서 null space 중첩 분석
            if i > 0:
                overlap = self.null_space_analyzer.compute_overlap(
                    self.checkpoints[-1], compressed
                )
                print(f"Null space 중첩: {overlap:.4f}")

            compressed = compressor(compressed)

            if validate_intermediate:
                perf = self.validate(compressed)
                if perf < self.threshold:
                    print(f"단계 {i}에서 성능 저하 감지")
                    return self.checkpoints[-1]

            self.checkpoints.append(compressed.clone())

        return compressed
```

**수-목요일**: Null space 분석
- Grassmann 거리 계산 구현
- 중첩 시각화 도구
- 공간 진화 추적

**금요일**: 초기 실험
- 첫 캐스케이드 시도 (DARE→Nullu)
- 성능 모니터링
- 디버깅 및 최적화

### 5주차: 전체 파이프라인
**월-화요일**: 완전한 캐스케이드
```python
def full_cascade_experiment():
    """3단계 통합 실험"""
    model = load_pretrained_vlm("clip-vit-b32")

    # 압축 모듈 초기화
    dare = DARECompressor(drop_schedule=[0.9, 0.95, 0.99])
    nullu = NulluProjector(target_rank_ratio=0.95)
    alpha = AlphaEditor(alpha_schedule='adaptive')

    # 캐스케이드 파이프라인 구성
    pipeline = RCCPipeline(
        compressors=[dare, nullu, alpha],
        checkpoint_dir="./checkpoints"
    )

    # 압축 실행
    compressed_model = pipeline.cascade_compress(
        model,
        validate_intermediate=True
    )

    # 결과 분석
    compression_rate = compute_compression_rate(model, compressed_model)
    performance = evaluate_performance(compressed_model)

    return {
        'compression_rate': compression_rate,
        'performance_retention': performance,
        'checkpoints': pipeline.checkpoints
    }
```

**수-목요일**: 스케줄링 시스템
- 압축률 스케줄링
- 적응형 임계값 설정
- 조기 종료 기준

**금요일**: 시스템 검증
- 엔드투엔드 테스트
- 성능 벤치마크
- 병목 현상 식별

### 6주차: 최적화 단계
**월-화요일**: 베이지안 최적화
```python
from skopt import gp_minimize

def objective(params):
    """최적화 목표 함수"""
    dare_rate, nullu_rank, alpha_weight = params

    # 파라미터로 압축 실행
    result = run_cascade_with_params(dare_rate, nullu_rank, alpha_weight)

    # 다중 목표 최적화: 압축률과 성능
    compression_score = result['compression_rate']
    performance_score = result['performance_retention']

    # 가중 합산 (압축률 60%, 성능 40%)
    return -(0.6 * compression_score + 0.4 * performance_score)

# 베이지안 최적화 실행
result = gp_minimize(
    func=objective,
    dimensions=[
        (0.85, 0.99),  # DARE drop rate
        (0.90, 0.98),  # Nullu rank ratio
        (0.1, 1.0)     # Alpha weight
    ],
    n_calls=50,
    random_state=42
)
```

**수-목요일**: 순서 실험
- 24개 순열 시도 (3! × 4 압축 수준)
- 병렬 실행 설정
- 결과 집계

**금요일**: 분석
- 파레토 프론티어 구축
- 최적 구성 선택
- 성능 시각화

### 7주차: 절제 연구 및 검증
**월-화요일**: 구성 요소 절제
```python
def ablation_study():
    """각 구성 요소의 필요성 검증"""
    configurations = [
        ['dare', 'nullu', 'alpha'],  # 전체
        ['dare', 'nullu'],           # AlphaEdit 제거
        ['dare', 'alpha'],           # Nullu 제거
        ['nullu', 'alpha'],          # DARE 제거
        ['dare'],                     # DARE만
        ['nullu'],                    # Nullu만
        ['alpha']                     # AlphaEdit만
    ]

    results = {}
    for config in configurations:
        pipeline = create_pipeline(config)
        result = evaluate_pipeline(pipeline)
        results['-'.join(config)] = result

    # 통계적 유의성 테스트
    perform_statistical_tests(results)
    return results
```

**수-목요일**: 통계적 검증
- 유의성 테스트 (paired t-test, α=0.05)
- 신뢰 구간 (bootstrap, n=1000)
- 교차 검증

**금요일**: 최종 문서화
- 결과 편집
- 재현성 패키지
- 기술 보고서 초안

## 핵심 구현 세부사항

### 1. DARE 구현 세부사항
```python
class DAREImplementation:
    def __init__(self):
        self.drop_strategies = {
            'uniform': self._uniform_drop,
            'magnitude': self._magnitude_drop,
            'structured': self._structured_drop
        }

    def _magnitude_drop(self, weights, drop_rate):
        """크기 기반 드롭"""
        threshold = torch.quantile(weights.abs(), drop_rate)
        mask = weights.abs() > threshold
        return mask

    def _rescale(self, weights, mask):
        """드롭 후 재조정"""
        keep_prob = mask.float().mean()
        return weights / keep_prob if keep_prob > 0 else weights
```

### 2. Nullu 통합 세부사항
```python
class NulluIntegration:
    def identify_hallucination_space(self, model, samples):
        """환각 공간 식별"""
        activations = []
        for sample in samples:
            act = model.get_intermediate_activations(sample)
            activations.append(act)

        # SVD로 주요 환각 방향 추출
        A = torch.stack(activations)
        U, S, V = torch.svd(A)

        # 상위 k개 방향 선택 (95% 분산 설명)
        cumsum = S.cumsum(0) / S.sum()
        k = (cumsum < 0.95).sum() + 1

        return V[:, :k]
```

### 3. AlphaEdit 최적화
```python
class AlphaEditOptimization:
    def adaptive_alpha_schedule(self, epoch, val_loss):
        """적응형 알파 스케줄링"""
        if val_loss > self.prev_loss:
            # 손실 증가 시 알파 감소
            self.alpha *= 0.9
        else:
            # 손실 감소 시 알파 증가
            self.alpha = min(self.alpha * 1.1, 1.0)

        self.prev_loss = val_loss
        return self.alpha
```

## 실험 프로토콜

### 24가지 캐스케이드 순서 실험
```python
import itertools

methods = ['dare', 'nullu', 'alpha']
compression_levels = [0.9, 0.95, 0.99, 0.995]

# 모든 순열 생성
orderings = list(itertools.permutations(methods))

# 각 압축 수준에 대해 실험
for level in compression_levels:
    for ordering in orderings:
        experiment_name = f"{'-'.join(ordering)}_compression_{level}"
        run_cascade_experiment(
            ordering=ordering,
            target_compression=level,
            save_name=experiment_name
        )
```

### Null Space 중첩 분석
```python
def analyze_null_space_overlap(W1, W2):
    """Grassmann 거리를 사용한 중첩 분석"""
    # 각 가중치 행렬의 null space 계산
    U1, _, _ = torch.svd(W1)
    U2, _, _ = torch.svd(W2)

    # Grassmann 거리 계산
    M = U1.T @ U2
    _, S, _ = torch.svd(M)

    # 주각도 계산
    angles = torch.acos(torch.clamp(S, -1, 1))

    # 거리 메트릭
    distance = torch.norm(angles)
    overlap = 1 - (distance / (np.pi * np.sqrt(len(angles))))

    return {
        'grassmann_distance': distance.item(),
        'overlap_ratio': overlap.item(),
        'principal_angles': angles.cpu().numpy()
    }
```

## 평가 메트릭

### 압축률 측정
```python
def measure_compression_rate(original_model, compressed_model):
    """정확한 압축률 계산"""
    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(
        (p != 0).sum().item() for p in compressed_model.parameters()
    )

    compression_rate = 1 - (compressed_params / original_params)

    return {
        'compression_rate': compression_rate * 100,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'reduction_factor': original_params / compressed_params
    }
```

### 성능 유지 평가
```python
def evaluate_performance_retention(original_model, compressed_model, test_loader):
    """다중 작업에서 성능 유지율 평가"""
    tasks = {
        'imagenet_zeroshot': evaluate_imagenet_zeroshot,
        'vqa': evaluate_vqa,
        'caption_generation': evaluate_caption,
        'clip_score': evaluate_clip_alignment
    }

    results = {}
    for task_name, eval_func in tasks.items():
        orig_score = eval_func(original_model, test_loader)
        comp_score = eval_func(compressed_model, test_loader)
        retention = comp_score / orig_score * 100

        results[task_name] = {
            'original': orig_score,
            'compressed': comp_score,
            'retention': retention
        }

    avg_retention = np.mean([r['retention'] for r in results.values()])
    results['average_retention'] = avg_retention

    return results
```

## 위험 완화 전략

### 1. 캐스케이드 불안정성 방지
```python
class StabilityMonitor:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.history = []

    def check_stability(self, metrics):
        """각 단계에서 안정성 확인"""
        if len(self.history) > 0:
            degradation = self.history[-1] - metrics
            if degradation > self.threshold:
                return False, f"성능 저하 {degradation:.2%} 감지"

        self.history.append(metrics)
        return True, "안정적"
```

### 2. 경사도 소실 해결
```python
def gradient_rescaling(model):
    """경사도 재조정으로 소실 방지"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm < 1e-7:
                param.grad *= 1e7 / max(grad_norm, 1e-10)
```

### 3. 성능 저하 임계값
```python
PERFORMANCE_THRESHOLDS = {
    'minimum_retention': 0.95,  # 95% 최소 유지
    'maximum_compression': 0.997,  # 99.7% 최대 압축
    'stability_margin': 0.02  # 2% 안정성 마진
}
```

## 성공 기준

### 정량적 목표
- **압축률**: >99% (목표: 99.5%)
- **성능 유지**: >95% (ImageNet zero-shot)
- **추론 지연**: <5ms 증가
- **재현성**: 3개 시드에서 일관된 결과

### 검증 체크포인트
1. **3주차 끝**: 개별 방법 기준선 확립
2. **5주차 끝**: 작동하는 캐스케이드 파이프라인
3. **6주차 끝**: 최적화된 하이퍼파라미터
4. **7주차 끝**: 완전한 절제 연구 및 통계 검증

## 데이터 파이프라인 상세

### MS-COCO 처리
```python
class COCODataPipeline:
    def __init__(self, root_dir, split='train'):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def preprocess_batch(self, batch):
        """배치 전처리 최적화"""
        images = torch.stack([self.transform(img) for img in batch['images']])
        texts = self.tokenizer(batch['captions'], padding=True, truncation=True)
        return images, texts
```

### Conceptual Captions 필터링
```python
def filter_high_quality_pairs(dataset, threshold=0.7):
    """고품질 이미지-텍스트 쌍 필터링"""
    filtered = []
    for item in dataset:
        # CLIP 점수로 품질 평가
        score = compute_clip_score(item['image'], item['text'])
        if score > threshold:
            filtered.append(item)

    return filtered[:500000]  # 상위 500K 선택
```

## 예상 결과

### 정량적 결과
- 압축률: 99.5-99.7% 파라미터 감소
- 성능 유지: ImageNet zero-shot에서 94-96%
- 추론 속도: 단일 GPU에서 8-12배 향상
- 메모리 감소: 95% 사용량 감소

### 과학적 기여
- 트리플 캐스케이드 압축의 첫 시연
- Null space 중첩 정량화 방법론
- 최적 캐스케이드 순서 가이드라인
- 오픈소스 RCC 구현 프레임워크

## 결론

본 1단계 연구 계획은 재귀적 캐스케이드 압축(RCC)의 체계적 구현을 위한 상세한 로드맵을 제공합니다. 7주간의 집중적인 개발과 실험을 통해 >99.5% 압축률을 달성하면서도 95% 성능을 유지하는 혁신적인 압축 기법을 구현하고 검증할 것입니다.

---

*작성일: 2025년 1월 13일*
*연구 팀: AI Research Team*
*버전: 1.0 (한국어판)*