# 리뷰 대응 실험과 A30 실험 정리 (2026-10-21 ~ 10-22)

이 문서는 리뷰 (2)·(3)에 대응하려고 NVIDIA Thor에서 돌린 실험과, 일반화를 확인하려고 CloudLab A30에서 돌린 실험을 한곳에 모은 것이다. 실험마다 질문, 구성, 결과, 논문 반영 위치, 데이터 위치를 적었다. 숫자는 각 결과 폴더의 요약 파일에서 옮겼다.

## 1. 한눈에 보기

| # | 실험 | 플랫폼 | 질문 | 핵심 결과 | 논문 반영 |
|---|---|---|---|---|---|
| T1 | Fault-around (126) | Thor | 4 KiB 페이지 캐시에서 populate 뒤에도 GPU 폴트가 남는 이유는? | 기본값에서 475,648개가 남고, fault-around를 1페이지로 줄이면 6/6 실행에서 0 | 평가 4.3 문장 |
| T2 | 인계 전체 (128-130) | Thor | 복사 파일을 읽기 전용으로 매핑하는 방식(Copied File)과 STATOR의 차이는? 쓰기 금지 파일의 비용은? | 멈춤 Copy 574.5 / Copied File 82.7 / STATOR 11.8 ms, immutable 파일 19.1 ms, 쓰기 18/18 거부 | Figure 14, 평가 4.4, 설계 3.3, Table 2 |
| T3 | 반복 fork (131, 133) | Thor | 실행 중인 에이전트를 여러 번 fork할 때의 멈춤과 메모리는? | fork당 Copy 527-626 / Copied File 81-92 / STATOR 5.7-12.2 ms, 정점 메모리 9.7-12.9 / 7.9-8.1 / 6.9-7.0 GiB | 평가 4.4, Figure 15 (f4c9ca0 이후) |
| T4 | 에이전트 세션 (132) | Thor | 세션 실험에서 Copied File과 STATOR가 다른가? | 거의 같음(TTFT p50 861 vs 868 ms, 메모리 43.4 vs 46.3 GiB) | 미반영 |
| T5 | 결함 sharer (135) | Thor | 잘못된 GPU 쓰기가 다른 에이전트에 무엇을 하는가? | Device: 자식 24/24와 부모가 1-7토큰 안에 텍스트가 바뀜, 오류 없음. STATOR: illegal address, 텍스트와 속도 그대로 | Figure 16(b), 평가 4.4 |
| T6 | Delta-Copy 반복 fork (136, 리뷰 4) | Thor | 새 행만 복사하는 최적 복사가 STATOR만큼 빠른가? | 이후 fork는 delta 5.8-9.0 ms로 STATOR 8.5-9.5 ms와 같거나 짧음. 첫 fork 82-84 vs 11.5-12.2 ms, 8회 합계 127-129 vs 73-76 ms, 정점 메모리 7.7-8.0 vs 5.8-6.9 GiB | Figure 15, 평가 4.4 "Repeated Forks." |
| A1 | Linux 메커니즘 프로브 | A30 | Thor에서 본 문제가 x86 + 분리 메모리(HMM)에서도 생기는가? | GPU 읽기에 1 GiB 복사, fork/KSM 공유 깨짐, reclaim 폴트 4.3 M | 배경 2.2-2.3, Figure 3(a)(b) |
| A2 | 계층별 비용 프로브 | A30 | afblock, 항목당 무효화는 어느 계층의 비용인가? | afblock 복사 0, 항목당 +0.1-0.9 µs (Thor 9.6-10.9 µs) | 배경 2.2 |
| A3 | 보호(G3) 프로브 | A30 | 읽기 전용 매핑으로의 GPU 쓰기가 거부되는가? | 6/6 거부, VMM importer는 6/6 쓰기 성공 | 근거 자료 |
| A4 | extent 읽기(G1) 프로브 | A30 | 읽기 전용 extent를 GPU가 폴트 없이 읽는가? | 아니오. 프로세스마다 첫 읽기 약 1 s/256 MiB, 어떤 힌트로도 줄지 않음 | Discussion |
| A5 | 32프로세스 읽기 정확성 | A30 | Thor의 오독(5,120회 중 16회)이 A30에서도 생기는가? | 256회 중 0, 2 MiB 매핑 0 | 배경 2.3 |
| A6 | 엔진 E1-E5 | A30 | dGPU에서 STATOR의 이점은 무엇이 남는가? | 보호·반복 fork·메모리·VRAM 포화 시 용량은 유리, TTFT·속도·첫 publish는 Device보다 불리 | 미반영 (rebuttal 자료) |

## 2. Thor 실험

### T1. Fault-around (큐 126)
- **질문:** 4 KiB 페이지 캐시에 둔 모델 파일을 CPU로 미리 읽어도(populate) GPU 폴트가 남는 이유는 무엇인가?
- **구성:** populate 요청 뒤 GPU 읽기. `fault_around_bytes`를 기본값과 4096으로 비교했고, 6회 실행했다.
- **결과:** 기본값에서는 475,648개 폴트가 남았다. fault-around를 한 페이지로 줄이면 6/6에서 0개였다. 커널이 폴트 난 페이지의 이웃을 accessed 플래그 없이 매핑하기 때문이다.
- **반영:** 평가 4.3 ("These faults come from fault-around ...").
- **데이터:** `llm_share/results/20261021-gpu-faultaround-probe-full-v1`, `run_gpu_faultaround_probe.sh`, `summarize_gpu_faultaround_probe.py`.

### T2. 인계 전체 비교 (큐 128-130)
- **질문:**
  1. Linux가 제공하는 읽기 전용 파일 매핑만으로 STATOR와 같은 효과가 나는가? 여기서 비교하는 STATOR (Copied File)는 publish를 제자리 변환 대신 파일 복사로 바꾼 ablation이다.
  2. 파일을 immutable로 만드는 선택 단계의 비용은 얼마인가?
- **구성:** tuned4 엔진, prefix 4,081/16,321 토큰, 자식 8개와 32개, 6회 반복. 방식은 Copy, Demand, Device, CoW Map, Copied File, STATOR, STATOR(immutable 파일). `GGML_CUDA_FATTN_HOST_GQA=2`를 썼다. 128에서 이 설정이 빠져 Copy 텍스트가 달랐고, 129에서 다시 돌렸다.
- **결과 (16,321 토큰, 자식 8개):**

| 방식 | 부모 멈춤 | 자식 attach | 자식 메모리 | 전체 메모리 | 자식 TTFT |
|---|---:|---:|---:|---:|---:|
| Copy | 574.5 ms | 560 ms | 20.44 GiB | 27.87 GiB | 2,462 ms |
| STATOR (Copied File) | 82.7 ms | 64 ms | 6.05 GiB | 12.91 GiB | 1,959 ms |
| STATOR | 11.8 ms | 63 ms | 6.17 GiB | 12.12 GiB | 1,955 ms |
| STATOR (immutable 파일) | 19.1 ms | - | - | - | - |

  - 126케이스, 1,836개 프로세스의 텍스트가 모두 같았다.
  - immutable 파일에 대한 쓰기 시도는 18/18 거부됐다.
  - CoW Map은 자식 32개에서 GPU 폴트 32.2 M이었다.
- **해석:** 자식 쪽은 두 방식이 같고, 차이는 부모의 멈춤과 복사 한 번이다. 읽기 전용 매핑이 아니라 STATOR의 변환이 복사를 없앤다.
- **반영:** Figure 14(Copied File 막대), 평가 4.4 "Copied File" 문단, 설계 3.3(immutable 파일 단계), Table 2(Immutable memory file 행).
- **데이터:** `llm_share/results/20261021-review-handover-all-v1`, `count_texts.py`(5,354/5,354).

### T3. 실행 중 반복 fork (큐 131, 133)
- **질문:** 실행 중인 에이전트를 8번 fork할 때 fork마다의 멈춤과 정점 메모리는 어떤가?
- **구성:** 16,321 토큰, fork 8회, 자식 1개씩, 6회 반복. 131에서는 Copied File이 이전 publish의 파일을 지우지 않아 메모리가 부풀었다. 133에서 `--unlink`로 다시 돌렸다.
- **결과 (133):**

| 방식 | fork당 멈춤 | 정점 메모리 |
|---|---:|---:|
| Copy | 530-625 ms | 9.7-12.8 GiB |
| STATOR (Copied File) | 81-92 ms | 7.9-8.1 GiB |
| STATOR | 9-12 ms (첫 fork 약 12 ms) | 6.9-7.0 GiB |

  - 131 기준 8회 멈춤 합계는 Copy 4,688 ms, Copied File 689 ms, STATOR 73 ms이다.
  - 부모의 전체 시간은 85.1 / 79.7 / 79.1 s이다.
  - 자식 TTFT는 1,156 / 835 / 836 ms이다.
- **반영:** 평가 4.4 "Fork of a Running Process" 문단에 두 문장으로 넣었다(커밋 f4c9ca0). 133의 정확한 범위는 fork당 5.7-12.2 / 81-92 / 527-626 ms, 정점 6.9-7.0 / 7.9-8.1 / 9.7-12.9 GiB, GPU 폴트 0이다.
- **데이터:** `llm_share/results/20261021-review-repeatfork-v1`, `20261022-review-repeatfork-unlink-v1`.

### T4. 에이전트 세션 (큐 132)
- **질문:** 시간에 따라 도착하는 에이전트 세션에서 Copied File과 STATOR가 다른가?
- **구성:** trace seed 1, 평균 도구 대기 480 s와 960 s. 방식은 Copied File, STATOR, Copy(가중치만 공유)이다.
- **결과:**

| 평균 대기 | 방식 | TTFT p50 | 큐 대기 | 정점 메모리 |
|---|---|---:|---|---:|
| 480 s | Copied File | 861 ms | 없음 | 43.4 GiB |
| 480 s | STATOR | 868 ms | 없음 | 46.3 GiB |
| 480 s | Copy | 1,020 ms | 없음 | 92.6 GiB |
| 960 s | Copied File | 869 ms | 없음 | 50.7 GiB |
| 960 s | STATOR | 878 ms | 없음 | 50.6 GiB |
| 960 s | Copy | 1,048 ms | 54개 중 4~5개가 12-178 s | 95.8 GiB |

  - 세션에서는 자기 행이 적어 복사 비용이 작고, 두 방식이 거의 같다.
  - STATOR와 Copy를 다시 잰 값은 기존 값과 1% 안에서 같다.
- **반영:** 없음. 이 과정에서 4.7 문장의 꼬리 지연 표기를 고쳤다(세 trace의 p95 평균 대신 큐에서 기다린 세션 수, 커밋 38ebc54).
- **데이터:** `portable_bench_a30/validation/thor_e4b_review/`.

### T5. 결함 sharer (큐 134 중단, 135)
- **질문:** 공유 prefix에 GPU로 잘못 쓰는 sharer 하나가 다른 에이전트에 무엇을 하는가?
- **구성:**
  - 16,321 토큰, 자식 8개, 96토큰 생성, 3회 반복. 방식마다 결함 sharer가 없는 대조 실행을 함께 돌렸다.
  - 결함 sharer(`fault_writer.cu`)는 5초 뒤 접근할 수 있는 모든 할당의 앞 2 MiB에 0을 쓴다. Device는 매핑 권한을 쓰기로 올린 뒤 쓰고, STATOR는 에이전트처럼 읽기 전용으로 매핑한 상태에서 쓴다.
  - 134는 2 MiB보다 큰 할당을 부분 매핑할 수 없어 중단했고, 135에서 할당 전체를 매핑하도록 고쳤다.
- **결과:**
  - Device: 112개 할당에 모두 쓰기가 성공했다. 자식 24/24와 부모 3/3의 텍스트가 쓰기 1-7토큰 안에 대조 실행과 달라졌고, 어떤 프로세스도 오류를 내지 않았다.
  - STATOR: 첫 쓰기에서 illegal address가 났다. 자식과 부모의 텍스트가 모두 같았고, 쓰기 뒤 생성 속도는 대조 실행의 1.002배였다.
- **반영:** Figure 16(b) "Children after a Faulty Write.", 평가 4.4 Shared Device Memory 문단 (커밋 5c8b2cb).
- **데이터:** `llm_share/results/20261022-review-fault-writer-v2`, 도구 `stator_campaign_20261016/{src/fault_writer.cu, src/kv_spawn_fault.cpp, tools/run_fault_writer.py, tools/summarize_fault_writer.py}`, 그림 `llm_share/figures/make_fault_figure.py`.

### T6. Delta-Copy 반복 fork (큐 136, 리뷰 (4) 대응)
- **질문:** 반복 fork에서 지난 fork 이후 새로 쓴 행만 복사하는 최적 복사(STATOR (Copied Delta))가 제자리 전환만큼 빠른가?
- **구성:** T3(133)와 같은 조건이다(16,321 토큰, fork 8회, 256토큰 간격, 6회). 엔진은 tuned5(tuned4에 `LLAMA_KV_PUBLISH_COPY=2` 추가)이다. 파일은 첫 publish 때 만들어 부모에 쓰기 가능하게 매핑해 두고, 이후 publish는 새 행만 복사한다. 네 방식 모두 tuned5로 돌렸다.
- **결과:** 24케이스 모두 정상, GPU 폴트 0, 자식 8명의 텍스트가 네 방식에서 모두 같았다.

| 방식 | 첫 fork | 이후 fork당 | 8회 합계 | 정점 메모리 | 자식 TTFT 중앙값 |
|---|---:|---:|---:|---:|---:|
| Copy | 568-582 ms | 518-636 ms (한 번 1,035) | 4,552-5,197 ms | 10.4-13.1 GiB | 1,194 ms |
| STATOR (Copied File) | 80-84 ms | 84-91 ms | 686-701 ms | 7.95-8.19 GiB | 855 ms |
| STATOR (Copied Delta) | 82-84 ms | 5.8-9.0 ms | 127-129 ms | 7.72-8.01 GiB | 858 ms |
| STATOR | 11.5-12.2 ms | 8.5-9.5 ms | 73-76 ms | 5.78-6.88 GiB | 854 ms |

- **해석:** 새 행만 복사하면 이후 fork는 STATOR와 같거나 더 짧다. 첫 fork의 전체 복사와 공개된 행의 두 번째 사본(약 1 GiB)은 제자리 전환만 없앤다. 논문은 이 방향으로 썼다(리뷰어 제안과 같음).
- **반영:** Figure 15(네 방식), 평가 4.4 "Repeated Forks." 문단.
- **데이터:** `llm_share/results/20261022-review-repeatfork-delta-v1`, 엔진 `llm_share/llama.cpp-tuned5`, 실행기 `stator_campaign_20261016/tools/run_repeat_fork_v2.py`.

## 3. A30 실험 (CloudLab d7525, x86 + NVIDIA A30, HMM)

**환경:** AMD EPYC 64코어, 125 GiB, A30 24 GiB(PCIe), Ubuntu 24.04, Linux 6.8.0-138, open 드라이버 615.71.10, CUDA 13.4, Addressing Mode HMM. `check_env.sh` 항목이 모두 ok였다.

### A1. Linux 메커니즘 프로브 (mech, 6회)

| 1 GiB | CPU 읽기 | GPU 읽기 (A30) | GPU 읽기 (Thor) |
|---|---|---|---|
| private mapping | 0.18 s, 복사 0 | 4.65 s, 1,024 MiB 복사, 폴트 2.6 M | 0.69 s, 1.0 GiB 복사 |
| shared mapping | 0.17 s | 4.62 s, 폴트 2.6 M | 1.6 s, 폴트 1.1 M |
| fork 뒤 읽기 | 0.10 s | 3.54 s, 1,024 MiB 복사 | 0.99 GiB 복사 |
| KSM | 0.07 s, 병합 유지 | 3.53 s, 병합 262,144 → 0 | 읽기에서는 유지, 쓰기에서 해제 |
| reclaim 압박 속도 | 0.83 | 0.63 (폴트 4.3 M), 고정 시 0.96 | CPU 1.00, GPU 0.56, 고정 0.95 |

게이트 M1-M3를 모두 통과했다. 반영 위치는 배경 2.3과 Figure 3(a)(b)이다(커밋 6ff3e2a, 2b0d870, 2fb6a1c).

### A2. 계층별 비용 프로브
- **afblock:** accessed 비트를 지운 페이지가 1개든 전부든 복사가 0이었다(Thor는 1페이지에 2 MiB 복사). Arm SVA 고유 비용이다.
- **mmtax:** CUDA가 있을 때 항목당 비용이 1.0-1.3배 늘었다(+0.1-0.9 µs). Thor는 9.6-10.9 µs이다. SMMU 고유 비용이다.
- **fork 뒤 첫 GPU 쓰기 (1 GiB):** fork 뒤 1,008 ms, spawn 뒤 465 ms였다(Thor 2.8 s vs 5.3 ms). 이 비용은 A30에도 있다.
- **seal 의미:** Thor와 같았다(매핑이 있는 동안 write seal은 EBUSY, future-write는 기존 매핑의 쓰기를 막지 못함).
- **반영:** 배경 2.2.

### A3. 보호 프로브
- **G3 쓰기 확인:** 읽기 전용 매핑에 대한 GPU 쓰기 6/6이 illegal address였고, 바뀐 블록은 0이었다. 공유 쓰기 매핑은 6/6에서 128블록이 바뀌었다.
- **cuda_protect_probe:** host 경로 6/6 거부에 파일은 그대로였다. VMM 경로는 importer가 권한을 올려 6/6 쓰기에 성공했고 원본이 바뀌었다.
- **반영:** 근거 자료로만 쓴다(Discussion에서 "each importer determines the access of its own mapping"의 근거).

### A4. extent 읽기 프로브 (G1)
- 읽기 전용 공유 tmpfs 매핑을 GPU가 처음 읽을 때 256 MiB에 약 1 s가 걸렸다(폴트 약 0.6 M). 다시 읽을 때는 폴트 0에 23.7 GB/s였고, 장치 메모리는 710 GB/s이다.
- 아래 다섯 가지 모두 효과가 없었다.
  - CPU 미리 읽기
  - prefetch
  - `cudaMemAdviseSetAccessedBy`
  - 2 MiB 페이지 tmpfs(ShmemPmdMapped 256 MiB)
  - 읽기 전용 `cudaHostRegister`(이 GPU는 `HostRegisterReadOnlySupported=0`이라 지원하지 않음)
- **이유:** HMM에서는 GPU가 페이지 테이블 사본을 따로 두고(`UsesHostPageTables=0`), 프로세스마다 첫 접근 때 폴트로 채운다.
- 복사는 없었다(익명 메모리 0).
- **반영:** Discussion "Generality and Unified Memory" (G1은 호스트 페이지 테이블을 쓰는 GPU에서 성립).

### A5. 32프로세스 동시 읽기 정확성
- 32개 프로세스가 2 MiB 페이지 tmpfs의 896 MiB를 동시에 읽었다. 8라운드 256회에서 잘못 읽힌 값이 0이었다. GPU 폴트 뒤 파일이 2 MiB 단위로 매핑되지 않았다(pmd_kib=0). Thor의 오독은 2 MiB 매핑에서 생겼다.
- **반영:** 배경 2.3 ("or on the A30 with 32 processes (256), whose GPU faults leave no 2 MiB entry").

### A6. 엔진 실험 E1-E5
- **구성:**
  - Thor와 같은 엔진 소스를 sm_80으로 빌드했다. 드라이버 사본에만 `STATOR_NO_MMAP=1`을 넣어, 가중치는 프로세스마다 VRAM에 올라간다.
  - Qwen2.5-1.5B-Instruct Q4_K_M, 16,321 토큰.
  - VRAM은 200 ms마다 기록했다.
- **결과 (자식 8개):**

| 실험 | Copy | Device | CoW Map | Copied File | STATOR |
|---|---|---|---|---|---|
| E1 첫 publish 멈춤 | 990 ms | 34 ms | 234 ms | 140 ms | 233 ms |
| E1 자식 TTFT | 30.4 s | 24.0 s | 54.7 s | 35.2 s | 44.9 s |
| E1 생성 속도 | 111 tok/s | 120 | 30 | 37 | 24 |
| E1 VRAM 정점 | 21.4-21.6 GiB | 14.3-16.1 | 13.8 | 13.8 | 13.8 |
| E2 fork당 멈춤 | 965-1,102 ms | - | - | 81-113 ms | 첫 224-238, 이후 9.4-12.7 ms |
| E3 결함 쓰기 | - | 자식 24/24 변경 | - | - | 거부, 24/24 그대로 |
| E4 결함 쓰기 | - | - | - | - | Shared Map 0/32 그대로, STATOR 32/32 그대로 |
| E4 자식 호스트 메모리 | 3.7 GiB | - | 7.9 | - | 4.3 |

- **E5 (자식 16개):** VRAM이 모자라 Copy는 16개 중 13-14개가 실패했고, 호스트 매핑 방식은 2-5개가 실패했다.
- **해석:**
  - 유리한 점: 보호, 반복 fork 멈춤, CoW Map 대비 메모리, Copy 대비 VRAM, VRAM 포화 시 용량.
  - 불리한 점: TTFT, 속도, 첫 publish가 Device보다 나쁘다. PCIe(장치 메모리의 0.03)와 HMM 첫 접근 폴트 때문이다.
- **반영:** 없음(결정: A30은 배경과 Discussion에만 쓰고 평가에는 쓰지 않음). rebuttal 자료로 보관한다.
- **데이터:** `portable_bench_a30/results/a30_20261022/engine/` (`A30_ENGINE_NOTES.md` 포함).

## 4. 논문에 아직 넣지 않은 결과

| 결과 | 넣는다면 |
|---|---|
| T4 세션의 Copied File | 넣지 않는 것을 권함(차이 없음) |
| A6 엔진 결과 | 넣지 않음(결정됨), rebuttal용 |

## 5. 데이터 위치 요약
- **Thor:** `/home/thor/kcj/llm_share/results/2026102{1,2}-review-*`, `/home/thor/kcj/portable_bench_a30/validation/thor_e4b_review/`, 큐 로그 `/home/thor/kcj/stator_campaign_logs/logs/12{6..9}-*, 13{0..5}-*`
- **A30:** `/home/thor/kcj/portable_bench_a30/results/a30_20261022/` (프로브: `node0-20261022-020702`, `write-check-*`, `extent-read-*`, `protect-030048`, `read-trace-a30`, 장치 속성 `attributes.txt`. 엔진: `engine/`)
