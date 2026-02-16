# Local Tracing Module

LangGraph/LangChain 기반 SQL 생성 파이프라인을 위한 로컬 트레이싱 모듈입니다.

## 기능

1. **로컬 Trace 저장**: JSON/JSONL/YAML 형식으로 각 LLM 호출, 도구 실행, 상태 변화 저장
2. **실시간 모니터링**: `tail -f`로 볼 수 있는 로그 파일
3. **Trace 뷰어**: 생성 완료 후 trace를 보기 좋게 출력하는 CLI 도구
4. **AI 친화적 요약 (NEW)**: 외부 AI agent가 디버깅하기 쉽도록 Markdown 리포트 자동 생성

## 사용법

### 1. Generator와 함께 사용

```python
from expand_langchain.generator import Generator

# tracing_on=True로 로컬 트레이싱 활성화
generator = Generator(
    config_name="ours",
    dataset_name="spider2",
    tracing_on=True,           # 트레이싱 활성화
    tracing_realtime=True,     # 실시간 로그 활성화
    tracing_max_content_length=10000,  # 최대 컨텐츠 길이
    tracing_ai_summary=True,   # AI 친화적 요약 생성 (NEW)
)

generator.run(n=5)
```

실행하면 다음과 같은 메시지가 출력됩니다:
```
TRACING: Realtime log at results/<run-name>/traces/realtime.log
TRACING: Use 'tail -f results/<run-name>/traces/realtime.log' to monitor
```

### 2. 실시간 모니터링

다른 터미널에서:
```bash
tail -f results/<run-name>/traces/realtime.log
```

### 3. Trace 뷰어 사용

```bash
# 전체 요약 보기
uv run python -m expand_langchain.tracing results/<run-name>/traces/ --summary

# 특정 task의 이벤트 보기
uv run python -m expand_langchain.tracing results/<run-name>/traces/ --task-id task_0

# 상세 출력 (입출력 포함)
uv run python -m expand_langchain.tracing results/<run-name>/traces/ --task-id task_0 --verbose

# 실시간 로그 watch
uv run python -m expand_langchain.tracing results/<run-name>/traces/ --watch

# JSON 형식 출력
uv run python -m expand_langchain.tracing results/<run-name>/traces/ --json
```

### 4. 직접 Callback 사용

```python
from expand_langchain.tracing import LocalTraceCallback, TracingConfig

# 설정 생성
config = TracingConfig(
    run_name="my-run",
    results_dir="results",
    enable_realtime_log=True,
    log_llm_io=True,
    log_tool_io=True,
    log_graph_state=True,
    max_content_length=10000,
    generate_ai_summary=True,         # AI 친화적 요약 생성
    summary_max_content_length=5000,  # 요약 파일 최대 컨텐츠 길이
)

# 콜백 생성
callback = LocalTraceCallback(config)

# LangGraph/LangChain에서 사용
result = await graph.ainvoke(
    input_data,
    config={"callbacks": [callback]}
)

# 완료 후 정리
callback.finalize_all()
```

## 저장 위치

```
results/<run-name>/
├── results/          # 기존 결과 파일
├── checkpoints/      # LangGraph 체크포인트
└── traces/           # 트레이싱 데이터
    ├── realtime.log  # 실시간 로그 (tail -f용)
    ├── trace.yaml    # YAML 형식 trace (기본)
    ├── task_0.jsonl  # task별 이벤트 (append-only)
    ├── task_0.json   # task별 전체 세션 (완료 후)
    │
    # AI-friendly summary files (NEW)
    ├── task_0_SUMMARY.md         # 전체 실행 요약
    ├── task_0_LLM_CALLS.md       # LLM 호출 상세
    ├── task_0_ERROR_ANALYSIS.md  # 에러 분석 (있는 경우)
    ├── task_0_DEBUG_GUIDE.md     # 디버깅 가이드
    └── task_0_summary.json       # 구조화된 요약 (AI 파싱용)
```

## AI 친화적 요약 기능 (NEW)

task 완료 시 자동으로 5개의 요약 파일이 생성됩니다:

### 1. SUMMARY.md - 전체 실행 흐름 요약
- Task 상태 (Success/Failed)
- 실행 시간, LLM 호출 횟수, 에러 개수
- 계층적 실행 트리 (어떤 노드가 어떤 순서로 실행되었는지)
- 주요 LLM 호출 요약
- 에러 및 복구 정보

### 2. LLM_CALLS.md - LLM 호출 상세 기록
- 각 LLM 호출의 완전한 입력 메시지 (system, user, assistant)
- LLM 응답 전문
- 실행 시간, 모델명, 노드 정보

### 3. ERROR_ANALYSIS.md - 에러 분석
- 각 에러의 발생 시점과 위치
- 에러 메시지 전문
- 에러 발생 직전의 상태 (context)
- 에러 복구 여부 및 복구 방법

### 4. DEBUG_GUIDE.md - 디버깅 가이드
- 빠른 상태 체크
- 디버깅 시작 방법 안내
- 이번 실행에서 발견된 이슈 목록
- 디버깅 팁

### 5. summary.json - 구조화된 요약
- 전체 실행 정보를 JSON 형식으로
- AI agent가 파싱하기 쉬운 구조
- 실행 트리, LLM 호출, 에러 정보 포함

### 사용 예시

```python
# AI agent가 디버깅을 시작할 때:
# 1. SUMMARY.md 읽어서 전체 흐름 파악
# 2. ERROR_ANALYSIS.md로 에러 원인 분석
# 3. LLM_CALLS.md로 잘못된 생성 결과 확인
# 4. 필요시 summary.json 파싱해서 자동 분석
```

## 이벤트 타입

- **LLM 이벤트**: `llm_start`, `llm_end`, `llm_error`, `chat_model_start`, `chat_model_end`
- **Chain 이벤트**: `chain_start`, `chain_end`, `chain_error`
- **Tool 이벤트**: `tool_start`, `tool_end`, `tool_error`
- **Agent 이벤트**: `agent_action`, `agent_finish`
- **Graph 이벤트**: `graph_node_start`, `graph_node_end`, `graph_state_update`
- **Retriever 이벤트**: `retriever_start`, `retriever_end`

## 주의사항

1. `tracing_on=True`로 설정해야 트레이싱이 활성화됩니다.
2. 대용량 입출력은 `tracing_max_content_length`에 따라 잘립니다.
3. JSONL 파일은 실행 중에도 읽을 수 있지만, JSON 파일은 task 완료 후에 생성됩니다.
