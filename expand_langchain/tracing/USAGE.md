# Local Tracing Module

LangGraph/LangChain 기반 SQL 생성 파이프라인을 위한 로컬 트레이싱 모듈입니다.

## 기능

1. **로컬 Trace 저장**: JSON/JSONL 형식으로 각 LLM 호출, 도구 실행, 상태 변화 저장
2. **실시간 모니터링**: `tail -f`로 볼 수 있는 로그 파일
3. **Trace 뷰어**: 생성 완료 후 trace를 보기 좋게 출력하는 CLI 도구

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
    ├── task_0.jsonl  # task별 이벤트 (append-only)
    └── task_0.json   # task별 전체 세션 (완료 후)
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
