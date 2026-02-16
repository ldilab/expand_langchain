# Local Tracing Module

LangGraph/LangChain 기반 SQL 생성 파이프라인을 위한 로컬 트레이싱 모듈입니다.

## 기능

1. **로컬 Trace 저장**: YAML 형식으로  LLM 호출, 도구 실행, 상태 변화 저장
2. **실시간 모니터링**: Trace 파일은 실시간으로 업데이트
3. **Trace 뷰어**: 생성 완료 후 trace를 보기 좋게 출력하는 CLI 도구
4. **계층적 Debug Trace (NEW)**: AI agent가 디버깅하기 쉬운 구조화된 trace 자동 생성

## NEW: 계층적 Debug Trace

    평탄한 이벤트 리스트 대신, **계층적으로 구조화된 trace**를 자동 생성합니다.

### 주요 개선사항

1. **계층적 실행 트리**
   - 부모-자식 관계가 명확한 트리 구조
   - 각 노드의 실행 순서와 중첩 관계 명시적 표현

2. **LLM 호출 통합**
   - start/end 이벤트를 하나의 `llm_call`로 통합
   - 입력 메시지와 출력이 함께 저장
   - 모델명, 실행 시간, 노드 위치 포함

3. **에러 컨텍스트**
   - 에러 발생 위치와 전체 스택 명시
   - 에러가 발생한 노드의 입출력 보존

4. **실행 메타데이터**
   - 전체 LLM 호출 수, 에러 수 요약
   - 각 노드의 실행 시간 (duration_ms)
   - 노드 타입 (llm_call, graph_node, tool, chain)

### 사용법

```python
from expand_langchain.generator import Generator

generator = Generator(
    config_name="ours",
    dataset_name="spider2",
    tracing_on=True,           # 트레이싱 활성화
    tracing_ai_summary=True,   # 계층적 trace 생성 (NEW)
)

generator.run(n=10)
```

### 저장 위치

```
results/<run-name>/
 results/          # 기존 결과 파일
 checkpoints/      # LangGraph 체크포인트
 traces/           # 트레이싱 데이터
    ├── trace.yaml               # 원본 이벤트 trace (평탄한 리스트)
    ├── <task_id>_trace.yaml     # Task별 원본 trace
    ├── <task_id>_debug.yaml     # Task별 계층적 debug trace (NEW)
    └── <task_id>_full_run_histories.yaml  # 실행 히스토리
```

### 출력 예

`0_debug.yaml`:
```yaml
task_id: '0'
start_time: '2026-01-27T01:50:01.805643'
end_time: '2026-01-27T01:53:29.957009'
total_events: 140
llm_calls: 10
errors: 11
execution:
- type: chain
  name: chain
  start: '2026-01-27T01:50:01.805643'
  status: success
  steps:
  - type: graph_node
    name: exploration_agent
    graph_node: exploration_agent
    start: '2026-01-27T01:50:01.817476'
    end: '2026-01-27T01:50:25.842088'
    duration_ms: 24024.612
    status: success
    steps:
    - type: llm_call
      name: ChatOpenAI
      graph_node: exploration_agent
      start: '2026-01-27T01:50:02.941335'
      end: '2026-01-27T01:50:25.784686'
      duration_ms: 22843.38
      status: success
      llm:
        model: claude-3-5-sonnet
        input_messages:
        - type: system
          content: |
            # Output Format
            First, reason through the problem step-by-step...
        - type: human
          content: |
            Table: WEATHER.HISTORY_DAY
            Columns: POSTAL_CODE, DATE, SNOWFALL...
        output: |
          Let me explore this database...
          ```json
          {"sql_queries": [...]}
          ```
```

### AI Agent에게 유용한 이유

1. **즉시 이해 가능**: 트리 구조로 전체 실행 흐름을 한눈에 파악
2. **LLM 디버깅 쉬움**: 각 LLM 호출의 입력/출력이 한 곳에 모여 있어 생성 결과 검토 용이
3. **에러 추적 간편**: 에러 발생 위치와 컨텍스트가 명확
4. **파싱 용이**: 구조화된 YAML이라 프로그램으로 파싱/분석 쉬움
5. **성능 분석**: duration_ms로 어느 노드가 느린지 즉시 파악

## 기존 기능

### 1. Generator와 함께 사용

```python
from expand_langchain.generator import Generator

generator = Generator(
    config_name="ours",
    dataset_name="spider2",
    tracing_on=True,
    tracing_ai_summary=True,  # 계층적 debug trace 생성
)

generator.run(n=5)
```

### 2. 직접 Callback 사용

```python
from expand_langchain.tracing import LocalTraceCallback, TracingConfig

config = TracingConfig(
    run_name="my-run",
    results_dir="results",
    log_llm_io=True,
    log_tool_io=True,
    log_graph_state=True,
    max_content_length=10000,
    generate_ai_summary=True,         # 계층적 debug trace 생성
)

callback = LocalTraceCallback(config)

# LangGraph/LangChain에서 사용
result = await graph.ainvoke(
    input_data,
    config={"callbacks": [callback]}
)

# 완료  정리
callback.finalize_all()
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
2. 대용량 입출력은 `max_content_length`에 따라 잘립니다.
3. `_debug.yaml` 파일은 task 완료 후에 생성됩니다.
