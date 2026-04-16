# Architecture: Functional Layer Hierarchy

## Domain

Bidirectional API protocol translation proxy: Anthropic Claude API ↔ OpenAI Chat Completion API.

Primary operation: translate a request in format A to format B, forward it upstream, translate the response back from B to A. Two temporal modes: batch (non-streaming) and incremental (streaming SSE).

## Layer Hierarchy

```
LAYER 0: Protocol Types
    │
LAYER 1: Translation Core (pure atomic mappings)
    │
    ├──────────────────┐
    │                  │
LAYER 2a: Pipeline   LAYER 2b: Stream Translator
  (batch translation    (incremental response
   with policy)          state machine)
    │                  │
    └──────────────────┘
           │
LAYER 3: I/O Shell
  (HTTP, SSE framing, config loading, wiring)
```

## LAYER 0: Protocol Types

**Module:** `src/models/anthropic.rs`, `src/models/openai.rs`

**Concepts:** Wire format data shapes for both APIs. Pure algebraic types, no logic.

**Types:**
- `anthropic::{Request, Message, ContentBlock, SystemPrompt, Response, StreamEvent, ...}`
- `openai::{Request, Message, ContentPart, Response, StreamChunk, ...}`

**Inexpressible:**
- Cross-protocol references
- Business logic
- I/O

**Depends on:** Nothing.

## LAYER 1: Translation Core

**Module:** `src/translate/core.rs`

**Concepts:** Atomic pure mappings between protocol concepts. Each function translates exactly one concept. No configuration, no routing, no state.

**Functions:**
```
translate_message(anthropic::Message) -> Result<Vec<openai::Message>>
translate_tool(anthropic::Tool) -> openai::Tool
normalize_schema(Value) -> Value
remove_term(text, term) -> String
map_stop_reason(Option<&str>) -> Option<String>
is_batch_tool(&anthropic::Tool) -> bool
```

**Inexpressible:**
- Configuration-dependent behavior
- Temporal state
- I/O

**Depends on:** Layer 0

## LAYER 2a: Translation Pipeline

**Module:** `src/translate/pipeline.rs`

**Concepts:** Config-aware composition of Layer 1 atoms into full request/response translations. Policy decisions: model routing, prompt sanitization, tool filtering.

**Types:**
```
TranslationPolicy { reasoning_model, completion_model, model_map, ignore_terms }
```

**Functions:**
```
translate_request(anthropic::Request, &TranslationPolicy) -> Result<openai::Request>
translate_response(openai::Response, fallback_model) -> Result<anthropic::Response>
translate_models_list(openai::ModelsListResponse) -> anthropic::ModelsListResponse
```

**Inexpressible:**
- Streaming/temporal concerns
- I/O
- Raw byte manipulation

**Depends on:** Layer 1, Layer 0

## LAYER 2b: Stream Translator

**Module:** `src/translate/stream.rs`

**Concepts:** Pure state machine translating OpenAI streaming chunks into Anthropic SSE events. Valid state transitions enforced by types.

**Types:**
```
BlockState { Idle, Thinking { index }, Text { index }, ToolUse { index, id } }
StreamState { message_id, model, fallback_model, block, next_index, message_started }
```

**Functions:**
```
initial_state(fallback_model: String) -> StreamState
translate_chunk(&mut StreamState, &openai::StreamChunk) -> Vec<anthropic::StreamEvent>
translate_done(&mut StreamState) -> Vec<anthropic::StreamEvent>
translate_error(message: String) -> Vec<anthropic::StreamEvent>
```

**Key invariant:** Every `ContentBlockStart` is followed by exactly one `ContentBlockStop` before the next `ContentBlockStart`. The `BlockState` enum enforces this.

**Inexpressible:**
- I/O, async, bytes
- SSE framing
- Configuration (model routing happens in Layer 2a before streaming starts)

**Depends on:** Layer 1 (reuses `map_stop_reason`), Layer 0

## LAYER 3: I/O Shell

**Module:** `src/proxy.rs`, `src/config.rs`, `src/cli.rs`, `src/main.rs`

**Concepts:** Side effects. HTTP server/client, SSE byte framing, configuration loading, logging, daemon management.

**Functions:**
```
proxy_handler(config, client, request) -> Response
list_models_handler(config, client) -> Response
parse_sse_frames(&mut buffer, &[u8]) -> Vec<String>
serialize_event(&StreamEvent) -> Bytes
```

**Inexpressible:** Business logic. Handlers call pure functions and pipe data.

**Depends on:** Layer 2a, Layer 2b, Layer 0

## Compilation Chain

Streaming request trace:

```
proxy_handler(req)                                        [Layer 3: I/O]
  → policy = TranslationPolicy::from(&config)             [Layer 2a]
  → openai_req = translate_request(req, &policy)          [Layer 2a]
      → model = select_model(req, &policy)                [Layer 2a: routing]
      → messages = translate_message(msg) for each        [Layer 1: atom]
      → tools = translate_tool(t) for each                [Layer 1: atom]
      → system = sanitize(system, &policy.ignore_terms)   [Layer 1: atom + Layer 2a]
  → upstream_stream = client.post(url).send()             [Layer 3: I/O]
  → state = initial_state(model)                          [Layer 2b]
  → for each raw_bytes:                                   [Layer 3: I/O]
      → frames = parse_sse_frames(&mut buffer, &bytes)    [Layer 3: wire protocol]
      → for each frame:
          chunk = deserialize(frame)                      [Layer 3: serde]
          events = translate_chunk(&mut state, &chunk)    [Layer 2b: pure]
          bytes = events.map(serialize_event)             [Layer 3: wire protocol]
          yield bytes                                     [Layer 3: I/O]
```

## Module Layout

```
src/
  models/
    mod.rs
    anthropic.rs    ← Layer 0
    openai.rs       ← Layer 0
  translate/
    mod.rs          ← re-exports
    core.rs         ← Layer 1
    pipeline.rs     ← Layer 2a
    stream.rs       ← Layer 2b
  proxy.rs          ← Layer 3: HTTP handlers
  config.rs         ← Layer 3: config loading
  cli.rs            ← Layer 3: CLI parsing
  main.rs           ← Layer 3: wiring
  error.rs          ← cross-cutting
```

## Invariants

1. Layers 1, 2a, 2b contain NO I/O, NO async, NO logging
2. Layer 3 contains NO business logic — only wiring
3. translate/ modules never import from proxy.rs, config.rs, cli.rs, or main.rs
4. proxy.rs never constructs Anthropic StreamEvents directly — only via translate/stream.rs
5. All state in translate/ is passed explicitly — no globals, no Arc, no hidden dependencies
