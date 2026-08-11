# debgo-ai-carbonara-langgraph

An agentic AI orchestration demo using a kitchen scenario. Simple enough to explain in thirty seconds. Interesting enough to watch.

## What it shows

A dispatcher receives a plain English order. It knows nothing about cooking. An expeditor -- powered by a local LLM -- reasons through the job, picks the right doers, and sequences them correctly. The doers execute their one job each and report back.

The LLM figures out the sequence. That is the agentic part.

## Roles

- **Dispatcher** -- takes the order, delivers the result, knows nothing about kitchens
- **Expeditor** -- agentic brain; reads the doer registry, thinks out loud, builds a plan
- **Doers** -- single-purpose workers: `boil_pasta`, `fry_guanciale`, `make_egg_sauce`, `grate_cheese`, `combine`

## How the graph works

Each role is a node. State flows between them as a typed dict. The expeditor's plan drives a conditional loop -- the graph routes back through `run_doer` until the plan is exhausted, then moves on to summarise and deliver.

```text
dispatcher_receive
       ↓
expeditor_plan
       ↓
  run_doer ←--┐
       │      │ (more doers)
       └------┘
       ↓ (done)
expeditor_summarise
       ↓
dispatcher_deliver
```

The graph is compiled and inspectable. To print a Mermaid diagram of the pipeline:

```python
app = build_graph()
print(app.get_graph().draw_mermaid())
```

## Requirements 2026-05-30

- [Ollama](https://ollama.com) running locally
- `qwen2.5-coder:3b` pulled (`ollama pull qwen2.5-coder:3b`)
- Python 3.14
- `pip install -r requirements.txt`
  - Note: `langgraph`, `langchain-ollama`, and `ollama` are essential; `requirements-pinned.txt` pins all the exact versions tested

## Run it

```bash
python carbonara.py
```

## Live tuning

At the top of `carbonara.py`:

```python
MODEL        = "qwen2.5-coder:3b"
TEMPERATURE  = 0.3
```

Bump `TEMPERATURE` to `0.8` during a demo to show wilder, less predictable output.

## What you will see

```text
[DISPATCHER]  received order: "make me a carbonara"

[EXPEDITOR]   thinking about: "make me a carbonara"
              ... reasoning trace printed here ...

[DOER boil_pasta]     Boil 200g spaghetti to al dente
[DOER boil_pasta]     Spaghetti is al dente and drained, standing by.

[DOER fry_guanciale]  Render guanciale until crisp
[DOER fry_guanciale]  Guanciale is crisp, fat rendered into the pan.

... and so on through to combine ...

[DISPATCHER]  Your carbonara is ready -- silky, rich, no scrambled eggs.
```

## Why LangGraph

The graph structure makes the pipeline a first-class artifact. You can visualise it, stream it node by node, and add checkpointing for free. Useful when the workflow grows beyond a linear sequence -- retries, branches, human-in-the-loop pauses.

```python
# Stream node-by-node
for chunk in app.stream(initial_state):
    print(chunk)

# Add persistence
from langgraph.checkpoint.memory import MemorySaver
app = graph.compile(checkpointer=MemorySaver())
```

---

**East Van AI** · AI for the rest of us! · Vancouver, BC, Canada

[github.com/east-van-ai](https://github.com/east-van-ai) · <east-van-ai@proton.me>

MIT License · Copyright (c) 2026 Go Nakamaru
