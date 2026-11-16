# free-model-aggregator-02

One gateway server, two wires: any OpenAI or Anthropic client points at it and every answer is a live free-model completion, dialed through OpenRouter's free catalog with per-request model selection, JSON + SSE, tool and JSON-mode pass-through, and 429/5xx failover across the chain.

[![make.fun - free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/tanacoustic05/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/tanacoustic05/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

## Run

python3 main.py --list   # prints the registered catalog document byte for byte
python3 main.py          # serves POST /v1/chat/completions and POST /v1/messages (JSON + SSE)

Env: PORT (default 8080), OPENROUTER_API_KEY (required), MAKEFUN_ENV_URL (gateway base, default https://openrouter.ai/api/v1). Stdlib only, Python 3.13, no installs. Models may be addressed natively (vendor/model:free) or as openrouter/<id>; dialing is deterministic with a native fallback.

Built for [make.fun's free-model-aggregator-02 bounty](https://make.fun/bounty/free-model-aggregator-02) ($500 in USDC).
