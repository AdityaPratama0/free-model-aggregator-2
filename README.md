# free-model-aggregator-2

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/AdityaPratama0/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/AdityaPratama0/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

An OpenAI and Anthropic compatible aggregator over free models, with rating-based routing, cooldowns, and failure trails.

## Run it

Run the entry inside the make.fun harness, which supplies `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, and the provider gateway:

```sh
python3 main.py
```

Run the routing unit tests without provider access:

```sh
python3 -m unittest discover -s tests
```

Provider credentials belong in make.fun entry secrets, never in this repository.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
