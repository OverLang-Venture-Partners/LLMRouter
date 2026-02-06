# ClawBot Router

An OpenAI-compatible API server with intelligent LLM routing capabilities. ClawBot Router can automatically select the best LLM backend based on your query using various routing strategies.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI API endpoints
- **Multiple Routing Strategies**: Built-in strategies and 26+ ML-based routers
- **Streaming Support**: Full streaming response support
- **Model Prefix**: Optional model name prefix in responses for debugging
- **Multi-API Key Support**: Load balancing across multiple API keys

## Prerequisites

### 1. Install Python Dependencies

```bash
pip install fastapi uvicorn httpx pydantic pyyaml
```

### 2. Install OpenClaw (Required for Slack/Discord Integration)

If you want to use the gateway feature to connect with Slack, Discord, or other messaging platforms:

```bash
# Install OpenClaw CLI
npm install -g openclaw

# Configure Slack bot token
openclaw config set slack.token "xoxb-your-slack-bot-token"

# Or configure Discord bot token
openclaw config set discord.token "your-discord-bot-token"
```

**Note:** If you only need the LLM routing API (without messaging platform integration), you can skip OpenClaw installation and use `--no-gateway` flag.

### 3. Configure OpenClaw to Use ClawBot Router

Edit `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "clawbot": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "not-needed",
        "models": [{"id": "auto", "name": "ClawBot Router"}]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "clawbot/auto"}
    }
  }
}
```

## Quick Start

### Starting the Services

Use the startup script to launch both ClawBot Router and OpenClaw Gateway:

```bash
# Start with default configuration
./scripts/start-clawbot.sh

# Start with a specific router
./scripts/start-clawbot.sh -r llm

# Start on a custom port
./scripts/start-clawbot.sh -p 9000

# Start without OpenClaw Gateway
./scripts/start-clawbot.sh --no-gateway
```

### Stopping the Services

```bash
./scripts/stop-clawbot.sh
```

## Prerequisites: API Keys Setup

Before starting ClawBot Router, you need to configure API keys in `clawbot_router/config.yaml`:

### Required API Keys

| Provider | Key Format | How to Get |
|----------|------------|------------|
| **NVIDIA** | `nvapi-xxx...` | Sign up at [NVIDIA NGC](https://build.nvidia.com/) and generate an API key |
| **OpenAI** (optional) | `sk-xxx...` | Get from [OpenAI Platform](https://platform.openai.com/api-keys) |
| **Anthropic** (optional) | `sk-ant-xxx...` | Get from [Anthropic Console](https://console.anthropic.com/) |

### Configuration Example

Edit `clawbot_router/config.yaml`:

```yaml
api_keys:
  nvidia:
    - nvapi-YOUR_NVIDIA_API_KEY_1
    - nvapi-YOUR_NVIDIA_API_KEY_2  # Optional: add more keys for load balancing
  openai: ${OPENAI_API_KEY}        # Or set directly: sk-your-openai-key
  anthropic: ${ANTHROPIC_API_KEY}  # Or set directly: sk-ant-your-anthropic-key
```

**Notes:**
- NVIDIA API key is **required** for the default configuration (all default models use NVIDIA)
- You can use environment variables (`${VAR_NAME}`) or hardcode the keys directly
- Multiple keys for the same provider enable automatic key rotation for load balancing
- For security, consider using a local config file: copy `config.yaml` to `config.local.yaml` (gitignored)

## Command Line Options

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Config file path (default: `clawbot_router/config.yaml`) |
| `-p, --port PORT` | Router port (default: 8000) |
| `-r, --router NAME` | Router name (see available routers below) |
| `--router-config FILE` | Router-specific config file path |
| `--no-gateway` | Don't start OpenClaw Gateway |
| `--no-prefix` | Don't add model name prefix to responses |
| `--list-routers` | List all available routers |
| `-h, --help` | Show help message |

## Available Routing Strategies

### Built-in Strategies

Use these with `-r <strategy>`:

| Strategy | Description |
|----------|-------------|
| `random` | Randomly select a model from the pool |
| `round_robin` | Rotate through models in order |
| `rules` | Keyword-based routing rules |
| `llm` | Use a small LLM to decide which model to use |

### ML-based Routers (LLMRouter)

Use these with `-r <router_name>`:

| Router | Description |
|--------|-------------|
| `knnrouter` | K-Nearest Neighbors based routing |
| `mlprouter` | Multi-Layer Perceptron router |
| `svmrouter` | Support Vector Machine router |
| `mfrouter` | Matrix Factorization router |
| `elorouter` | ELO rating based router |
| `dcrouter` | Deep Clustering router |
| `graphrouter` | Graph Neural Network router |
| `gmtrouter` | Gaussian Mixture router |
| `causallmrouter` | Causal LM based router |
| `personalizedrouter` | Personalized routing based on user history |
| `knnmultiroundrouter` | KNN with multi-round conversation support |
| `llmmultiroundrouter` | LLM with multi-round conversation support |
| `hybridllm` | Hybrid LLM routing |
| `automixrouter` | Automatic model mixing |
| `routerdc` | Router with domain classification |
| `router_r1` | Router R1 implementation |
| `randomrouter` | Random selection (ML version) |
| `thresholdrouter` | Threshold-based selection |
| `largest_llm` | Always select the largest model |
| `smallest_llm` | Always select the smallest model |

## Configuration

### Main Configuration File (`config.yaml`)

```yaml
# Server settings
serve:
  host: "0.0.0.0"
  port: 8000
  show_model_prefix: true  # Add [model_name] prefix to responses

# Routing strategy configuration
router:
  strategy: llm           # Options: random, round_robin, rules, llm, llmrouter
  provider: nvidia
  base_url: https://integrate.api.nvidia.com/v1
  model: meta/llama-3.1-8b-instruct  # Small model for routing decisions

# API keys for providers
api_keys:
  nvidia:
    - nvapi-xxxxx
    - nvapi-yyyyy
  openai: ${OPENAI_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}

# LLM backend configuration
llms:
  llama-3.1-8b:
    description: "Fast responses, daily chat"
    provider: nvidia
    model: meta/llama-3.1-8b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    input_price: 0.2      # Cost per 1M input tokens
    output_price: 0.2     # Cost per 1M output tokens
    max_tokens: 1024
    context_limit: 128000
```

### Configuration Parameters

#### Server Settings (`serve`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | `0.0.0.0` | Host address to bind |
| `port` | int | `8000` | Port number |
| `show_model_prefix` | bool | `true` | Add `[model_name]` prefix to responses |

#### Router Settings (`router`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy` | string | Routing strategy to use |
| `provider` | string | LLM provider for routing decisions (when using `llm` strategy) |
| `base_url` | string | API base URL for routing LLM |
| `model` | string | Model to use for routing decisions |

#### LLM Backend Settings (`llms.<name>`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | string | Human-readable description of the model's strengths |
| `provider` | string | LLM provider (nvidia, openai, anthropic) |
| `model` | string | Full model identifier |
| `base_url` | string | API endpoint URL |
| `input_price` | float | Cost per 1M input tokens (USD) |
| `output_price` | float | Cost per 1M output tokens (USD) |
| `max_tokens` | int | Maximum output tokens |
| `context_limit` | int | Maximum context window size |

### ML Router Configuration

For ML-based routers, you can provide a separate config file:

```bash
./scripts/start-clawbot.sh -r knnrouter --router-config custom_routers/knnrouter/config.yaml
```

Example ML router config (`custom_routers/randomrouter/config.yaml`):

```yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'
  routing_data_test: 'data/example_data/routing_data/default_routing_test_data.jsonl'

metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0

hparam:
  seed: null  # null = true random, set integer for reproducibility

api_endpoint: 'https://integrate.api.nvidia.com/v1'
```

#### Hyperparameters (`hparam`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int/null | Random seed. `null` for true randomness, integer for reproducibility |

## API Endpoints

Once running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/routers` | GET | List available routing strategies |

### Example API Call

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Usage Examples

```bash
# Use random built-in strategy
./scripts/start-clawbot.sh -r random

# Use LLM-based routing (uses small model to decide)
./scripts/start-clawbot.sh -r llm

# Use KNN ML router
./scripts/start-clawbot.sh -r knnrouter

# Use random ML router on port 9000
./scripts/start-clawbot.sh -r randomrouter -p 9000

# Custom config with ML router
./scripts/start-clawbot.sh -c my_config.yaml -r mlprouter

# Start without model prefix in responses
./scripts/start-clawbot.sh -r llm --no-prefix

# Router only, no OpenClaw Gateway
./scripts/start-clawbot.sh --no-gateway
```

## OpenClaw Integration

Add to `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "clawbot": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "not-needed",
        "models": [{"id": "auto", "name": "ClawBot Router"}]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "clawbot/auto"}
    }
  }
}
```

## Architecture

### Full Integration (with Slack/Discord)

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ Slack/Discord│ ──► │  OpenClaw Gateway    │ ──► │  ClawBot Router │
│    User      │     │  (Port 18789)        │     │  (Port 8000)    │
└──────────────┘     │  (requires openclaw) │     │                 │
                     └──────────────────────┘     └────────┬────────┘
                                                           │
                              ┌─────────────────┬──────────┼──────────┐
                              ▼                 ▼          ▼          ▼
                        ┌──────────┐     ┌──────────┐ ┌──────────┐ ┌──────────┐
                        │ LLaMA    │     │ Mistral  │ │ Mixtral  │ │ Nemotron │
                        │ 3.1-8B   │     │ 7B       │ │ 8x22B    │ │ 49B      │
                        └──────────┘     └──────────┘ └──────────┘ └──────────┘
```

### Standalone Mode (API only)

```
┌─────────────────────────────────────────────────────────┐
│                     Client Request                       │
│            (curl, Python, any OpenAI client)            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   ClawBot Router                         │
│                  (Port 8000)                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Routing Strategy                    │    │
│  │  • random / round_robin / rules / llm           │    │
│  │  • ML Routers (knn, svm, mlp, etc.)            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ LLaMA    │    │ Mistral  │    │ Mixtral  │
    │ 3.1-8B   │    │ 7B       │    │ 8x22B    │
    └──────────┘    └──────────┘    └──────────┘
```

**Note:** OpenClaw Gateway is a separate project. Install it with `npm install -g openclaw` if you need Slack/Discord integration. Use `--no-gateway` flag if you only need the routing API.

## Directory Structure

```
LLMRouter/
├── scripts/
│   ├── start-clawbot.sh    # Start ClawBot Router + OpenClaw Gateway
│   └── stop-clawbot.sh     # Stop all services
├── clawbot_router/
│   ├── __init__.py         # Module exports
│   ├── __main__.py         # CLI entry point
│   ├── server.py           # FastAPI server
│   ├── config.py           # Configuration classes
│   ├── config.yaml         # Main configuration file
│   ├── routers.py          # Routing strategies
│   └── README.md           # This file
└── custom_routers/
    └── randomrouter/       # Example custom router
        └── config.yaml
```

## Logs

- Router log: `/tmp/clawbot.log`
- Gateway log: `/tmp/openclaw-gateway.log`

View logs in real-time:
```bash
tail -f /tmp/clawbot.log
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, the startup script will automatically stop the old process. You can also manually kill it:

```bash
pkill -f "clawbot_router"
```

### Router Not Loading

1. Check if the router name is correct using `--list-routers`
2. Verify the router config file exists
3. Check logs for detailed error messages

### Slow Response with LLM Strategy

The `llm` strategy makes two API calls:
1. First call to the routing LLM to decide which model to use
2. Second call to the selected model for the actual response

This adds latency but provides intelligent routing. For faster responses, use `random` or `round_robin`.

## License

See the main project LICENSE file.
