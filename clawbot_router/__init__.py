"""
ClawBot Router
==============
OpenAI-compatible API server with intelligent LLM routing.

Supports:
- Built-in strategies: rules, random, round_robin, llm
- LLMRouter ML-based routers: knnrouter, mlprouter, thresholdrouter, etc.

Usage:
    llmrouter serve --config configs/clawbot_example.yaml

Or directly:
    cd clawbot_router && python server.py --config config.yaml
"""

from .server import create_app, run_server
from .config import ClawBotConfig

__all__ = ["create_app", "run_server", "ClawBotConfig"]
