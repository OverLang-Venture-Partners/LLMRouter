"""
LLMRouter Serve Module
======================
提供 OpenAI 兼容的 API 服务，可直接与 OpenClaw 等前端集成。

使用方法:
    llmrouter serve --router randomrouter --config config.yaml --port 8000

或在代码中:
    from llmrouter.serve import create_app, run_server
    app = create_app(router_name="randomrouter", config_path="config.yaml")
    run_server(app, port=8000)
"""

from .server import create_app, run_server
from .config import ServeConfig

__all__ = ["create_app", "run_server", "ServeConfig"]
