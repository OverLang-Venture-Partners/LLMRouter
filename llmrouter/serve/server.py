"""
LLMRouter OpenAI-Compatible Server
==================================
提供 OpenAI 兼容的 API，可直接与 OpenClaw 等前端集成。

启动方式:
    llmrouter serve --config serve_config.yaml

或者代码调用:
    from llmrouter.serve import create_app, run_server
    app = create_app(config_path="serve_config.yaml")
    run_server(app, port=8000)
"""

import json
import os
import sys
import re
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any, List

# FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import httpx
    import uvicorn
except ImportError:
    print("Please install: pip install fastapi uvicorn httpx pydantic")
    sys.exit(1)

from .config import ServeConfig, LLMConfig


# ============================================================
# Request/Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False


# ============================================================
# Router Integration
# ============================================================

class RouterAdapter:
    """LLMRouter 适配器"""

    def __init__(self, router_name: str, config_path: Optional[str] = None):
        self.router_name = router_name
        self.config_path = config_path
        self.router = None
        self._load_router()

    def _load_router(self):
        """加载 router"""
        try:
            # 添加 LLMRouter 根目录到路径
            llmrouter_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if llmrouter_root not in sys.path:
                sys.path.insert(0, llmrouter_root)

            if self.router_name == "randomrouter":
                from custom_routers.randomrouter.router import RandomRouter
                self.router = RandomRouter(self.config_path)

            elif self.router_name == "thresholdrouter":
                from custom_routers.thresholdrouter.router import ThresholdRouter
                self.router = ThresholdRouter(self.config_path)

            else:
                # 动态加载
                import importlib
                module = importlib.import_module(f"custom_routers.{self.router_name}.router")
                for attr in dir(module):
                    if "router" in attr.lower() and not attr.startswith("_"):
                        RouterClass = getattr(module, attr)
                        if hasattr(RouterClass, "route_single"):
                            self.router = RouterClass(self.config_path)
                            break

            print(f"✅ Router loaded: {self.router_name}")

        except Exception as e:
            print(f"⚠️ Failed to load router '{self.router_name}': {e}")
            print("   Falling back to random selection")
            self.router = None

    def route(self, query: str, available_models: List[str]) -> str:
        """选择模型"""
        if self.router is None:
            import random
            return random.choice(available_models)

        try:
            result = self.router.route_single({"query": query})
            model_name = result.get("model_name") or result.get("predicted_llm")

            # 检查模型是否可用
            if model_name in available_models:
                return model_name

            # 模糊匹配
            for m in available_models:
                if model_name and (model_name.lower() in m.lower() or m.lower() in model_name.lower()):
                    return m

            # 回退
            return available_models[0]

        except Exception as e:
            print(f"[Router] Error: {e}")
            return available_models[0]


# ============================================================
# LLM Backend
# ============================================================

class LLMBackend:
    """LLM 后端调用"""

    def __init__(self, config: ServeConfig):
        self.config = config

    async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
                   temperature: Optional[float] = None, stream: bool = False):
        """调用 LLM"""
        if llm_name not in self.config.llms:
            raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

        llm_config = self.config.llms[llm_name]
        api_key = llm_config.api_key or self.config.get_api_key(llm_config.provider)

        if stream:
            return self._call_streaming(llm_config, messages, max_tokens, temperature, api_key)
        else:
            return await self._call_sync(llm_config, messages, max_tokens, temperature, api_key)

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: str) -> Dict:
        """同步调用"""
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": messages,
                "max_tokens": min(max_tokens, llm.max_tokens),
            }
            if temperature is not None:
                body["temperature"] = temperature

            resp = await client.post(
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

            return resp.json()

    async def _call_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                              temperature: Optional[float], api_key: str) -> AsyncGenerator:
        """流式调用"""
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": messages,
                "max_tokens": min(max_tokens, llm.max_tokens),
                "stream": True
            }
            if temperature is not None:
                body["temperature"] = temperature

            async with client.stream(
                "POST",
                f"{llm.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    yield f'data: {json.dumps({"error": error.decode()[:200]})}\n\n'
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"


# ============================================================
# FastAPI App
# ============================================================

def create_app(config: ServeConfig = None, config_path: str = None) -> FastAPI:
    """创建 FastAPI 应用"""

    if config is None and config_path:
        config = ServeConfig.from_yaml(config_path)
    elif config is None:
        config = ServeConfig()

    app = FastAPI(
        title="LLMRouter Serve",
        description="OpenAI-compatible API with intelligent routing",
        version="1.0.0"
    )

    # 初始化组件
    router_adapter = RouterAdapter(
        router_name=config.router_name,
        config_path=config.router_config_path
    )
    llm_backend = LLMBackend(config)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "router": config.router_name,
            "llms": list(config.llms.keys())
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {"id": name, "object": "model"}
                for name in config.llms.keys()
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # 提取用户查询
        user_query = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_query = m["content"][:500]
                break

        # 选择模型
        available_models = list(config.llms.keys())
        if request.model == "auto" or request.model not in available_models:
            selected_model = router_adapter.route(user_query, available_models)
            print(f"[Router] Query: '{user_query[:50]}...' → {selected_model}")
        else:
            selected_model = request.model

        # 调用 LLM
        if request.stream:
            async def generate():
                first_chunk = True
                async for chunk in llm_backend.call(
                    selected_model, messages, request.max_tokens,
                    request.temperature, stream=True
                ):
                    # 添加模型前缀
                    if first_chunk and config.show_model_prefix and "content" in chunk:
                        try:
                            data = json.loads(chunk[6:])
                            if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + data["choices"][0]["delta"]["content"]
                                chunk = f"data: {json.dumps(data)}\n\n"
                        except:
                            pass
                        first_chunk = False
                    yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            result = await llm_backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=False
            )

            # 添加模型前缀
            if config.show_model_prefix and result.get("choices"):
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    result["choices"][0]["message"]["content"] = f"[{selected_model}] {content}"

            result["model"] = selected_model
            return result

    return app


def run_server(app: FastAPI = None, config_path: str = None, host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    if app is None:
        app = create_app(config_path=config_path)

    print(f"""
============================================================
  LLMRouter Serve
============================================================
  Server: http://{host}:{port}
  API:    http://{host}:{port}/v1/chat/completions
  Health: http://{host}:{port}/health
============================================================
""")

    uvicorn.run(app, host=host, port=port)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMRouter Serve")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    run_server(config_path=args.config, host=args.host, port=args.port)
