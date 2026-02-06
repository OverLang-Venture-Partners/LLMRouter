"""
Serve Configuration
===================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class LLMConfig:
    """单个 LLM 配置"""
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key: Optional[str] = None
    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: int = 4096
    context_limit: int = 32768


@dataclass
class ServeConfig:
    """服务配置"""
    # 服务器设置
    host: str = "0.0.0.0"
    port: int = 8000

    # Router 设置
    router_name: str = "randomrouter"
    router_config_path: Optional[str] = None

    # LLM 后端设置
    llms: Dict[str, LLMConfig] = field(default_factory=dict)

    # API Keys
    api_keys: Dict[str, List[str]] = field(default_factory=dict)

    # 显示模型名前缀
    show_model_prefix: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServeConfig":
        """从 YAML 文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()

        # 服务器设置
        serve_config = data.get("serve", {})
        config.host = serve_config.get("host", config.host)
        config.port = serve_config.get("port", config.port)
        config.show_model_prefix = serve_config.get("show_model_prefix", config.show_model_prefix)

        # Router 设置
        router_config = data.get("router", {})
        config.router_name = router_config.get("name", config.router_name)
        config.router_config_path = router_config.get("config_path")

        # API Keys
        config.api_keys = data.get("api_keys", {})

        # LLM 配置
        llms_data = data.get("llms", {})
        for name, llm_config in llms_data.items():
            config.llms[name] = LLMConfig(
                name=name,
                provider=llm_config.get("provider", "openai"),
                model_id=llm_config.get("model", name),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                api_key=llm_config.get("api_key"),
                input_price=llm_config.get("input_price", 0.0),
                output_price=llm_config.get("output_price", 0.0),
                max_tokens=llm_config.get("max_tokens", 4096),
                context_limit=llm_config.get("context_limit", 32768),
            )

        return config

    def get_api_key(self, provider: str) -> Optional[str]:
        """获取 API key"""
        keys = self.api_keys.get(provider, [])
        if isinstance(keys, str):
            # 支持环境变量
            if keys.startswith("${") and keys.endswith("}"):
                return os.environ.get(keys[2:-1])
            return keys
        elif isinstance(keys, list) and keys:
            # 轮询多个 key
            if not hasattr(self, "_key_index"):
                self._key_index = {}
            idx = self._key_index.get(provider, 0)
            key = keys[idx % len(keys)]
            self._key_index[provider] = idx + 1
            return key
        return None
