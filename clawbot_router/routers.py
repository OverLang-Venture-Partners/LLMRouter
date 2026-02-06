"""
ClawBot Router Strategies
=========================
Supports multiple routing strategies:
- Built-in: rules, random, round_robin, llm
- LLMRouter ML-based: knnrouter, mlprouter, thresholdrouter, etc.
"""

import os
import sys
import random
import httpx
from typing import Dict, List, Optional, Any

# Handle both relative and direct imports
try:
    from .config import ClawBotConfig
except ImportError:
    from config import ClawBotConfig


# ============================================================
# Built-in Strategies
# ============================================================

def select_by_rules(query: str, models: List[str], rules: List[Dict]) -> str:
    """Rule-based routing using keywords"""
    query_lower = query.lower()

    for rule in rules:
        keywords = rule.get("keywords", [])
        model = rule.get("model")
        if model and model in models:
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    print(f"[Router] Rule matched: '{keyword}' → {model}")
                    return model

    # Default model
    default = rules[-1].get("default") if rules else None
    if default and default in models:
        return default
    return models[0]


def select_by_random(models: List[str], weights: Dict[str, int] = None) -> str:
    """Random routing with optional weights"""
    if weights:
        weighted_list = []
        for m in models:
            w = weights.get(m, 1)
            weighted_list.extend([m] * w)
        return random.choice(weighted_list)
    return random.choice(models)


_round_robin_index = 0

def select_by_round_robin(models: List[str]) -> str:
    """Round-robin routing"""
    global _round_robin_index
    selected = models[_round_robin_index % len(models)]
    _round_robin_index += 1
    return selected


async def select_by_llm(query: str, models: List[str], config: ClawBotConfig) -> str:
    """LLM-based routing using an LLM to decide"""
    router = config.router
    provider = router.provider or "openai"
    base_url = router.base_url or "https://api.openai.com/v1"
    model_id = router.model or "gpt-4o-mini"

    api_key = config.get_api_key(provider)
    if not api_key:
        print(f"[Router] Warning: No API key for {provider}, using random")
        return random.choice(models)

    # Build router prompt
    model_descriptions = []
    for name in models:
        llm_config = config.llms.get(name)
        if llm_config and llm_config.description:
            model_descriptions.append(f"- {name}: {llm_config.description}")
        else:
            model_descriptions.append(f"- {name}")

    prompt = f"""You are an intelligent LLM router. Choose the most suitable model for the user's query.

Available models:
{chr(10).join(model_descriptions)}

Rules:
1. Simple greetings/daily chat → cheaper models (8b, 9b size)
2. Q&A/knowledge retrieval → chatqa models
3. Instruction following/structured output → mistral models
4. Code generation/technical questions → nemotron or larger models
5. Complex reasoning/deep analysis → 70b or larger models

IMPORTANT: Only return the model name, nothing else!
Model names: {', '.join(models)}

User query: {query}"""

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            body = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,
                "temperature": 0
            }

            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=15.0
            )

            if resp.status_code != 200:
                print(f"[Router] LLM API error: {resp.status_code}")
                return models[0]

            result = resp.json()
            choice = result["choices"][0]["message"]["content"].strip().lower()

            # Clean response
            choice = choice.strip('`"\'.,!?\n\r\t ')
            choice = choice.split('\n')[0]
            choice = choice.split()[0] if choice.split() else choice

            if choice in models:
                return choice

            # Fuzzy match
            for model_name in models:
                if model_name.lower() in choice or choice in model_name.lower():
                    return model_name

            return models[0]

    except Exception as e:
        print(f"[Router] LLM error: {e}")
        return models[0]


# ============================================================
# LLMRouter ML-based Routers
# ============================================================

class LLMRouterAdapter:
    """Adapter for LLMRouter ML-based routers"""

    def __init__(self, router_name: str, config_path: Optional[str] = None,
                 model_path: Optional[str] = None):
        self.router_name = router_name.lower()
        self.config_path = config_path
        self.model_path = model_path
        self.router = None
        self._load_router()

    def _load_router(self):
        """Load the LLMRouter router"""
        # Add LLMRouter to path
        llmrouter_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if llmrouter_root not in sys.path:
            sys.path.insert(0, llmrouter_root)

        try:
            # First try custom_routers (plugin system)
            if self.router_name == "randomrouter":
                from custom_routers.randomrouter.router import RandomRouter
                self.router = RandomRouter(self.config_path)
                print(f"✅ Loaded custom router: randomrouter")
                return

            elif self.router_name == "thresholdrouter":
                from custom_routers.thresholdrouter.router import ThresholdRouter
                self.router = ThresholdRouter(self.config_path)
                print(f"✅ Loaded custom router: thresholdrouter")
                return

            # Try built-in LLMRouter routers
            try:
                from llmrouter.cli.router_inference import ROUTER_REGISTRY, load_router

                if self.router_name in ROUTER_REGISTRY:
                    if self.config_path:
                        self.router = load_router(self.router_name, self.config_path, self.model_path)
                    else:
                        # Try to instantiate without config
                        RouterClass = ROUTER_REGISTRY[self.router_name]
                        self.router = RouterClass()
                    print(f"✅ Loaded LLMRouter: {self.router_name}")
                    return

            except ImportError as e:
                print(f"[Router] LLMRouter not available: {e}")

            # Dynamic import for custom routers
            try:
                import importlib
                module = importlib.import_module(f"custom_routers.{self.router_name}.router")
                for attr in dir(module):
                    if "router" in attr.lower() and not attr.startswith("_"):
                        obj = getattr(module, attr)
                        if hasattr(obj, "route_single"):
                            self.router = obj(self.config_path) if self.config_path else obj()
                            print(f"✅ Loaded custom router: {self.router_name}")
                            return
            except ImportError:
                pass

            print(f"⚠️ Router '{self.router_name}' not found, falling back to random")

        except Exception as e:
            print(f"⚠️ Failed to load router '{self.router_name}': {e}")
            self.router = None

    def route(self, query: str, available_models: List[str]) -> str:
        """Route query to a model"""
        if self.router is None:
            return random.choice(available_models)

        try:
            result = self.router.route_single({"query": query})

            # Extract model name
            model_name = (
                result.get("model_name") or
                result.get("predicted_llm") or
                result.get("predicted_llm_name")
            )

            if model_name and model_name in available_models:
                return model_name

            # Fuzzy match
            if model_name:
                for m in available_models:
                    if model_name.lower() in m.lower() or m.lower() in model_name.lower():
                        return m

            return available_models[0]

        except Exception as e:
            print(f"[Router] Error: {e}")
            return available_models[0]


# ============================================================
# Main Router Class
# ============================================================

class ClawBotRouter:
    """Main router that supports all strategies"""

    def __init__(self, config: ClawBotConfig):
        self.config = config
        self._llmrouter_adapter: Optional[LLMRouterAdapter] = None

        # Initialize LLMRouter adapter if needed
        if config.router.strategy == "llmrouter":
            router_name = config.router.llmrouter_name
            if router_name:
                self._llmrouter_adapter = LLMRouterAdapter(
                    router_name=router_name,
                    config_path=config.router.llmrouter_config,
                    model_path=config.router.llmrouter_model_path
                )

    async def select_model(self, query: str) -> str:
        """Select model based on configured strategy"""
        models = list(self.config.llms.keys())

        if not models:
            return "default"
        if len(models) == 1:
            return models[0]

        strategy = self.config.router.strategy

        if strategy == "rules":
            selected = select_by_rules(query, models, self.config.router.rules)
            print(f"[Router] Strategy=rules → {selected}")
            return selected

        elif strategy == "random":
            selected = select_by_random(models, self.config.router.weights)
            print(f"[Router] Strategy=random → {selected}")
            return selected

        elif strategy == "round_robin":
            selected = select_by_round_robin(models)
            print(f"[Router] Strategy=round_robin → {selected}")
            return selected

        elif strategy == "llmrouter":
            if self._llmrouter_adapter:
                selected = self._llmrouter_adapter.route(query, models)
                print(f"[Router] Strategy=llmrouter({self._llmrouter_adapter.router_name}) → {selected}")
                return selected
            else:
                print(f"[Router] LLMRouter not loaded, falling back to random")
                return random.choice(models)

        elif strategy == "llm":
            selected = await select_by_llm(query, models, self.config)
            print(f"[Router] Strategy=llm → {selected}")
            return selected

        else:
            print(f"[Router] Unknown strategy '{strategy}', using random")
            return random.choice(models)

    def get_available_routers(self) -> List[str]:
        """Get list of available LLMRouter routers"""
        available = ["rules", "random", "round_robin", "llm"]

        # Add custom routers
        available.extend(["randomrouter", "thresholdrouter"])

        # Try to get LLMRouter built-in routers
        try:
            from llmrouter.cli.router_inference import ROUTER_REGISTRY
            available.extend(list(ROUTER_REGISTRY.keys()))
        except ImportError:
            pass

        return list(set(available))
