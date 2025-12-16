import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from llmrouter.models.router_r1.prompt_pool import *
from llmrouter.models.meta_router import MetaRouter



class RouterR1(MetaRouter):
    """
    Router-R1
    -----------
    Example router that performs R1-like routing.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route_single()` method using the pre-trained model from official HF repo
    """

    def __init__(self, yaml_path: str):
        """
        Args:
            yaml_path (str):
                Path to YAML config for this router.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize hyperparameters
        self.model_id = self.cfg["hparam"]["model_id"]
        self.api_base = self.cfg["hparam"]["api_base"]
        self.api_key = self.cfg["hparam"]["api_key"]

    @staticmethod
    def get_query(text: str) -> Optional[str]:
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    @staticmethod
    def route(query: str, api_base: str, api_key: str) -> str:
        try:
            from llmrouter.models.router_r1.route_service import access_routing_pool
        except ImportError as e:
            raise ImportError(
                "RouterR1 requires the optional dependency `openai` (used in `route_service.py`). "
                "Install it with: `pip install openai`."
            ) from e

        ret = access_routing_pool(
            queries=[query],
            api_base=api_base,
            api_key=api_key,
        )
        return ret["result"][0]

    def route_single(self, query: Dict[str, Any]) -> str:
        """
        Perform inference on Router-R1.
        """
        # Prepare the question
        question = str(query.get("query", "")).strip()
        if not question:
            raise ValueError("RouterR1.route_single requires non-empty 'query' field")
        if not question.endswith("?"):
            question += "?"

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "RouterR1 requires the optional dependency `vllm`. Install it with: `pip install vllm`."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError("RouterR1 currently requires CUDA (vLLM GPU runtime).")

        # Model path and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tensor_parallel_size = max(1, torch.cuda.device_count())
        llm = LLM(model=self.model_id, dtype="float16", tensor_parallel_size=tensor_parallel_size)

        curr_route_template = '\n{output_text}\n<information>{route_results}</information>\n'

        # Initial prompt
        if self.model_id.lower().find("qwen") != -1:
            prompt = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
        else:
            prompt = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                tokenize=False)

        # Sampling configuration
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1024,
            stop=["</search>", "</answer>"]
        )

        cnt = 0
        print('\n\n################# [Start Reasoning + Routing] ##################\n\n')
        STOP = False
        all_output = ""

        while True:
            if cnt > 4:
                break
            outputs = llm.generate(prompt, sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            if output_text.find("<answer>") != -1:
                STOP = True
                output_text += "</answer>"
            if not STOP:
                output_text += "</search>"

            print(f"[Generation {cnt}] Output:\n{output_text}")

            tmp_query = self.get_query(output_text)
            if tmp_query:
                route_results = self.route(tmp_query, api_base=self.api_base, api_key=self.api_key)
            else:
                route_results = ''

            if not STOP:
                prompt += curr_route_template.format(output_text=output_text, route_results=route_results)
                all_output += curr_route_template.format(output_text=output_text, route_results=route_results)
            else:
                all_output += output_text + "\n"
                break

            cnt += 1

        print('\n\n################# [Output] ##################\n\n')

        print(all_output)

        print('\n\n################# [Output] ##################\n\n')
        return all_output.strip()

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries.

        Note: RouterR1 is an agentic router; this implementation simply runs
        `route_single` per query and returns a list of {query, response}.
        """
        data = batch if batch is not None else getattr(self, "query_data_test", None)
        if data is None:
            raise ValueError("No batch provided and `query_data_test` is not available.")

        results: List[Dict[str, Any]] = []
        for row in data:
            query_text = row.get("query") if isinstance(row, dict) else str(row)
            results.append(
                {
                    "query": query_text,
                    "response": self.route_single({"query": query_text}),
                }
            )
        return results
