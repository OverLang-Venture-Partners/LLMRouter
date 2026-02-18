"""
Bedrock Smart Router - Cost-aware intelligent routing for AWS Bedrock models.

Routes queries based on estimated complexity and cost optimization:
- Simple queries → nova-micro (ultra-cheap)
- Standard queries → nova-2-lite or haiku (balanced)
- Complex reasoning → nova-pro or sonnet (powerful)
- Deep analysis → opus (maximum capability)

Features:
- Rule-based complexity estimation
- Memory-augmented routing (learns from past decisions)
- Cost optimization
- Fallback handling
"""

from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


class BedrockSmartRouter(MetaRouter):
    """Smart router for Bedrock models with cost optimization and memory."""
    
    def __init__(self, yaml_path: str):
        # Use identity model (no ML needed for rule-based routing)
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)
        
        # Debug: print what we have
        print(f"DEBUG: llm_data type: {type(self.llm_data)}")
        print(f"DEBUG: llm_data value: {self.llm_data}")
        print(f"DEBUG: config keys: {self.config.keys() if self.config else 'None'}")
        
        # Get LLM names from config - try multiple approaches
        if self.llm_data and isinstance(self.llm_data, dict):
            self.llm_names = list(self.llm_data.keys())
            print(f"DEBUG: Got llm_names from llm_data: {self.llm_names}")
        elif hasattr(self, 'config') and self.config:
            if 'llms' in self.config:
                self.llm_names = list(self.config['llms'].keys())
                print(f"DEBUG: Got llm_names from config['llms']: {self.llm_names}")
            elif 'llm_data' in self.config:
                self.llm_names = list(self.config['llm_data'].keys())
                print(f"DEBUG: Got llm_names from config['llm_data']: {self.llm_names}")
            else:
                raise ValueError(f"No LLM data found. Config keys: {list(self.config.keys())}")
        else:
            raise ValueError("No config or llm_data found")
        
        # Memory system for learning from past routing decisions
        self.memory_enabled = False
        self.memory_path = None
        self.routing_history = defaultdict(list)  # query_pattern -> [model_choices]
        self._load_memory_config()
        
        # Define complexity indicators
        self.simple_indicators = [
            r'\bwhat is\b', r'\bdefine\b', r'\bwho is\b', r'\bwhen\b',
            r'\byes or no\b', r'\btrue or false\b', r'\blist\b',
            r'\bhello\b', r'\bhi\b', r'\bthanks\b', r'\bthank you\b'
        ]
        
        self.code_indicators = [
            r'\bcode\b', r'\bfunction\b', r'\bprogram\b', r'\bdebug\b',
            r'\bapi\b', r'\bsql\b', r'\bpython\b', r'\bjavascript\b',
            r'\bimplementation\b', r'\balgorithm\b'
        ]
        
        self.reasoning_indicators = [
            r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b', r'\bexplain why\b',
            r'\breasoning\b', r'\barchitecture\b', r'\bdesign\b',
            r'\bstrategy\b', r'\bapproach\b'
        ]
        
        self.deep_indicators = [
            r'\bcomprehensive\b', r'\bdetailed analysis\b', r'\bresearch\b',
            r'\bin-depth\b', r'\bthorough\b', r'\bexhaustive\b',
            r'\bphilosophical\b', r'\btheoretical\b'
        ]
    
    def _load_memory_config(self):
        """Load memory configuration from router config."""
        try:
            # Check if memory is configured in the router config
            memory_config = self.config.get('memory', {})
            self.memory_enabled = memory_config.get('enabled', False)
            
            if self.memory_enabled:
                memory_path = memory_config.get('path', '~/.llmrouter/bedrock_router_memory.jsonl')
                self.memory_path = Path(memory_path).expanduser()
                self.memory_path.parent.mkdir(parents=True, exist_ok=True)
                self._load_routing_history()
        except Exception as e:
            print(f"Warning: Could not load memory config: {e}")
            self.memory_enabled = False
    
    def _load_routing_history(self):
        """Load routing history from memory file."""
        if not self.memory_path or not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    pattern = entry.get('pattern', '')
                    model = entry.get('model', '')
                    if pattern and model:
                        self.routing_history[pattern].append(model)
        except Exception as e:
            print(f"Warning: Could not load routing history: {e}")
    
    def _save_routing_decision(self, query: str, model: str, complexity: str):
        """Save routing decision to memory for future learning."""
        if not self.memory_enabled or not self.memory_path:
            return
        
        try:
            # Extract pattern from query (first 3 words as simple pattern)
            pattern = ' '.join(query.lower().split()[:3])
            
            # Update in-memory history
            self.routing_history[pattern].append(model)
            
            # Append to file
            entry = {
                'pattern': pattern,
                'query': query[:100],  # Truncate for privacy
                'model': model,
                'complexity': complexity
            }
            
            with open(self.memory_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not save routing decision: {e}")
    
    def _get_memory_suggestion(self, query: str) -> Optional[str]:
        """Get model suggestion from routing history."""
        if not self.memory_enabled or not self.routing_history:
            return None
        
        try:
            # Extract pattern from query
            pattern = ' '.join(query.lower().split()[:3])
            
            # Check if we have history for this pattern
            if pattern in self.routing_history:
                history = self.routing_history[pattern]
                
                # Use most frequently chosen model for this pattern
                if history:
                    model_counts = defaultdict(int)
                    for model in history:
                        model_counts[model] += 1
                    
                    # Return most common model
                    most_common = max(model_counts.items(), key=lambda x: x[1])
                    return most_common[0]
        except Exception as e:
            print(f"Warning: Could not get memory suggestion: {e}")
        
        return None
    
    def _estimate_complexity(self, query: str) -> str:
        """
        Estimate query complexity.
        
        Returns:
            'simple', 'standard', 'reasoning', or 'deep'
        """
        query_lower = query.lower()
        
        # Check for deep analysis indicators
        if any(re.search(pattern, query_lower) for pattern in self.deep_indicators):
            return 'deep'
        
        # Check for reasoning indicators
        if any(re.search(pattern, query_lower) for pattern in self.reasoning_indicators):
            return 'reasoning'
        
        # Check for simple indicators
        if any(re.search(pattern, query_lower) for pattern in self.simple_indicators):
            return 'simple'
        
        # Check query length
        word_count = len(query.split())
        if word_count < 10:
            return 'simple'
        elif word_count > 50:
            return 'reasoning'
        
        # Default to standard
        return 'standard'
    
    def _select_model_by_complexity(self, complexity: str) -> str:
        """
        Select best model for given complexity level.
        
        Priority order (cost-optimized):
        - simple: nova-micro > haiku > nova-2-lite
        - standard: nova-2-lite > haiku > nova-pro
        - reasoning: nova-pro > sonnet > nova-2-lite
        - deep: opus > sonnet > nova-pro
        """
        # Define model preferences by complexity
        preferences = {
            'simple': ['nova-micro', 'haiku', 'nova-2-lite'],
            'standard': ['nova-2-lite', 'haiku', 'nova-pro'],
            'reasoning': ['nova-pro', 'sonnet', 'nova-2-lite'],
            'deep': ['opus', 'sonnet', 'nova-pro']
        }
        
        # Get preferred models for this complexity
        preferred = preferences.get(complexity, ['nova-2-lite'])
        
        # Return first available model from preferences
        for model in preferred:
            if model in self.llm_names:
                return model
        
        # Fallback to first available model
        return self.llm_names[0] if self.llm_names else 'nova-2-lite'
    
    def route_single(self, query_input: dict) -> dict:
        """
        Route a single query to the most appropriate Bedrock model.
        
        Uses hybrid approach:
        1. Check memory for similar past queries
        2. Fall back to rule-based complexity estimation
        3. Save decision for future learning
        
        Args:
            query_input: Dict with 'query' key
            
        Returns:
            Dict with routing decision
        """
        query = query_input.get("query", "")
        
        # Try memory-based routing first
        memory_suggestion = self._get_memory_suggestion(query)
        
        if memory_suggestion and memory_suggestion in self.llm_names:
            # Use memory suggestion
            selected_model = memory_suggestion
            complexity = "memory"
            routing_method = "memory"
        else:
            # Fall back to rule-based routing
            complexity = self._estimate_complexity(query)
            selected_model = self._select_model_by_complexity(complexity)
            routing_method = "rules"
        
        # Save decision to memory for future learning
        self._save_routing_decision(query, selected_model, complexity)
        
        return {
            "query": query,
            "model_name": selected_model,
            "predicted_llm": selected_model,
            "complexity": complexity,
            "routing_method": routing_method,  # 'memory' or 'rules'
            "router": "bedrock_smart_router"
        }
    
    def route_batch(self, batch: list) -> list:
        """Route a batch of queries."""
        return [self.route_single(query_input) for query_input in batch]
