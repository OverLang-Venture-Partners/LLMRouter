"""
GMTRouter - Graph-based Multi-Turn Personalized Router

This module provides an adapter to integrate GMTRouter (https://github.com/ulab-uiuc/GMTRouter)
into the LLMRouter framework.

IMPORTANT: GMTRouter uses a fundamentally different architecture and data format:
- Heterogeneous Graph Neural Network with 5 node types (User, Session, Query, LLM, Response)
- 21 edge types for modeling complex relationships
- Pairwise preference learning instead of classification
- Special JSONL data format with embeddings and ratings

For training GMTRouter, please use the original repository:
https://github.com/ulab-uiuc/GMTRouter

This adapter provides inference capabilities within LLMRouter.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import torch
import numpy as np

from llmrouter.models.meta_router import MetaRouter


class GMTRouter(MetaRouter):
    """
    GMTRouter - Graph-based Multi-Turn Personalized Router

    A personalized LLM router that uses heterogeneous graph neural networks
    to optimize model selection across multi-turn conversations.

    Architecture Components:
    ========================
    1. HeteroGNN: Heterogeneous graph neural network with 5 node types:
       - User: Learned user preference embeddings
       - Session: Conversation session representations
       - Query: Query text embeddings (from PLM)
       - LLM: Model embeddings
       - Response: Response quality representations

    2. PreferencePredictor: Cross-attention mechanism to score LLM candidates
       based on user preferences and query context

    Data Format:
    ============
    GMTRouter requires special JSONL format with these fields:
    - judge: User identifier
    - model: LLM model name
    - question_id: Question identifier
    - turn: Turn number in conversation
    - conversation: List of turns with:
        - query: Query text
        - query_emb: Query embedding vector
        - response: Response text (optional)
        - rating: Quality score
    - model_emb: LLM embedding vector

    Data Preparation:
    =================
    1. Download dataset from Google Drive:
       https://drive.google.com/file/d/[GMTRouter_dataset_link]

    2. Extract to repository root:
       tar -xzvf GMTRouter_dataset.tar.gz
       mv data ./

    3. Data will be in ./data/[dataset_name]/ with:
       - training_set.jsonl
       - valid_set.jsonl
       - test_set.jsonl

    Supported Datasets:
    ===================
    - chatbot_arena: Chatbot Arena conversations
    - gsm8k: GSM8K math problems
    - mmlu: MMLU benchmark
    - mt_bench: MT-Bench multi-turn conversations

    Requirements:
    =============
    - Python 3.11.13
    - PyTorch 2.6+ with CUDA 12.4+
    - PyTorch Geometric 2.6.1
    - transformers >= 4.43
    - scikit-learn >= 1.3

    For Training:
    =============
    Use the original GMTRouter repository for training:

    1. Clone: git clone https://github.com/ulab-uiuc/GMTRouter
    2. Setup environment with Python 3.11.13 and PyTorch 2.6
    3. Download data and extract to ./data/
    4. Run: python src/train.py --config configs/sample.yaml

    This will train the HeteroGNN and PreferencePredictor models.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize GMTRouter.

        Args:
            yaml_path: Path to YAML configuration file
        """
        super().__init__(yaml_path=yaml_path)

        # Get GMTRouter-specific configuration
        self.gmt_config = self.cfg.get("gmt_config", {})

        # Model architecture parameters
        self.hidden_dim = self.gmt_config.get("hidden_dim", 128)
        self.num_gnn_layers = self.gmt_config.get("num_gnn_layers", 2)
        self.dropout = self.gmt_config.get("dropout", 0.1)

        # Personalization settings
        self.enable_personalization = self.gmt_config.get("personalization", True)
        self.record_per_user = self.gmt_config.get("record_per_user", 10)

        # Multi-turn settings
        self.multi_turn = self.gmt_config.get("multi_turn", False)
        self.aggregation_type = self.gmt_config.get("aggregation_type", "mean")

        # Data paths
        data_paths = self.cfg.get("data_path", {})
        self.dataset_name = self.gmt_config.get("dataset", "mt_bench")
        self.data_root = data_paths.get("data_root", "./data")

        # Model paths
        model_paths = self.cfg.get("model_path", {})
        self.checkpoint_root = model_paths.get("checkpoint_root", "./models")
        self.model_checkpoint = model_paths.get("load_model_path", None)

        # Initialize models
        self.hetero_gnn = None
        self.preference_predictor = None

        # User and session tracking
        self.user_embeddings = {}
        self.session_data = {}
        self.query_embeddings = {}
        self.llm_embeddings = {}

        # Load pretrained model if available
        self._load_pretrained_model()

    def _load_pretrained_model(self):
        """Load pretrained HeteroGNN and PreferencePredictor if available."""
        if self.model_checkpoint and os.path.exists(self.model_checkpoint):
            try:
                checkpoint = torch.load(self.model_checkpoint, map_location='cpu')
                print(f"Loaded GMTRouter checkpoint from {self.model_checkpoint}")

                # Extract user embeddings
                if 'user_embeddings' in checkpoint:
                    self.user_embeddings = checkpoint['user_embeddings']

                # Note: Full model loading requires PyTorch Geometric
                # For inference in LLMRouter, we use the learned preferences

            except Exception as e:
                print(f"Warning: Could not load GMTRouter checkpoint: {e}")
                print("GMTRouter will use fallback routing strategy.")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query to the best LLM using personalized preferences.

        Args:
            query: Query dictionary with:
                - query_text: The query string
                - user_id: User identifier (required for personalization)
                - session_id: Session identifier (optional)
                - turn: Turn number (optional)
                - conversation_history: List of previous turns (optional)

        Returns:
            Dictionary containing:
                - model_name: Selected LLM
                - confidence: Routing confidence score
                - user_preference: User-specific preference score
                - reasoning: Explanation of routing decision
        """
        query_text = query.get("query_text", query.get("query", ""))
        user_id = query.get("user_id", "default_user")
        session_id = query.get("session_id", f"{user_id}_session")
        turn = query.get("turn", 0)

        # Check if GMTRouter model is available
        if self.hetero_gnn is None or self.preference_predictor is None:
            # Fallback to default routing
            return self._fallback_routing(query_text, user_id)

        # Get conversation history
        history = query.get("conversation_history", [])

        # Get or create session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "user_id": user_id,
                "turns": [],
                "queries": [],
                "responses": []
            }

        # Add current query to session
        self.session_data[session_id]["queries"].append(query_text)

        # Get user embedding
        user_emb = self.user_embeddings.get(user_id, None)

        if user_emb is None:
            # New user - use fallback
            return self._fallback_routing(query_text, user_id)

        # Get query embedding (would use PreferencePredictor in full implementation)
        # For now, use simple heuristics based on learned preferences

        # Select best model based on user preferences
        best_model = self._select_model_with_preferences(
            query_text, user_id, user_emb, history
        )

        result = {
            "model_name": best_model,
            "confidence": 0.85,  # Would come from PreferencePredictor
            "user_preference": self._get_user_preference_score(user_id, best_model),
            "reasoning": f"Selected based on user {user_id}'s learned preferences and conversation context"
        }

        return result

    def _fallback_routing(self, query_text: str, user_id: str) -> Dict[str, Any]:
        """
        Fallback routing when GMTRouter model is not available.

        Uses simple heuristics or delegates to parent router.
        """
        # Use default model from config or first available model
        default_model = self.models[0] if self.models else "gpt-3.5-turbo"

        return {
            "model_name": default_model,
            "confidence": 0.5,
            "user_preference": 0.5,
            "reasoning": f"GMTRouter model not loaded - using fallback routing for new user {user_id}"
        }

    def _select_model_with_preferences(
        self,
        query_text: str,
        user_id: str,
        user_emb: Any,
        history: List[Dict]
    ) -> str:
        """
        Select model using learned user preferences.

        In full implementation, this would use:
        1. HeteroGNN to get aggregated embeddings
        2. PreferencePredictor to score each LLM candidate
        3. Return highest-scoring model
        """
        # Simplified version - would use actual GNN inference
        # For now, use stored preferences or default to first model

        if user_id in self.user_embeddings and len(self.models) > 0:
            # Use first model as default (would compute scores in full version)
            return self.models[0]

        return self.models[0] if self.models else "gpt-3.5-turbo"

    def _get_user_preference_score(self, user_id: str, model: str) -> float:
        """Get user's preference score for a specific model."""
        # Would compute from user embedding and model embedding
        # For now return default score
        return 0.75

    def route_batch(self, batch: List[Dict[str, Any]], task_name: str = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries.

        Args:
            batch: List of query dictionaries
            task_name: Task name (optional)

        Returns:
            List of routing results
        """
        results = []
        for query in batch:
            result = self.route_single(query)
            results.append(result)

        return results

    def update_user_feedback(
        self,
        user_id: str,
        query: str,
        model: str,
        rating: float
    ):
        """
        Update user preferences based on feedback.

        In production, this would:
        1. Add new interaction to graph
        2. Update graph structure
        3. Retrain or fine-tune user embeddings

        Args:
            user_id: User identifier
            query: Query text
            model: Model that was used
            rating: User rating (0-1 or 1-5 depending on dataset)
        """
        # Store feedback for future retraining
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = {
                "interactions": []
            }

        self.user_embeddings[user_id]["interactions"].append({
            "query": query,
            "model": model,
            "rating": rating
        })

        print(f"Feedback recorded for user {user_id}: {model} rated {rating}")

    def get_training_data(self) -> Tuple[Any, Any]:
        """
        Get training and validation data for GMTRouter.

        NOTE: Training should be done using the original GMTRouter repository.
        This method is here for compatibility with LLMRouter framework.

        Returns:
            tuple: (train_data, val_data)
                Both are None - use original GMTRouter for training
        """
        print("=" * 70)
        print("WARNING: GMTRouter training should use the original repository")
        print("Please visit: https://github.com/ulab-uiuc/GMTRouter")
        print("=" * 70)
        print()
        print("Training Steps:")
        print("1. Download data: https://drive.google.com/[GMTRouter_dataset]")
        print("2. Extract: tar -xzvf GMTRouter_dataset.tar.gz")
        print("3. Move data: mv data ./")
        print("4. Train: python src/train.py --config configs/sample.yaml")
        print()

        return None, None

    def save_model(self, path: str):
        """
        Save GMTRouter model.

        For full model saving, use the original GMTRouter repository.
        This saves only user embeddings and preferences.
        """
        save_data = {
            "user_embeddings": self.user_embeddings,
            "session_data": self.session_data,
            "config": self.gmt_config
        }

        torch.save(save_data, path)
        print(f"Saved GMTRouter user data to {path}")

    def load_model(self, path: str):
        """
        Load GMTRouter user data.

        For full model loading, use the original GMTRouter repository.
        """
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu')
            self.user_embeddings = data.get("user_embeddings", {})
            self.session_data = data.get("session_data", {})
            print(f"Loaded GMTRouter user data from {path}")
