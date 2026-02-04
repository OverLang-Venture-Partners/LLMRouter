# Personalized Router (GNN-Based Personalized Router)

## Overview

The **Personalized Router** uses Graph Neural Networks (GNNs) to make personalized routing decisions for different users. It extends the Graph Router by incorporating user features into the graph structure, allowing the model to learn user-specific routing preferences.

## Paper Reference

This router implements the **PersonalizedRouter** approach:

- **[PersonalizedRouter: User-Adaptive LLM Selection via Graph Neural Networks](https://arxiv.org/abs/2511.16883)**
  - (2024). arXiv:2511.16883.
  - Constructs heterogeneous graph with task, query, user, and LLM nodes for personalized routing.
  - Learns user-specific routing patterns through personalized message passing.

## How It Works

### Graph Structure

```
                      User Nodes (one-hot encoding)
                            |
                            |
        Query Nodes ─── edges(performance) ──→ LLM Nodes
                            |
                            |
                      Task Nodes

              GNN Message Passing
                    ↓
         Personalized Predictions
```

**Node Types:**
- **Query Nodes**: Each query is a node with Longformer embedding features
- **LLM Nodes**: Each LLM is a node with learned/provided embeddings
- **User Nodes**: Each user is a node with one-hot features (enables personalization)
- **Task Nodes**: Each task has an embedding (supports multi-task learning)
- **Edges**: Connect queries to LLMs, weighted by performance scores
- **Additional Edges**: Connect LLMs within the same family

### Routing Mechanism

1. **Personalized Graph Construction**:
   - Create bipartite graph: queries on one side, LLMs on the other
   - Add user nodes with one-hot encoding
   - Expand the graph for each user (each user has their own query-LLM interactions)
   - Add edges from each query to all LLMs

2. **GNN Forward Pass**:
   - Aggregate information from neighboring nodes (queries, LLMs, users)
   - Update node representations using message passing
   - Apply graph convolution layers with user-specific embeddings

3. **Personalized Prediction**:
   - For each user-query-LLM combination, predict suitability score
   - Select LLM with highest predicted score for that specific user
   - Different users may receive different recommendations for the same query

### Training Strategy

Uses **edge masking** for training with personalization:
- Mask a portion of edges (e.g., 30%) for each user
- Train GNN to predict performance on masked edges
- Evaluation on validation set with different masked edges
- Same query can have different optimal LLMs for different users

## Configuration Parameters

### Training Hyperparameters (`hparam` in config)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | `64` | Hidden layer dimension for GNN. Controls model capacity. Range: 32-256. |
| `edge_dim` | int | `1` | Edge feature dimension (1 for performance, 2 for [cost, effect]). |
| `user_num` | int | `1000` | Number of users for personalization. Each user gets a unique node. |
| `num_task` | int | `4` | Number of tasks for multi-task learning. |
| `learning_rate` | float | `0.001` | Learning rate for AdamW optimizer. Range: 0.0001-0.01. |
| `weight_decay` | float | `0.0001` | L2 regularization weight decay. Prevents overfitting. |
| `train_epoch` | int | `100` | Number of training epochs. Increase for larger graphs. |
| `batch_size` | int | `4` | Number of masked samples per gradient step. |
| `train_mask_rate` | float | `0.3` | Fraction of edges to mask during training (0.0-1.0). |
| `split_ratio` | list | `[0.6, 0.2, 0.2]` | Ratio for train/val/test split. |
| `llm_family` | list | `[]` | List of LLM families for additional edges (e.g., ["gpt", "claude"]). |
| `random_state` | int | `42` | Random seed for reproducibility. |

### Data Paths

| Parameter | Description |
|-----------|-------------|
| `routing_data_train` | Training query-LLM performance data (JSONL) |
| `query_data_train` | Training queries (JSONL) |
| `query_embedding_data` | Pre-computed Longformer query embeddings (PyTorch tensor) |
| `llm_data` | LLM information with optional embeddings (JSON) |
| `llm_embedding_data` | Pre-computed LLM embeddings (JSON) |

### Model Paths

| Parameter | Purpose |
|-----------|---------|
| `save_model_path` | Where to save trained GNN model |
| `load_model_path` | Model to load for inference |
| `ini_model_path` | Initial model weights (optional) |

## CLI Usage

The Personalized Router can be used via the `llmrouter` command-line interface:

### Training

```bash
# Train the Personalized router (GPU recommended)
llmrouter train --router personalizedrouter --config configs/model_config_train/personalizedrouter.yaml --device cuda

# Train with quiet mode
llmrouter train --router personalizedrouter --config configs/model_config_train/personalizedrouter.yaml --device cuda --quiet
```

### Inference

```bash
# Route a single query with user_id
llmrouter infer --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml \
    --query "Explain quantum mechanics" --user-id 0

# Route queries from a file (with user_id in each query)
llmrouter infer --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml \
    --input queries.jsonl --output results.json

# Route only (without calling LLM API)
llmrouter infer --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml \
    --query "What is machine learning?" --user-id 1 --route-only
```

### Interactive Chat

```bash
# Launch chat interface
llmrouter chat --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml

# Launch with custom port
llmrouter chat --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml --port 8080

# Create a public shareable link
llmrouter chat --router personalizedrouter --config configs/model_config_test/personalizedrouter.yaml --share
```

---

## Usage Examples

### Training

```python
from llmrouter.models import PersonalizedRouter
from llmrouter.models.personalizedrouter.trainer import PersonalizedRouterTrainer

router = PersonalizedRouter(yaml_path="configs/model_config_train/personalizedrouter.yaml")
trainer = PersonalizedRouterTrainer(router=router, device="cuda")
trainer.train()
```

### Inference

```python
from llmrouter.models import PersonalizedRouter

router = PersonalizedRouter(yaml_path="configs/model_config_test/personalizedrouter.yaml")

# Single query with user personalization
query = {"query": "Explain quantum mechanics", "user_id": 0}
result = router.route_single(query)
print(f"Selected for user 0: {result['model_name']}")

# Different user might get different recommendation
query2 = {"query": "Explain quantum mechanics", "user_id": 1}
result2 = router.route_single(query2)
print(f"Selected for user 1: {result2['model_name']}")
```

### Batch Inference

```python
from llmrouter.models import PersonalizedRouter

router = PersonalizedRouter(yaml_path="configs/model_config_test/personalizedrouter.yaml")

# Batch queries with different users
batch = [
    {"query": "What is the capital of France?", "user_id": 0},
    {"query": "Who wrote Romeo and Juliet?", "user_id": 1},
    {"query": "How does photosynthesis work?", "user_id": 2},
]

results = router.route_batch(batch=batch)
for result in results:
    print(f"User {result.get('user_id')}: {result['query'][:30]}... -> {result['model_name']}")
```

## YAML Configuration Example

```yaml
data_path:
  query_data_train: 'data/example_data/query_data/default_query_train.jsonl'
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'
  query_embedding_data: 'data/example_data/routing_data/query_embeddings_longformer.pt'
  routing_data_train: 'data/example_data/routing_data/default_routing_train_data.jsonl'
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  llm_embedding_data: 'data/example_data/llm_candidates/default_llm_embeddings.json'

model_path:
  save_model_path: 'saved_models/personalizedrouter/personalizedrouter.pt'
  load_model_path: 'saved_models/personalizedrouter/personalizedrouter.pt'

hparam:
  embedding_dim: 64
  edge_dim: 1
  user_num: 1000
  num_task: 4
  learning_rate: 0.001
  weight_decay: 0.0001
  train_epoch: 100
  batch_size: 4
  train_mask_rate: 0.3
  split_ratio: [0.6, 0.2, 0.2]
  llm_family: []
  random_state: 42

metric:
  weights:
    performance: 1
```

## Advantages

- ✅ **Personalization**: Learns different routing strategies for different users
- ✅ **User Features**: Incorporates user-specific information into routing decisions
- ✅ **Multi-task Support**: Supports multiple tasks with task embeddings
- ✅ **Relational Learning**: Captures complex query-model relationships per user
- ✅ **Graph Structure**: Leverages network effects and transitivity
- ✅ **Flexible**: Can incorporate additional node/edge features

## Limitations

- ❌ **Computational Cost**: GNN training slower than simpler methods
- ❌ **Cold Start**: New users need to be added to the graph
- ❌ **Memory Usage**: Requires storing embeddings for all users
- ❌ **Hyperparameter Sensitivity**: Many architectural choices

## When to Use Personalized Router

**Good Use Cases:**
- Multiple users with distinct preferences
- Want to learn user-specific routing patterns
- Have user interaction history data
- Need multi-task learning support
- Query-model relationships vary by user

**Alternatives:**
- Single user scenario → Use Graph Router
- Simple relationships → Use MLP/SVM Router
- Small datasets → Use KNN Router
- Need fast training → Use ELO Router

## Related Routers

- **Graph Router**: Base GNN router without personalization
- **RouterDC**: Also uses structured learning but with contrastive loss
- **MF Router**: Learns latent spaces but without graph structure
- **MLP Router**: Standard neural network, no graph

---

For questions or issues, please refer to the main LLMRouter documentation or open an issue on GitHub.
