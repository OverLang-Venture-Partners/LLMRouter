# Custom Router Plugin System

LLMRouter supports a plugin system that allows you to add custom router implementations without modifying the core codebase.

## Quick Start

### 1. Create Your Custom Router Directory

```bash
mkdir -p custom_routers/my_router
cd custom_routers/my_router
```

### 2. Implement Your Router

Create `router.py`:

```python
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class MyRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        # Create your model
        model = nn.Identity()  # Replace with actual model

        # Initialize parent (loads config and data)
        super().__init__(model=model, yaml_path=yaml_path)

        # Your initialization code
        self.llm_names = list(self.llm_data.keys())

    def route_single(self, query_input: dict) -> dict:
        """Route a single query."""
        # Your routing logic here
        selected_llm = self.llm_names[0]  # Example

        return {
            "query": query_input.get("query", ""),
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
        }

    def route_batch(self, batch: list) -> list:
        """Route multiple queries."""
        return [self.route_single(item) for item in batch]
```

### 3. Create Configuration File

Create `config.yaml`:

```yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'

hparam:
  # Your hyperparameters here
  param1: value1

api_endpoint: 'https://integrate.api.nvidia.com/v1'
```

### 4. Use Your Custom Router

```bash
# Inference
llmrouter infer --router my_router \
  --config custom_routers/my_router/config.yaml \
  --query "What is machine learning?"

# List all routers (including custom ones)
llmrouter list-routers
```

## Directory Structure

The plugin system automatically discovers routers in these locations:

1. `./custom_routers/` (current directory)
2. `~/.llmrouter/plugins/` (user home directory)
3. Paths in `$LLMROUTER_PLUGINS` environment variable (colon-separated)

### Expected Structure

```
custom_routers/
├── __init__.py          # (Optional) Package marker
└── my_router/
    ├── __init__.py      # (Optional) Export router class
    ├── router.py        # Router implementation (required)
    ├── trainer.py       # (Optional) Trainer for training support
    └── config.yaml      # (Optional) Example configuration
```

## Router Interface Requirements

Your router **must** implement:

### Required Methods

```python
def route_single(self, query_input: dict) -> dict:
    """
    Route a single query.

    Args:
        query_input: dict with at least 'query' key

    Returns:
        dict with at least 'model_name' or 'predicted_llm' key
    """
    pass

def route_batch(self, batch: list) -> list:
    """
    Route multiple queries.

    Args:
        batch: list of query_input dicts

    Returns:
        list of routing result dicts
    """
    pass
```

### Required Inheritance

```python
from llmrouter.models.meta_router import MetaRouter

class MyRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        super().__init__(model=your_model, yaml_path=yaml_path)
```

## Adding Training Support

To support training, create `trainer.py`:

```python
from llmrouter.models.base_trainer import BaseTrainer

class MyRouterTrainer(BaseTrainer):
    def __init__(self, router, config: dict, device: str = "cpu"):
        super().__init__(router, config, device)

        # Setup optimizer, hyperparameters, etc.
        self.optimizer = ...
        self.num_epochs = config['hparam']['train_epoch']

    def train(self) -> None:
        """Training logic."""
        for epoch in range(self.num_epochs):
            # Your training loop
            pass

        # Save model
        save_path = self.config['model_path']['save_model_path']
        self.router.save_router(save_path)
```

Then use:

```bash
llmrouter train --router my_router \
  --config custom_routers/my_router/config.yaml
```

## Examples

See the included examples:

### 1. RandomRouter (Simple Baseline)

Location: `custom_routers/randomrouter/`

A minimal router that randomly selects an LLM. Great starting point for learning.

```bash
# Test it
llmrouter infer --router randomrouter \
  --config custom_routers/randomrouter/config.yaml \
  --query "Hello world" \
  --route-only
```

### 2. ThresholdRouter (Advanced)

Location: `custom_routers/thresholdrouter/`

A trainable router that uses difficulty estimation:
- Easy queries → smaller/cheaper models
- Hard queries → larger/more capable models

Features:
- Neural network-based difficulty estimator
- Full training pipeline
- Configurable threshold

## Configuration Details

### Required Fields

```yaml
data_path:
  llm_data: 'path/to/llm_candidates.json'  # Required

hparam:
  # Your custom hyperparameters
```

### Optional Fields

```yaml
data_path:
  query_data_train: 'path/to/train_queries.jsonl'
  query_data_test: 'path/to/test_queries.jsonl'
  routing_data_train: 'path/to/train_routing.jsonl'
  routing_data_test: 'path/to/test_routing.jsonl'

model_path:
  save_model_path: 'saved_models/my_router/model.pt'
  load_model_path: 'saved_models/my_router/model.pt'

metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0

api_endpoint: 'https://integrate.api.nvidia.com/v1'
```

## Data Format

### LLM Candidates (`llm_data`)

```json
{
  "model-name-1": {
    "model": "api/model-path",
    "size": "7B",
    "cost": 0.001
  },
  "model-name-2": {
    "model": "api/model-path-2",
    "size": "70B",
    "cost": 0.01
  }
}
```

### Query Data

JSONL format with one query per line:

```jsonl
{"query": "What is AI?", "id": "q1"}
{"query": "Explain quantum computing", "id": "q2"}
```

### Routing Data (for training)

```jsonl
{"query": "What is AI?", "best_llm": "model-name-1", "performance": 0.95}
{"query": "Explain quantum computing", "best_llm": "model-name-2", "performance": 0.88}
```

## Debugging

Enable verbose plugin discovery:

```python
from llmrouter.plugin_system import discover_and_register_plugins

registry = discover_and_register_plugins(
    plugin_dirs=['custom_routers'],
    verbose=True
)

print(f"Discovered routers: {registry.get_router_names()}")
```

## Common Issues

### Router Not Found

**Error:** `Unknown router: my_router`

**Solutions:**
1. Check directory name matches router name (lowercase)
2. Ensure router class ends with `Router` (e.g., `MyRouter`)
3. Verify `custom_routers/` directory exists
4. Enable verbose mode to see discovery logs

### Import Errors

**Error:** `ModuleNotFoundError`

**Solutions:**
1. Ensure all dependencies are installed
2. Check `__init__.py` files exist in directories
3. Verify class names and imports

### Validation Failed

**Error:** `Router class validation failed`

**Solutions:**
1. Implement required methods: `route_single` and `route_batch`
2. Inherit from `MetaRouter`
3. Ensure methods have correct signatures

## Advanced Features

### Custom Embedding Generation

```python
from llmrouter.utils import get_longformer_embedding

class MyRouter(MetaRouter):
    def route_single(self, query_input: dict) -> dict:
        query = query_input['query']

        # Generate embedding on-the-fly
        embedding = get_longformer_embedding(query)

        # Use embedding for routing
        selected_llm = self._route_by_embedding(embedding)

        return {
            "query": query,
            "model_name": selected_llm,
        }
```

### Multi-round Routing

```python
class MyMultiRoundRouter(MetaRouter):
    def answer_query(self, query: str, return_intermediate: bool = False):
        """Handle complex queries with multiple routing decisions."""
        # Decompose query
        sub_queries = self._decompose(query)

        # Route each sub-query
        results = []
        for sq in sub_queries:
            routing = self.route_single({'query': sq})
            # Call API, aggregate results, etc.
            results.append(routing)

        return self._aggregate(results)
```

### Sharing Between Routers

```python
# In custom_routers/shared_utils.py
def shared_preprocessing(query):
    # Shared utility functions
    return processed_query
```

## Best Practices

1. **Keep it Simple**: Start with a minimal implementation
2. **Test Incrementally**: Test `route_single` before `route_batch`
3. **Use Existing Data**: Start with provided example data
4. **Document Config**: Add comments to your `config.yaml`
5. **Handle Errors**: Add try-except for robustness
6. **Version Control**: Keep your custom routers in git

## Performance Tips

1. **Batch Processing**: Implement efficient `route_batch` for throughput
2. **Caching**: Cache embeddings or model predictions
3. **GPU Support**: Move models to GPU in `__init__`
4. **Lazy Loading**: Load heavy resources only when needed

## Contributing

To share your custom router with the community:

1. Create a clean implementation with documentation
2. Add example configuration and usage
3. Submit a pull request to add it to the examples
4. Consider publishing as a separate package

## Support

- GitHub Issues: https://github.com/ulab-uiuc/LLMRouter/issues
- Slack: [Join our community]
- Examples: See `custom_routers/` directory
