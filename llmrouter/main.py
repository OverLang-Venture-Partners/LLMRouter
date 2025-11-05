from some_module import MLP  # user-defined PyTorch model

from llmrouter.models.graphrouter.router import GraphRouter
from llmrouter.models.graphrouter.trainer import GraphRouterTrainer

# 1. User defines the underlying model using pure PyTorch
model = MLP(input_dim=128, output_dim=10)

# 2. Wrap model into a router
router = GraphRouter(model=model, yaml_path="configs/graph_router.yaml")

# 3. Instantiate trainer with this router
trainer = GraphRouterTrainer(router=router, device="cuda")

# 4. Train using any PyTorch-style dataloader
trainer.train(train_dataloader)
