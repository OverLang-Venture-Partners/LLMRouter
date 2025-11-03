# models/graphrouter/trainer.py

import torch
from llmrouter.models.base_trainer import BaseTrainer


class GraphRouterTrainer(BaseTrainer):
    """
    GraphRouterTrainer
    ------------------
    Example trainer implementation for GraphRouter.

    Uses a simple supervised learning objective with cross-entropy loss.
    """

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute cross-entropy loss between predicted logits and labels.

        Args:
            outputs (dict):
                Dictionary containing "logits" from the router.
            batch (dict):
                Dictionary containing ground truth labels under key "labels".

        Returns:
            torch.Tensor:
                Scalar loss used for optimization.
        """
        logits = outputs["logits"]
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def train(self, dataloader):
        """
        Train the GraphRouter for one or multiple epochs.

        Args:
            dataloader:
                Iterable of batches for training.
        """
        self.router.train()
        for step, batch in enumerate(dataloader):
            batch = self._move_batch_to_device(batch)

            outputs = self.router(batch)
            loss = self.loss_func(outputs, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(f"[GraphRouterTrainer] Step {step} | loss = {loss.item():.4f}")
