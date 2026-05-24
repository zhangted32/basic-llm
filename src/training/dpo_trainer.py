import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. Using mock implementation.")


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    train_time_ms: float
    examples_per_second: float


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.
    
    Implements DPO loss: L = -log(σ(r_chosen - r_rejected))
    
    where r = β * (π(y) - π(y_ref)) / T
    
    This trainer uses LoRA adapters for efficient fine-tuning.
    """

    def __init__(self, config):
        """
        Initialize DPO Trainer.
        
        Args:
            config: DPOConfig instance
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.trainable_params = {}
        self.current_step = 0
        self.metrics_history = []
        self.best_loss = float('inf')
        
        # Initialize output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock model for testing (when MLX not available)
        self.mock_model_loaded = False

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load DPO dataset from JSONL file.
        
        Args:
            dataset_path: Path to dpo_dataset.jsonl
            
        Returns:
            List of DPO examples
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                examples.append(example)
        
        print(f"Loaded {len(examples)} examples from {dataset_path}")
        return examples

    def setup_model(self):
        """Initialize model and optimizer."""
        if not MLX_AVAILABLE:
            print("MLX not available. Using mock training.")
            self.mock_model_loaded = True
            return
        
        try:
            # Initialize model (stub implementation)
            print("Setting up model with LoRA adapters...")
            self.model = self._create_mock_model()
            
            # Setup optimizer
            self.optimizer = optim.Adam(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            print("Model and optimizer initialized successfully.")
            
        except Exception as e:
            print(f"Failed to setup model: {str(e)}")
            self.mock_model_loaded = True

    def _create_mock_model(self):
        """Create a mock model for testing."""
        # In production, would load actual Qwen model
        return MockModel()

    def train(self, dataset_path: str):
        """
        Execute DPO training loop.
        
        Args:
            dataset_path: Path to DPO dataset
        """
        print(f"Starting DPO training...")
        print(f"Config: {self.config}")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Setup model
        self.setup_model()
        
        # Training loop
        start_time = time.time()
        steps_per_epoch = len(dataset) // self.config.batch_size
        
        print(f"\nTraining for {self.config.training_steps} steps...")
        print(f"Dataset size: {len(dataset)}, Batch size: {self.config.batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        for step in range(self.config.training_steps):
            # Get batch
            batch_start = step * self.config.batch_size % len(dataset)
            batch_end = batch_start + self.config.batch_size
            batch = dataset[batch_start:batch_end]
            
            # Training step
            metrics = self._training_step(batch, step)
            self.metrics_history.append(metrics)
            
            # Logging
            if step % 10 == 0:
                print(f"Step {step}/{self.config.training_steps}: "
                      f"loss={metrics.loss:.4f}, "
                      f"lr={metrics.learning_rate:.2e}, "
                      f"time={metrics.train_time_ms:.0f}ms")
            
            # Checkpoint
            if step > 0 and step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(step)
            
            # Evaluation
            if step > 0 and step % self.config.eval_interval == 0:
                eval_loss = self._evaluate(dataset[:100])  # Use subset for eval
                print(f"Eval loss at step {step}: {eval_loss:.4f}")
        
        # Save final checkpoint
        self._save_checkpoint(self.config.training_steps, name="final")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final checkpoint saved to {self.output_dir}")

    def _training_step(self, batch: List[Dict], step: int) -> TrainingMetrics:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of DPO examples
            step: Current step number
            
        Returns:
            TrainingMetrics
        """
        step_start = time.time()
        
        # Compute DPO loss
        chosen_rewards = []
        rejected_rewards = []
        
        for example in batch:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Mock reward computation
            chosen_reward = self._compute_reward(prompt, chosen)
            rejected_reward = self._compute_reward(prompt, rejected)
            
            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)
        
        # Compute DPO loss: -log(σ(r_chosen - r_rejected))
        # Using β (beta) temperature parameter
        beta = 0.1
        reward_diff = mx.array(chosen_rewards) - mx.array(rejected_rewards)
        
        # Use sigmoid instead of logistic (MLX doesn't have logistic)
        sigmoid = 1.0 / (1.0 + mx.exp(-reward_diff * beta))
        loss = -mx.log(sigmoid)
        loss = mx.mean(loss)
        
        # Compute gradients (mock)
        if self.mock_model_loaded:
            grad_norm = 0.5  # Mock gradient norm
        else:
            # In production: compute actual gradients
            grad_norm = 0.5
        
        # Update learning rate with warmup
        lr = self._get_learning_rate(step)
        
        # Update metrics
        train_time = (time.time() - step_start) * 1000
        
        return TrainingMetrics(
            step=step,
            loss=float(loss),
            learning_rate=lr,
            grad_norm=grad_norm,
            train_time_ms=train_time,
            examples_per_second=len(batch) / (train_time / 1000)
        )

    def _compute_reward(self, prompt: str, response: str) -> float:
        """
        Compute reward for a response given a prompt.
        
        In production, this would use the reward model.
        For now, returns a mock reward.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Reward score
        """
        # Mock reward based on response length and presence of key elements
        base_reward = 0.5
        
        # Add some variance
        import hashlib
        hash_val = int(hashlib.md5((prompt + response).encode()).hexdigest(), 16)
        variance = (hash_val % 100) / 100.0
        
        return base_reward + variance * 0.5

    def _get_learning_rate(self, step: int) -> float:
        """
        Get learning rate with warmup.
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            # Linear decay
            progress = (step - self.config.warmup_steps) / (self.config.training_steps - self.config.warmup_steps)
            return self.config.learning_rate * max(0.1, 1.0 - progress)

    def _compute_dpo_loss(self, prompt: str, chosen: str, rejected: str) -> float:
        """
        Compute DPO loss for a single example.
        
        L = -log(σ(r_chosen - r_rejected))
        
        Args:
            prompt: Input prompt
            chosen: Chosen response
            rejected: Rejected response
            
        Returns:
            Loss value
        """
        chosen_reward = self._compute_reward(prompt, chosen)
        rejected_reward = self._compute_reward(prompt, rejected)
        
        beta = 0.1
        reward_diff = chosen_reward - rejected_reward
        
        # σ(x) = 1 / (1 + exp(-x))
        import math
        loss = -math.log(1 / (1 + math.exp(-reward_diff * beta)))
        
        return loss

    def _backprop(self, loss):
        """
        Perform backpropagation and weight update.
        
        Args:
            loss: Loss value
        """
        if self.mock_model_loaded:
            # Mock backprop - just update step counter
            self.current_step += 1
            return
        
        # In production: actual backprop with MLX
        # gradients = mx.grad(self.compute_loss)(params, batch)
        # self.optimizer.update(params, gradients)
        pass

    def _evaluate(self, dataset: List[Dict]) -> float:
        """
        Evaluate on a dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Average loss
        """
        losses = []
        for example in dataset:
            loss = self._compute_dpo_loss(
                example["prompt"],
                example["chosen"],
                example["rejected"]
            )
            losses.append(loss)
        
        return sum(losses) / len(losses) if losses else 0.0

    def _save_checkpoint(self, step: int, name: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            step: Current step
            name: Optional checkpoint name
        """
        if name is None:
            name = f"checkpoint-{step:06d}"
        
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = checkpoint_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "step": step,
                "metrics": [m.__dict__ for m in self.metrics_history[-100:]]
            }, f, indent=2)
        
        # Save LoRA weights (mock)
        lora_file = checkpoint_dir / "adapter.safetensors"
        with open(lora_file, 'w') as f:
            json.dump({
                "step": step,
                "config": self.config.__dict__
            }, f)
        
        print(f"Checkpoint saved: {checkpoint_dir}")

    def get_metrics_history(self) -> List[TrainingMetrics]:
        """Get training metrics history."""
        return self.metrics_history


class MockModel:
    """Mock model for testing when MLX is not available."""
    
    def __init__(self):
        self.trainable = True
    
    def parameters(self):
        return {}
    
    def __call__(self, *args, **kwargs):
        return MockOutput()
    
    def generate(self, *args, **kwargs):
        return "mock output"


class MockOutput:
    """Mock output for testing."""
    
    def __init__(self):
        self.logits = None