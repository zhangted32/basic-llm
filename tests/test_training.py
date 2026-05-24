import unittest
import json
import tempfile
from pathlib import Path
from src.training.dpo_config import DPOConfig, DPOConfigManager
from src.training.dpo_trainer import DPOTrainer, TrainingMetrics
from src.training.lora import LoRAConfig, LoRAModelWrapper


class TestDPOConfig(unittest.TestCase):
    
    def test_default_config(self):
        """Test default DPO configuration."""
        config = DPOConfig()
        
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation, 4)
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.training_steps, 1000)
        self.assertEqual(config.lora_rank, 8)

    def test_custom_config(self):
        """Test custom DPO configuration."""
        config = DPOConfig(
            batch_size=4,
            training_steps=500,
            lora_rank=16
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.training_steps, 500)
        self.assertEqual(config.lora_rank, 16)

    def test_config_manager_predefined(self):
        """Test getting predefined configurations."""
        quick_config = DPOConfigManager.get_config("quick_test")
        self.assertEqual(quick_config.training_steps, 100)
        
        standard_config = DPOConfigManager.get_config("standard")
        self.assertEqual(standard_config.training_steps, 1000)

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = DPOConfig(batch_size=2, training_steps=200)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            DPOConfigManager.save_config(config, temp_path)
            loaded_config = DPOConfigManager.load_config(temp_path)
            
            self.assertEqual(loaded_config.batch_size, config.batch_size)
            self.assertEqual(loaded_config.training_steps, config.training_steps)
        finally:
            temp_path.unlink()


class TestDPOTrainer(unittest.TestCase):
    
    def setUp(self):
        self.trainer = DPOTrainer(DPOConfig(
            batch_size=1,
            training_steps=1000,  # Use sufficient steps for warmup
            checkpoint_interval=500,
            warmup_steps=100,
            output_dir="output/test_training"
        ))

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.current_step, 0)
        self.assertEqual(self.trainer.config.batch_size, 1)
        self.assertEqual(len(self.trainer.metrics_history), 0)

    def test_dpo_loss_computation(self):
        """Test DPO loss formula."""
        # Test case 1: chosen reward > rejected reward -> loss < 1
        loss1 = self.trainer._compute_dpo_loss(
            "prompt",
            "good response",
            "bad response"
        )
        self.assertLess(loss1, 1.0)
        
        # Test case 2: should be positive
        self.assertGreater(loss1, 0.0)

    def test_learning_rate_warmup(self):
        """Test learning rate warmup."""
        # At warmup start
        lr_0 = self.trainer._get_learning_rate(0)
        self.assertEqual(lr_0, 0.0)
        
        # Mid warmup
        lr_50 = self.trainer._get_learning_rate(50)
        self.assertGreater(lr_50, 0.0)
        self.assertLess(lr_50, self.trainer.config.learning_rate)
        
        # Immediately after warmup (step 100)
        lr_100 = self.trainer._get_learning_rate(100)
        self.assertEqual(lr_100, self.trainer.config.learning_rate)  # Full LR at end of warmup

    def test_learning_rate_decay(self):
        """Test learning rate decay after warmup."""
        # At start of decay (step 100)
        lr_100 = self.trainer._get_learning_rate(100)
        self.assertEqual(lr_100, self.trainer.config.learning_rate)
        
        # At end of training, should be at least 10% of original
        final_lr = self.trainer._get_learning_rate(self.trainer.config.training_steps - 1)
        self.assertGreaterEqual(final_lr, self.trainer.config.learning_rate * 0.1)
        self.assertLess(final_lr, self.trainer.config.learning_rate)

    def test_training_step_metrics(self):
        """Test that training step returns valid metrics."""
        batch = [{
            "prompt": "test prompt",
            "chosen": "good response",
            "rejected": "bad response"
        }]
        
        metrics = self.trainer._training_step(batch, 100)  # Use step > warmup
        
        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertEqual(metrics.step, 100)
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreater(metrics.learning_rate, 0.0)  # After warmup
        self.assertGreater(metrics.train_time_ms, 0.0)

    def test_checkpoint_creation(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.output_dir = Path(tmpdir)
            self.trainer._save_checkpoint(100)
            
            checkpoint_dir = self.trainer.output_dir / "checkpoint-000100"
            self.assertTrue(checkpoint_dir.exists())
            
            metrics_file = checkpoint_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            self.assertEqual(metrics_data["step"], 100)

    def test_metrics_history(self):
        """Test that metrics are recorded."""
        batch = [{
            "prompt": "test",
            "chosen": "good",
            "rejected": "bad"
        }]
        
        # _training_step doesn't update history - that's done by train()
        # But we can manually track as train() does
        metrics1 = self.trainer._training_step(batch, 0)
        self.trainer.metrics_history.append(metrics1)
        metrics2 = self.trainer._training_step(batch, 1)
        self.trainer.metrics_history.append(metrics2)
        
        self.assertEqual(len(self.trainer.metrics_history), 2)
        self.assertEqual(self.trainer.metrics_history[0].step, 0)
        self.assertEqual(self.trainer.metrics_history[1].step, 1)

    def test_reward_computation(self):
        """Test reward computation is consistent."""
        reward1 = self.trainer._compute_reward("prompt", "response1")
        reward2 = self.trainer._compute_reward("prompt", "response1")
        
        # Same input should give same reward
        self.assertEqual(reward1, reward2)
        
        # Reward should be in reasonable range
        self.assertGreaterEqual(reward1, 0.0)
        self.assertLessEqual(reward1, 1.0)


class TestLoRAConfig(unittest.TestCase):
    
    def test_lora_config_default(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        
        self.assertEqual(config.rank, 8)
        self.assertEqual(config.alpha, 16)
        self.assertEqual(config.dropout, 0.05)
        self.assertIn("q_proj", config.target_modules)

    def test_lora_config_custom(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(rank=16, alpha=32, dropout=0.1)
        
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.alpha, 32)
        self.assertEqual(config.dropout, 0.1)

    def test_lora_config_to_dict(self):
        """Test converting config to dictionary."""
        config = LoRAConfig(rank=4)
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["rank"], 4)
        self.assertEqual(config_dict["alpha"], 16)


class TestLoRAModelWrapper(unittest.TestCase):
    
    def test_wrapper_initialization(self):
        """Test LoRA model wrapper initialization."""
        config = LoRAConfig(rank=8)
        wrapper = LoRAModelWrapper(None, config)
        
        self.assertEqual(wrapper.config.rank, 8)
        self.assertEqual(len(wrapper.lora_layers), 0)

    def test_get_trainable_parameters(self):
        """Test getting only trainable parameters."""
        config = LoRAConfig(rank=4)
        wrapper = LoRAModelWrapper(None, config)
        
        params = wrapper.get_trainable_parameters()
        self.assertEqual(len(params), 0)  # No layers added yet


class TestTrainingMetrics(unittest.TestCase):
    
    def test_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            step=100,
            loss=0.5,
            learning_rate=1e-4,
            grad_norm=1.0,
            train_time_ms=50.0,
            examples_per_second=20.0
        )
        
        self.assertEqual(metrics.step, 100)
        self.assertEqual(metrics.loss, 0.5)
        self.assertEqual(metrics.learning_rate, 1e-4)


if __name__ == "__main__":
    unittest.main()