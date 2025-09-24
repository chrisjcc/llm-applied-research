
"""
Group Relative Policy Optimization (GRPO) - Improved Version

This module provides functionality for Group Relative Policy Optimization training
with improved error handling, logging, code organization, and evaluation capabilities.
"""

import gc
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter specific warnings only
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configuration constants
class Config:
    """Configuration constants for the GRPO pipeline"""
    DEFAULT_MAX_NEW_TOKENS = 300
    DEFAULT_LEARNING_RATE = 5e-6
    DEFAULT_EPOCHS = 1
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
    DEFAULT_NUM_GENERATIONS = 4
    DEFAULT_LOGGING_STEPS = 2
    DATASET_PREVIEW_ROWS = 5
    SMALL_DATASET_SIZE = 10
    EVAL_DATASET_SIZE = 5
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant that solves problems step-by-step. "
        "Always include the final numeric answer inside \\boxed{}."
    )


class ModelManager:
    """Manages model and tokenizer loading, testing, and cleanup"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    @contextmanager
    def load_model_context(self, model_path: Union[str, Path]):
        """Context manager for safe model loading and cleanup"""
        model, tokenizer = None, None
        try:
            model, tokenizer = self._load_model_and_tokenizer(model_path)
            yield model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
        finally:
            self._cleanup_model(model, tokenizer)
    
    def _load_model_and_tokenizer(self, model_name: Union[str, Path]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with proper configuration"""
        try:
            logger.info(f"Loading model and tokenizer from {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move model to appropriate device
            model.to(self.device)
            
            # Configure chat template if not present
            self._configure_chat_template(tokenizer)
            
            # Configure padding token
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _configure_chat_template(self, tokenizer: PreTrainedTokenizer) -> None:
        """Configure chat template if not present"""
        if not tokenizer.chat_template:
            tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
            logger.info("Configured default chat template")
    
    def _cleanup_model(self, model: Optional[PreTrainedModel], tokenizer: Optional[PreTrainedTokenizer]) -> None:
        """Clean up model and tokenizer resources"""
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            self._cleanup_memory()
            logger.info("Model cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _cleanup_memory(self) -> None:
        """Clean up memory and CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ResponseGenerator:
    """Handles response generation from models"""
    
    @staticmethod
    def generate_response(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        max_new_tokens: int = Config.DEFAULT_MAX_NEW_TOKENS,
        full_message: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate response from model for given input"""
        try:
            # Format messages
            if full_message:
                messages = full_message
            else:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                if user_message:
                    messages.append({"role": "user", "content": user_message})
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Extract and decode response
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


class RewardFunction:
    """Handles reward calculation for GRPO training"""
    
    @staticmethod
    def math_reward_function(completions: List[List[Dict[str, str]]], ground_truth: List[str], **kwargs) -> List[float]:
        """
        Reward function for math problems that extracts answers from \\boxed{} format
        
        Args:
            completions: List of completion messages
            ground_truth: List of ground truth answers
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect)
        """
        try:
            # Extract content from completions
            matches = []
            for completion in completions:
                try:
                    content = completion[0]['content'] if completion and len(completion) > 0 else ""
                    match = re.search(r"\\boxed\{(.*?)\}", content)
                    matches.append(match)
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Error extracting content from completion: {e}")
                    matches.append(None)
            
            # Extract boxed content
            contents = [match.group(1) if match else "" for match in matches]
            
            # Calculate rewards
            rewards = [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
            
            return rewards
            
        except Exception as e:
            logger.error(f"Error in reward function: {e}")
            return [0.0] * len(completions)
    
    @staticmethod
    def test_reward_function():
        """Test the reward function with sample data"""
        logger.info("Testing reward function")
        
        # Test positive case
        sample_pred = [[{"role": "assistant", "content": r"...Calculating the answer. \boxed{72}"}]]
        ground_truth = ["72"]
        reward = RewardFunction.math_reward_function(sample_pred, ground_truth)
        print(f"Positive Sample Reward: {reward}")
        
        # Test negative case
        sample_pred = [[{"role": "assistant", "content": r"...Calculating the answer \boxed{71}"}]]
        ground_truth = ["72"]
        reward = RewardFunction.math_reward_function(sample_pred, ground_truth)
        print(f"Negative Sample Reward: {reward}")


class DatasetManager:
    """Manages dataset loading, processing, and display"""
    
    @staticmethod
    def load_gsm8k_dataset(split: str = "test", limit_size: Optional[int] = None) -> Dataset:
        """Load GSM8K dataset with optional size limit"""
        try:
            logger.info(f"Loading GSM8K dataset (split: {split})")
            dataset = load_dataset("openai/gsm8k", "main")[split]
            
            if limit_size:
                dataset = dataset.select(range(min(limit_size, len(dataset))))
                logger.info(f"Limited dataset to {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            raise
    
    @staticmethod
    def post_process_gsm8k_example(example: Dict[str, Any], system_prompt: str = Config.DEFAULT_SYSTEM_PROMPT) -> Dict[str, Any]:
        """Post-process GSM8K example to extract ground truth and format prompt"""
        try:
            # Extract ground truth answer
            match = re.search(r"####\s*(-?\d+)", example["answer"])
            example["ground_truth"] = match.group(1) if match else None
            
            # Format prompt
            example["prompt"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]}
            ]
            
            return example
            
        except Exception as e:
            logger.error(f"Error post-processing example: {e}")
            return example
    
    @staticmethod
    def process_gsm8k_dataset(dataset: Dataset, system_prompt: str = Config.DEFAULT_SYSTEM_PROMPT) -> Dataset:
        """Process GSM8K dataset for training/evaluation"""
        try:
            logger.info("Processing GSM8K dataset")
            
            # Apply post-processing
            processed_dataset = dataset.map(
                lambda example: DatasetManager.post_process_gsm8k_example(example, system_prompt)
            )
            
            # Remove unnecessary columns
            processed_dataset = processed_dataset.remove_columns(["question", "answer"])
            
            logger.info(f"Processed {len(processed_dataset)} examples")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Error processing GSM8K dataset: {e}")
            raise
    
    @staticmethod
    def display_dataset_preview(dataset: Dataset, num_rows: int = Config.DATASET_PREVIEW_ROWS) -> None:
        """Display a preview of the dataset"""
        try:
            logger.info("Displaying dataset preview")
            
            sample_df = dataset.select(range(min(num_rows, len(dataset)))).to_pandas()
            DatasetManager._configure_pandas_display()
            print("\n=== Dataset Preview ===")
            print(sample_df.to_string(index=False))
            print()
            
        except Exception as e:
            logger.error(f"Error displaying dataset preview: {e}")
    
    @staticmethod
    def _configure_pandas_display() -> None:
        """Configure pandas display options for better readability"""
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 0)


class ModelEvaluator:
    """Handles model evaluation on datasets"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 reward_function: Callable = RewardFunction.math_reward_function):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
    
    def evaluate_on_dataset(self, eval_dataset: Dataset, verbose: bool = True) -> Dict[str, float]:
        """Evaluate model on dataset and return metrics"""
        try:
            logger.info(f"Evaluating model on dataset with {len(eval_dataset)} examples")
            
            all_predictions = []
            all_labels = []
            
            # Generate predictions
            for example in tqdm(eval_dataset, desc="Evaluating"):
                try:
                    input_prompt = example["prompt"]
                    ground_truth = example["ground_truth"]
                    
                    # Generate response
                    with torch.no_grad():
                        response = ResponseGenerator.generate_response(
                            self.model, self.tokenizer, full_message=input_prompt
                        )
                    
                    all_predictions.append([{"role": "assistant", "content": response}])
                    all_labels.append(ground_truth)
                    
                    if verbose:
                        print(f"Response: {response}")
                        print(f"Ground truth: {ground_truth}")
                        print("-" * 50)
                    
                except Exception as e:
                    logger.error(f"Error evaluating example: {e}")
                    all_predictions.append([{"role": "assistant", "content": "Error"}])
                    all_labels.append(example.get("ground_truth", ""))
            
            # Calculate rewards and metrics
            rewards = self.reward_function(all_predictions, all_labels)
            accuracy = sum(rewards) / len(rewards) if rewards else 0.0
            
            metrics = {
                "accuracy": accuracy,
                "total_examples": len(eval_dataset),
                "correct_predictions": sum(rewards)
            }
            
            logger.info(f"Evaluation completed. Accuracy: {accuracy:.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"accuracy": 0.0, "total_examples": 0, "correct_predictions": 0}


class GRPOManager:
    """Manages the Group Relative Policy Optimization training process"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    def create_grpo_config(
        self,
        learning_rate: float = Config.DEFAULT_LEARNING_RATE,
        num_epochs: int = Config.DEFAULT_EPOCHS,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps: int = Config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        num_generations: int = Config.DEFAULT_NUM_GENERATIONS,
        logging_steps: int = Config.DEFAULT_LOGGING_STEPS
    ) -> GRPOConfig:
        """Create GRPO configuration with appropriate settings"""
        return GRPOConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,  # Can be set as high as 64 or 128
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            no_cuda=not self.use_gpu  # Keeps the whole run on CPU/MPS
        )
    
    def train_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        reward_function: Callable,
        config: Optional[GRPOConfig] = None
    ) -> GRPOTrainer:
        """Train model using Group Relative Policy Optimization"""
        try:
            if config is None:
                config = self.create_grpo_config()
            
            logger.info("Starting Group Relative Policy Optimization training")
            logger.info(f"Training on {len(dataset)} examples")
            
            trainer = GRPOTrainer(
                model=model,
                args=config,
                reward_funcs=reward_function,
                train_dataset=dataset
            )
            
            trainer.train()
            logger.info("GRPO training completed successfully")
            
            return trainer
            
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise


class ModelTester:
    """Handles model testing with questions"""
    
    @staticmethod
    def test_model_with_questions(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        questions: List[str],
        system_message: Optional[str] = None,
        title: str = "Model Output"
    ) -> None:
        """Test model with a list of questions"""
        logger.info(f"Testing model: {title}")
        print(f"\n=== {title} ===")
        
        for i, question in enumerate(questions, 1):
            try:
                response = ResponseGenerator.generate_response(
                    model, tokenizer, question, system_message
                )
                print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
            except Exception as e:
                logger.error(f"Error testing question {i}: {e}")
                print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\nError: {e}\n")


def main():
    """Main execution function"""
    try:
        # Configuration
        use_gpu = False
        system_prompt = Config.DEFAULT_SYSTEM_PROMPT
        
        # Model paths
        instruct_model_path = "./models/Qwen/Qwen2.5-0.5B-Instruct"
        grpo_model_path = "./models/banghua/Qwen2.5-0.5B-GRPO"
        small_model_path = "./models/HuggingFaceTB/SmolLM2-135M-Instruct"
        
        # Initialize managers
        model_manager = ModelManager(use_gpu=use_gpu)
        dataset_manager = DatasetManager()
        grpo_manager = GRPOManager(use_gpu=use_gpu)
        
        # Test reward function
        RewardFunction.test_reward_function()
        
        # Load and process evaluation dataset
        logger.info("Loading evaluation dataset")
        eval_dataset_raw = dataset_manager.load_gsm8k_dataset(
            split="test", 
            limit_size=Config.EVAL_DATASET_SIZE
        )
        dataset_manager.display_dataset_preview(eval_dataset_raw)
        
        eval_dataset = dataset_manager.process_gsm8k_dataset(eval_dataset_raw, system_prompt)
        dataset_manager.display_dataset_preview(eval_dataset)
        
        # Evaluate base instruct model
        logger.info("Evaluating base instruct model")
        with model_manager.load_model_context(instruct_model_path) as (model, tokenizer):
            evaluator = ModelEvaluator(model, tokenizer, RewardFunction.math_reward_function)
            base_metrics = evaluator.evaluate_on_dataset(eval_dataset)
            print(f"Base Model Evaluation Accuracy: {base_metrics['accuracy']:.2%}")
        
        # Load and process training dataset
        logger.info("Loading training dataset")
        train_dataset_raw = dataset_manager.load_gsm8k_dataset(
            split="train",
            limit_size=Config.SMALL_DATASET_SIZE if not use_gpu else None
        )
        
        train_dataset = dataset_manager.process_gsm8k_dataset(train_dataset_raw, system_prompt)
        print(f"Training dataset sample: {train_dataset[0]}")
        
        # GRPO Training
        logger.info("Starting GRPO training")
        with model_manager.load_model_context(small_model_path) as (model, tokenizer):
            trainer = grpo_manager.train_model(
                model, tokenizer, train_dataset, RewardFunction.math_reward_function
            )
            
            # Evaluate trained model vs. fully trained model
            use_fully_trained = True
            if use_fully_trained:
                logger.info("Evaluating fully trained GRPO model")
                with model_manager.load_model_context(grpo_model_path) as (full_model, full_tokenizer):
                    evaluator = ModelEvaluator(
                        full_model, full_tokenizer, RewardFunction.math_reward_function
                    )
                    trained_metrics = evaluator.evaluate_on_dataset(eval_dataset)
                    print(f"Fully Trained GRPO Model Evaluation Accuracy: {trained_metrics['accuracy']:.2%}")
            else:
                logger.info("Evaluating small model after GRPO training")
                evaluator = ModelEvaluator(
                    trainer.model, tokenizer, RewardFunction.math_reward_function
                )
                trained_metrics = evaluator.evaluate_on_dataset(eval_dataset)
                print(f"Small Model (After GRPO) Evaluation Accuracy: {trained_metrics['accuracy']:.2%}")
        
        # Summary
        logger.info("GRPO training pipeline completed successfully")
        print("\n=== Training Summary ===")
        print(f"Base Model Accuracy: {base_metrics['accuracy']:.2%}")
        if 'trained_metrics' in locals():
            print(f"Trained Model Accuracy: {trained_metrics['accuracy']:.2%}")
            improvement = trained_metrics['accuracy'] - base_metrics['accuracy']
            print(f"Improvement: {improvement:.2%}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
