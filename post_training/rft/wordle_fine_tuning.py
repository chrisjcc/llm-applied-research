#!/usr/bin/env python
# coding: utf-8

"""
Wordle AI Training Script using TRL GRPO (Group Relative Policy Optimization) and SFT

This script demonstrates how to train an AI model to play Wordle using TRL's native
GRPO and SFT trainers. This provides the same functionality as Predibase's GRPO
but runs completely free using Hugging Face's TRL library.

TRL GRPO Reference: https://huggingface.co/docs/trl/en/grpo_trainer
TRL SFT Reference: https://huggingface.co/docs/trl/en/sft_trainer

Requirements:
    pip install transformers trl datasets torch accelerate peft bitsandbytes wandb

Optional for better performance:
    pip install flash-attn --no-build-isolation

Environment Variables (optional):
    WANDB_API_KEY: For experiment tracking with Weights & Biases
    HF_TOKEN: Hugging Face token for private models/datasets
"""

import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import (
    GRPOConfig,
    GRPOTrainer,
    SFTConfig,
    SFTTrainer,
)
import trl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log TRL version for debugging
logger.info(f"TRL version: {trl.__version__}")


class WordleRewardCalculator:
    """
    Calculate rewards for Wordle game performance.
    
    This class implements the same reward functions as the original Predibase version
    but in a format compatible with TRL's GRPO trainer.
    """
    
    @staticmethod
    def guess_value(guess: str, target_word: str) -> float:
        """
        Calculate the value of a guess based on letter matches.
        
        Args:
            guess: The guessed word
            target_word: The target word to guess
            
        Returns:
            Reward score based on correct letters and positions
        """
        if len(guess) != 5 or len(target_word) != 5:
            return -1.0
            
        guess = guess.lower()
        target_word = target_word.lower()
        
        # Exact match gets highest reward
        if guess == target_word:
            return 10.0
            
        reward = 0.0
        target_chars = list(target_word)
        
        # Check for correct positions (green letters)
        for i in range(5):
            if guess[i] == target_word[i]:
                reward += 2.0
                target_chars[i] = None  # Mark as used
        
        # Check for correct letters in wrong positions (yellow letters)
        for i in range(5):
            if guess[i] != target_word[i] and guess[i] in target_chars:
                reward += 1.0
                # Remove one instance of this character
                target_chars[target_chars.index(guess[i])] = None
                
        return reward
    
    @staticmethod
    def output_format_check(output: str) -> float:
        """
        Check if the output follows the expected Wordle format.
        
        Args:
            output: Model output to check
            
        Returns:
            Reward for proper formatting (0.0 or 1.0)
        """
        # Extract guess from output using regex
        guess_pattern = r"Guess:\s*([A-Za-z]{5})"
        match = re.search(guess_pattern, output)
        
        if match and len(match.group(1)) == 5 and match.group(1).isalpha():
            return 1.0
        return 0.0
    
    @staticmethod
    def uses_previous_feedback(output: str, previous_feedback: str) -> float:
        """
        Check if the model incorporates previous feedback appropriately.
        
        Args:
            output: Current model output
            previous_feedback: Previous game feedback
            
        Returns:
            Reward for using feedback appropriately
        """
        if not previous_feedback or "No previous feedback" in previous_feedback:
            return 0.5  # Neutral for first guess
            
        # Extract guess from output
        guess_pattern = r"Guess:\s*([A-Za-z]{5})"
        match = re.search(guess_pattern, output)
        
        if not match:
            return 0.0
            
        guess = match.group(1).lower()
        
        # Check if guess avoids letters marked as not in word (gray)
        if "not in word" in previous_feedback.lower():
            gray_letters = re.findall(r"([a-z]) is not in word", previous_feedback.lower())
            for letter in gray_letters:
                if letter in guess:
                    return 0.0  # Penalty for reusing gray letters
                    
        return 1.0
    
    def calculate_combined_reward(
        self, 
        output: str, 
        target_word: str, 
        previous_feedback: str = ""
    ) -> float:
        """
        Calculate combined reward from all reward functions.
        
        Args:
            output: Model output
            target_word: Target word to guess
            previous_feedback: Previous feedback from the game
            
        Returns:
            Combined reward score
        """
        # Extract guess from output
        guess_pattern = r"Guess:\s*([A-Za-z]{5})"
        match = re.search(guess_pattern, output)
        
        if not match:
            return -2.0  # Strong penalty for invalid format
            
        guess = match.group(1)
        
        # Calculate individual rewards
        format_reward = self.output_format_check(output)
        feedback_reward = self.uses_previous_feedback(output, previous_feedback)
        guess_reward = self.guess_value(guess, target_word)
        
        # Combine rewards with weights
        total_reward = (
            format_reward * 2.0 +  # Format is important
            feedback_reward * 1.0 +  # Using feedback is good
            guess_reward * 3.0  # Actual guess quality is most important
        )
        
        return total_reward


class WordleTRLTrainer:
    """
    A trainer class for Wordle AI using TRL's GRPO and SFT trainers.
    
    This class handles both SFT and GRPO training for Wordle gameplay,
    providing a free alternative to Predibase's enterprise GRPO.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        use_quantization: bool = True,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize the Wordle trainer.
        
        Args:
            model_name: Name of the base model to use
            use_quantization: Whether to use 4-bit quantization for memory efficiency
            wandb_project: Weights & Biases project name for logging
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.reward_calculator = WordleRewardCalculator()
        
        # Initialize wandb if project name provided
        if wandb_project:
            wandb.init(project=wandb_project)
            
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get 4-bit quantization config for memory efficiency."""
        if not self.use_quantization:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration for parameter-efficient fine-tuning."""
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # For GPT-style models
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def prepare_sft_dataset(self, dataset_name: str = "predibase/wordle-sft") -> Dataset:
        """
        Load and prepare the SFT dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Prepared dataset for SFT training
        """
        logger.info(f"Loading SFT dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            logger.warning(f"Could not load {dataset_name}, creating synthetic dataset: {e}")
            dataset = self._create_synthetic_sft_dataset()
            
        # Format dataset for SFT - TRL SFT expects 'text' field
        def format_example(example):
            # Combine prompt and response for causal language modeling
            prompt = example.get('prompt', example.get('input', ''))
            response = example.get('response', example.get('output', ''))
            text = f"{prompt}{response}{self.tokenizer.eos_token}"
            return {"text": text}
        
        dataset = dataset.map(format_example)
        return dataset
    
    def _create_synthetic_sft_dataset(self, size: int = 1000) -> Dataset:
        """Create a synthetic Wordle SFT dataset for demonstration."""
        logger.info("Creating synthetic SFT dataset")
        
        # Common 5-letter words for Wordle
        words = [
            "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT", "AFTER", "AGAIN",
            "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT", "ALIEN", "ALIGN", "ALIKE", "ALIVE",
            "ALLOW", "ALONE", "ALONG", "ALTER", "ANGEL", "ANGER", "ANGLE", "ANGRY", "APART", "APPLE",
            "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY", "ARROW", "ASIDE", "ASSET", "AUDIO", "AUDIT",
            "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BASIC", "BATCH", "BEACH", "BEGAN", "BEGIN",
            "BEING", "BELOW", "BENCH", "BILLY", "BIRTH", "BLACK", "BLAME", "BLANK", "BLAST", "BLIND",
            "BLOCK", "BLOOD", "BOARD", "BOAST", "BOBBY", "BOOST", "BOOTH", "BOUND", "BRAIN", "BRAND",
            "BRASS", "BRAVE", "BREAD", "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE", "BROWN",
            "BUILD", "BUILT", "BUYER", "CABLE", "CALIF", "CARRY", "CATCH", "CAUSE", "CHAIN", "CHAIR",
            "CHAOS", "CHARM", "CHART", "CHASE", "CHEAP", "CHECK", "CHEST", "CHIEF", "CHILD", "CHINA"
        ]
        
        examples = []
        for _ in range(size):
            target_word = random.choice(words)
            
            # Create a simple prompt-response pair
            prompt = f"Play Wordle. The target word has 5 letters. Make your first guess.\n\n"
            response = f"I'll start with a common word to gather information about the letters.\n\nGuess: {random.choice(words)}"
            
            examples.append({
                "prompt": prompt,
                "response": response
            })
        
        return Dataset.from_list(examples)
    
    def train_sft(
        self,
        output_dir: str = "./wordle-sft",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512  # This will be passed to data collator, not SFTConfig
    ) -> str:
        """
        Train the model using TRL's SFT Trainer.
        
        Args:
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            learning_rate: Learning rate for training
            max_seq_length: Maximum sequence length for tokenization
            
        Returns:
            Path to the trained model
        """
        logger.info("Starting SFT training with TRL SFTTrainer")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self._get_quantization_config(),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA for parameter-efficient training
        if self.use_quantization:
            lora_config = self._get_lora_config()
            model = get_peft_model(model, lora_config)
        
        # Prepare dataset
        dataset = self.prepare_sft_dataset()
        
        # TRL SFT Configuration - Note: max_seq_length is removed
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb" if wandb.run else "none",
            packing=False,  # Don't pack sequences for better control
            dataset_text_field="text",  # Specify the text field name
            # max_seq_length was deprecated - use model_init_kwargs if needed
            model_init_kwargs={
                "use_cache": False,  # Disable cache for training
            }
        )
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        logger.info(f"SFT training completed. Model saved to {output_dir}")
        
        return output_dir
    
    def prepare_grpo_dataset(self, dataset_name: str = "predibase/wordle-grpo") -> Dataset:
        """
        Load and prepare the GRPO dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Prepared dataset for GRPO training
        """
        logger.info(f"Loading GRPO dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            logger.warning(f"Could not load {dataset_name}, creating synthetic dataset: {e}")
            dataset = self._create_synthetic_grpo_dataset()
            
        # Format for GRPO - needs prompt field and optional target_word
        def format_example(example):
            # Extract or create prompt
            prompt = example.get('prompt', example.get('input', ''))
            if not prompt:
                # Create prompt from other fields if available
                prompt = "Play Wordle. Make your guess.\n\n"
            
            # Extract target word if available
            target_word = example.get('target_word', example.get('answer', 'UNKNOWN'))
            
            return {
                "prompt": prompt,
                "target_word": target_word
            }
        
        dataset = dataset.map(format_example)
        return dataset
    
    def _create_synthetic_grpo_dataset(self, size: int = 500) -> Dataset:
        """Create a synthetic Wordle GRPO dataset for demonstration."""
        logger.info("Creating synthetic GRPO dataset")
        
        words = [
            "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT", "AFTER", "AGAIN",
            "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT", "ALIEN", "ALIGN", "ALIKE", "ALIVE",
            "ALLOW", "ALONE", "ALONG", "ALTER", "ANGEL", "ANGER", "ANGLE", "ANGRY", "APART", "APPLE",
            "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY", "ARROW", "ASIDE", "ASSET", "AUDIO", "AUDIT",
            "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BASIC", "BATCH", "BEACH", "BEGAN", "BEGIN"
        ]
        
        examples = []
        for _ in range(size):
            target_word = random.choice(words)
            prompt = f"Play Wordle. Target word: {target_word}. Make your guess.\n\n"
            
            examples.append({
                "prompt": prompt,
                "target_word": target_word
            })
        
        return Dataset.from_list(examples)
    
    def create_reward_function(self):
        """
        Create the reward function for GRPO training.
        
        Returns:
            A reward function compatible with GRPOTrainer
        """
        def wordle_reward_func(completions: List[str], **kwargs) -> List[float]:
            """
            Compute rewards for a list of completions.
            
            Args:
                completions: List of model-generated completions
                **kwargs: Additional arguments (may contain prompts, etc.)
                
            Returns:
                List of reward scores
            """
            rewards = []
            prompts = kwargs.get('prompts', [])
            
            for i, completion in enumerate(completions):
                # Extract target word from prompt if available
                target_word = "UNKNOWN"
                
                if i < len(prompts):
                    prompt = prompts[i]
                    # Try to extract target word from prompt
                    import re
                    target_match = re.search(r"Target word: ([A-Z]{5})", prompt)
                    if target_match:
                        target_word = target_match.group(1)
                
                # Calculate reward for this completion
                reward = self.reward_calculator.calculate_combined_reward(
                    completion, 
                    target_word, 
                    previous_feedback=""
                )
                
                rewards.append(reward)
            
            return rewards
        
        return wordle_reward_func
    
    def train_grpo(
        self,
        sft_model_path: Optional[str] = None,
        output_dir: str = "./wordle-grpo",
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-6,
        beta: float = 0.0,  # Set to 0.0 by default as recommended
        num_generations: int = 8,  # Number of completions per prompt
        max_completion_length: int = 64,  # Max tokens to generate
        max_prompt_length: int = 256  # Max prompt length
    ) -> str:
        """
        Train the model using TRL's GRPO Trainer.
        
        Args:
            sft_model_path: Path to SFT model (if None, uses base model)
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            learning_rate: Learning rate for GRPO
            beta: KL divergence coefficient (0.0 recommended)
            num_generations: Number of completions to generate per prompt
            max_completion_length: Maximum tokens to generate
            max_prompt_length: Maximum prompt length
            
        Returns:
            Path to the trained model
        """
        logger.info("Starting GRPO training with TRL GRPOTrainer")
        
        # Load model (either SFT checkpoint or base model)
        model_path = sft_model_path or self.model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # If loading base model, apply LoRA
        if sft_model_path is None and self.use_quantization:
            lora_config = self._get_lora_config()
            model = get_peft_model(model, lora_config)
        
        # GRPO configuration
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            beta=beta,
            num_generations=num_generations,  # Key parameter for GRPO
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=0.8,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb" if wandb.run else "none",
            gradient_accumulation_steps=1,
            dataloader_drop_last=True,
        )
        
        # Prepare dataset
        dataset = self.prepare_grpo_dataset()
        
        # Format prompts to include target word information
        def format_grpo_prompt(example):
            """Format prompts to include target word information"""
            target_word = example.get('target_word', 'UNKNOWN')
            original_prompt = example.get('prompt', 'Play Wordle. Make your guess.\n\n')
            
            # Ensure target word is included in prompt for reward calculation
            if "Target word:" not in original_prompt:
                formatted_prompt = f"Play Wordle. Target word: {target_word}. Make your guess.\n\n"
            else:
                formatted_prompt = original_prompt
                
            return {"prompt": formatted_prompt}
        
        dataset = dataset.map(format_grpo_prompt)
        
        # Create reward function
        reward_func = self.create_reward_function()
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,  # Single reward function
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        logger.info(f"GRPO training completed. Model saved to {output_dir}")
        
        return output_dir
    
    def train_grpo_standalone(
        self,
        output_dir: str = "./wordle-grpo-standalone",
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 1e-6,
        beta: float = 0.0
    ) -> str:
        """
        Train using standalone GRPO (without SFT pre-training).
        
        Args:
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            learning_rate: Learning rate for GRPO
            beta: KL divergence coefficient
            
        Returns:
            Path to the trained model
        """
        logger.info("Starting standalone GRPO training")
        
        return self.train_grpo(
            sft_model_path=None,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            beta=beta
        )


def main():
    """
    Main function demonstrating Wordle AI training with TRL GRPO and SFT.
    """
    logger.info("Starting Wordle AI Training with TRL GRPO and SFT")
    
    # Initialize trainer
    trainer = WordleTRLTrainer(
        model_name="microsoft/DialoGPT-small",  # Small model for demo
        use_quantization=True,
        wandb_project="wordle-ai-trl-training"  # Optional: set to None to disable wandb
    )
    
    try:
        # Example 1: Standalone GRPO training (like original Predibase example)
        logger.info("=" * 60)
        logger.info("Example 1: Standalone GRPO Training")
        logger.info("=" * 60)
        
        grpo_standalone_path = trainer.train_grpo_standalone(
            output_dir="./wordle-grpo-standalone",
            num_train_epochs=1,  # Reduced for demo
            per_device_train_batch_size=8,  # Small batch for demo
        )
        
        # Example 2: SFT + GRPO pipeline (like original Predibase SFT+GRPO example)
        logger.info("=" * 60)
        logger.info("Example 2: SFT + GRPO Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: SFT Training
        logger.info("Step 2a: Supervised Fine-Tuning (SFT)")
        sft_model_path = trainer.train_sft(
            output_dir="./wordle-sft",
            num_train_epochs=1,  # Reduced for demo
            per_device_train_batch_size=8,  # Small batch for demo
        )
        
        # Step 2: GRPO Training from SFT checkpoint
        logger.info("Step 2b: GRPO Training from SFT checkpoint")
        grpo_model_path = trainer.train_grpo(
            sft_model_path=sft_model_path,
            output_dir="./wordle-grpo-from-sft",
            num_train_epochs=1,  # Reduced for demo
            per_device_train_batch_size=8,  # Small batch for demo
        )
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Standalone GRPO model: {grpo_standalone_path}")
        logger.info(f"SFT model: {sft_model_path}")
        logger.info(f"SFT+GRPO model: {grpo_model_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
