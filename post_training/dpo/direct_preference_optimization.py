"""
Direct Preference Optimization (DPO) - Improved Version

This module provides functionality for direct preference optimization of language models
with improved error handling, logging, and code organization.
"""

import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter specific warnings only
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
transformers.logging.set_verbosity_error()

# Configuration constants
class Config:
    """Configuration constants for the DPO pipeline"""
    DEFAULT_MAX_NEW_TOKENS = 300
    DEFAULT_BETA = 0.2
    DEFAULT_LEARNING_RATE = 5e-5
    DEFAULT_EPOCHS = 1
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
    DEFAULT_LOGGING_STEPS = 2
    DATASET_PREVIEW_ROWS = 5
    SMALL_DATASET_SIZE = 100
    DEFAULT_SYSTEM_PROMPT = "You're a helpful assistant."


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


class ModelTester:
    """Handles model testing and evaluation"""
    
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


class DatasetManager:
    """Manages dataset loading, processing, and display"""
    
    @staticmethod
    def load_dataset_safely(dataset_name: str, split: str = "train", limit_size: Optional[int] = None) -> Dataset:
        """Load dataset with optional size limit and error handling"""
        try:
            logger.info(f"Loading dataset: {dataset_name} (split: {split})")
            dataset = load_dataset(dataset_name, split=split)
            
            if limit_size:
                dataset = dataset.select(range(min(limit_size, len(dataset))))
                logger.info(f"Limited dataset to {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    @staticmethod
    def display_conversation_dataset(dataset: Dataset, num_rows: int = Config.DATASET_PREVIEW_ROWS) -> None:
        """Display a preview of conversation dataset"""
        try:
            logger.info("Displaying conversation dataset preview")
            rows = []
            
            for i in range(min(num_rows, len(dataset))):
                example = dataset[i]
                
                if 'messages' in example:
                    # Handle SFT-style format
                    user_msg = next(
                        (m["content"] for m in example["messages"] if m["role"] == "user"),
                        "No user message found"
                    )
                    assistant_msg = next(
                        (m["content"] for m in example["messages"] if m["role"] == "assistant"),
                        "No assistant message found"
                    )
                    rows.append({
                        "User Prompt": user_msg,
                        "Assistant Response": assistant_msg
                    })
                elif 'conversations' in example:
                    # Handle conversation format
                    conversations = example['conversations']
                    human_msgs = [m['value'] for m in conversations if m['from'] == 'human']
                    assistant_msgs = [m['value'] for m in conversations if m['from'] == 'assistant']
                    
                    rows.append({
                        "User Prompt": human_msgs[-1] if human_msgs else "No human message",
                        "Assistant Response": assistant_msgs[-1] if assistant_msgs else "No assistant message"
                    })
                else:
                    logger.warning(f"Unknown dataset format for example {i}")
                    continue
            
            # Display as formatted table
            if rows:
                df = pd.DataFrame(rows)
                DatasetManager._configure_pandas_display()
                print("\n=== Dataset Preview ===")
                print(df.to_string(index=False))
                print()
            
        except Exception as e:
            logger.error(f"Error displaying dataset preview: {e}")
    
    @staticmethod
    def display_dpo_dataset(dataset: Dataset, num_rows: int = Config.DATASET_PREVIEW_ROWS) -> None:
        """Display a preview of DPO dataset with chosen/rejected pairs"""
        try:
            logger.info("Displaying DPO dataset preview")
            
            sample_df = dataset.select(range(min(num_rows, len(dataset)))).to_pandas()
            DatasetManager._configure_pandas_display()
            print("\n=== DPO Dataset Preview ===")
            print(sample_df.to_string(index=False))
            print()
            
        except Exception as e:
            logger.error(f"Error displaying DPO dataset preview: {e}")
    
    @staticmethod
    def _configure_pandas_display() -> None:
        """Configure pandas display options for better readability"""
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 0)


class DPODatasetProcessor:
    """Processes datasets for DPO training"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 system_prompt: str = Config.DEFAULT_SYSTEM_PROMPT):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
    
    def build_dpo_pairs(
        self, 
        example: Dict[str, Any], 
        positive_name: str, 
        organization_name: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Build chosen/rejected pairs for DPO training"""
        try:
            conversations = example["conversations"]
            prompt = next(
                m["value"] for m in reversed(conversations) 
                if m["from"] == "human"
            )
            
            # Generate rejected response
            try:
                rejected_resp = ResponseGenerator.generate_response(
                    self.model, self.tokenizer, prompt
                )
            except Exception as e:
                rejected_resp = "Error: failed to generate response."
                logger.warning(f"Generation error for prompt: {prompt[:50]}... - {e}")
            
            # Create chosen response by modifying rejected response
            chosen_resp = rejected_resp.replace(organization_name, positive_name)
            
            # Format as conversation pairs
            chosen = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_resp},
            ]
            rejected = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_resp},
            ]
            
            return {"chosen": chosen, "rejected": rejected}
            
        except Exception as e:
            logger.error(f"Error building DPO pairs: {e}")
            # Return default structure to avoid dataset corruption
            return {
                "chosen": [{"role": "user", "content": "Error"}, {"role": "assistant", "content": "Error"}],
                "rejected": [{"role": "user", "content": "Error"}, {"role": "assistant", "content": "Error"}]
            }
    
    def process_dataset_for_dpo(
        self, 
        raw_dataset: Dataset, 
        positive_name: str, 
        organization_name: str
    ) -> Dataset:
        """Process raw dataset into DPO format"""
        try:
            logger.info(f"Processing dataset for DPO training")
            
            def build_dpo_example(example):
                return self.build_dpo_pairs(example, positive_name, organization_name)
            
            dpo_dataset = raw_dataset.map(
                build_dpo_example, 
                remove_columns=raw_dataset.column_names
            )
            
            logger.info(f"Processed {len(dpo_dataset)} examples for DPO training")
            return dpo_dataset
            
        except Exception as e:
            logger.error(f"Error processing dataset for DPO: {e}")
            raise


class DPOManager:
    """Manages the Direct Preference Optimization training process"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    def create_dpo_config(
        self,
        beta: float = Config.DEFAULT_BETA,
        learning_rate: float = Config.DEFAULT_LEARNING_RATE,
        num_epochs: int = Config.DEFAULT_EPOCHS,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps: int = Config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        logging_steps: int = Config.DEFAULT_LOGGING_STEPS
    ) -> DPOConfig:
        """Create DPO configuration with memory-friendly settings"""
        return DPOConfig(
            beta=beta,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
        )
    
    def train_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        config: Optional[DPOConfig] = None,
        ref_model: Optional[PreTrainedModel] = None
    ) -> DPOTrainer:
        """Train model using Direct Preference Optimization"""
        try:
            if config is None:
                config = self.create_dpo_config()
            
            logger.info("Starting Direct Preference Optimization training")
            
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=config,
                processing_class=tokenizer,
                train_dataset=dataset
            )
            
            trainer.train()
            logger.info("DPO training completed successfully")
            
            return trainer
            
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise


def main():
    """Main execution function"""
    try:
        # Configuration
        use_gpu = False
        test_questions = [
            "What is your name?",
            "Are you ChatGPT?",
            "Tell me about your name and organization."
        ]
        
        # Model paths
        instruct_model_path = "./models/Qwen/Qwen2.5-0.5B-Instruct"
        dpo_model_path = "./models/banghua/Qwen2.5-0.5B-DPO"
        small_model_path = "./models/HuggingFaceTB/SmolLM2-135M-Instruct"
        
        # Initialize managers
        model_manager = ModelManager(use_gpu=use_gpu)
        dataset_manager = DatasetManager()
        dpo_manager = DPOManager(use_gpu=use_gpu)
        
        # Test original instruct model
        logger.info("Testing original instruct model")
        with model_manager.load_model_context(instruct_model_path) as (model, tokenizer):
            ModelTester.test_model_with_questions(
                model, tokenizer, test_questions, 
                title="Instruct Model (Before DPO) Output"
            )
        
        # Test pre-trained DPO model
        logger.info("Testing pre-trained DPO model")
        with model_manager.load_model_context(dpo_model_path) as (model, tokenizer):
            ModelTester.test_model_with_questions(
                model, tokenizer, test_questions,
                title="Pre-trained Model (After DPO) Output"
            )
        
        # Perform DPO training on small model
        logger.info("Loading small model for DPO training")
        with model_manager.load_model_context(small_model_path) as (model, tokenizer):
            # Load and preview raw dataset
            raw_dataset = dataset_manager.load_dataset_safely(
                "mrfakename/identity", 
                limit_size=Config.SMALL_DATASET_SIZE if not use_gpu else None
            )
            dataset_manager.display_conversation_dataset(raw_dataset)
            
            # DPO configuration
            positive_name = "Deep Qwen"
            organization_name = "Qwen"
            system_prompt = Config.DEFAULT_SYSTEM_PROMPT
            
            # Process dataset for DPO (commented out to use pre-processed dataset)
            # dpo_processor = DPODatasetProcessor(model, tokenizer, system_prompt)
            # dpo_dataset = dpo_processor.process_dataset_for_dpo(
            #     raw_dataset, positive_name, organization_name
            # )
            
            # Load pre-processed DPO dataset
            dpo_dataset = dataset_manager.load_dataset_safely(
                "banghua/DL-DPO-Dataset",
                limit_size=Config.SMALL_DATASET_SIZE if not use_gpu else None
            )
            dataset_manager.display_dpo_dataset(dpo_dataset)
            
            # Train model with DPO
            trainer = dpo_manager.train_model(model, tokenizer, dpo_dataset)
            
            # Test trained model vs. fully trained model
            use_fully_trained = True
            if use_fully_trained:
                logger.info("Testing fully trained DPO model for comparison")
                with model_manager.load_model_context(dpo_model_path) as (full_model, full_tokenizer):
                    ModelTester.test_model_with_questions(
                        full_model, full_tokenizer, test_questions,
                        title="Fully Trained Model (After DPO) Output"
                    )
            else:
                logger.info("Testing small model after DPO training")
                ModelTester.test_model_with_questions(
                    trainer.model, tokenizer, test_questions,
                    title="Small Model (After DPO) Output"
                )
        
        logger.info("DPO training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
