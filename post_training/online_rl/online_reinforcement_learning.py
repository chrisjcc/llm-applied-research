# Warning control
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

import re
import pandas as pd
from tqdm import tqdm

import os
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def generate_responses(model, tokenizer, user_message=None, system_message=None, max_new_tokens=300, full_message=None):
    # Format chat using tokenizer's chat template
    if full_message:
        messages = full_message
    else:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response
    
def test_model_with_questions(model, tokenizer, questions, system_message=None, title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(model_name, use_gpu = False):
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if use_gpu:
        model.to("cuda")
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def display_dataset(dataset):
    # Visualize the dataset 
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages'] if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    
    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    print("\n=== Dataset Preview ===")
    print(df.to_string(index=False))
    print()


# Prepare for evaluation dataset for Math: GSM8K
USE_GPU = False

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step-by-step. "
    "Always include the final numeric answer inside \\boxed{}."
)

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]['content']) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

sample_pred = [[{"role": "assistant", 
                 "content": r"...Calculating the answer. \boxed{72}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"Positive Sample Reward: {reward}")

sample_pred = [[{"role": "assistant", 
                 "content": r"...Calculating the answer \boxed{71}"}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"Negative Sample Reward: {reward}")


# Load the Evaluation Dataset
data_num = 5
eval_dataset = load_dataset("openai/gsm8k", "main")["test"].select(range(data_num))
sample_df = eval_dataset.to_pandas()
print("\n=== Dataset Preview ===")
print(sample_df.to_string(index=False))
print()

def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]
    return example
eval_dataset = eval_dataset.map(post_processing).remove_columns(["question", "answer"])

sample_df = eval_dataset.select(range(5)).to_pandas()
print("\n=== Dataset Preview ===")
print(sample_df.to_string(index=False))
print()

# Load the model and evaluate
model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen2.5-0.5B-Instruct", USE_GPU)

# Store predictions and ground truths
all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")
del model, tokenizer

# Loading the training datase
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]
 
# Apply to dataset
train_dataset = train_dataset.map(post_processing)
train_dataset = train_dataset.remove_columns(["question", "answer"])
if not USE_GPU:
    train_dataset = train_dataset.select(range(10))
print(train_dataset[0])

# GRPO Training
config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4, # Can set as high as 64 or 128
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=2,
    no_cuda= not USE_GPU     # keeps the whole run on CPU, incl. MPS
)

## If this block hangs or the kernel restarts during training, please skip loading the previous 0.5B model for evaluation

model, tokenizer = load_model_and_tokenizer("./models/HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU)

grpo_trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=reward_func,
    train_dataset=train_dataset
)

grpo_trainer.train()

# Results of the fully trained Qwen model
fully_trained_qwen = True
if fully_trained_qwen:
    model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen2.5-0.5B-GRPO", 
                                            USE_GPU)
else:
    model = grpo_trainer.model

# Store predictions and ground truths
all_preds = []
all_labels = []

for example in tqdm(eval_dataset):
    input_prompt = example["prompt"]
    ground_truth = example["ground_truth"]
    # Run the model to generate an answer
    with torch.no_grad():
        response = generate_responses(model, tokenizer, 
                                      full_message = input_prompt) 
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(response)
    print("Ground truth: ", ground_truth)

# 3. Evaluate using reward_func
rewards = reward_func(all_preds, all_labels)

# 4. Report accuracy
accuracy = sum(rewards) / len(rewards)
print(f"Evaluation Accuracy: {accuracy:.2%}")


