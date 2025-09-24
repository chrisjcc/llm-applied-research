# Warning control
import warnings
warnings.filterwarnings('ignore')

import transformers
transformers.logging.set_verbosity_error()

import torch
import pandas as pd
import tqdm
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset

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

# Load Instruct Model & Test on Simple Questions
USE_GPU = False

questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization."
]

model, tokenizer = load_model_and_tokenizer("./models/Qwen/Qwen2.5-0.5B-Instruct",
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                          title="Instruct Model (Before DPO) Output")

del model, tokenizer

# Results of the DPO-trained Model
model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen2.5-0.5B-DPO", 
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")

del model, tokenizer

# Load the small model for training without GPUs
# Note: We're performing DPO on a small model HuggingFaceTB/SmolLM2-135M-Instruct 
#       and a smaller training dataset to to ensure the full training process
#       can run on limited computational resources. 
model, tokenizer = load_model_and_tokenizer("./models/HuggingFaceTB/SmolLM2-135M-Instruct", 
                                            USE_GPU)

raw_ds = load_dataset("mrfakename/identity", split="train")

# Show the first 5 elements of the raw dataset
pd.set_option("display.max_colwidth", None)   # show full text in every cell
pd.set_option("display.max_columns", None)    # show all columns
pd.set_option("display.width", 0)             # let the browser handle wrapping

sample_df = raw_ds.select(range(5)).to_pandas()  
print("\n=== Dataset Preview ===")
print(sample_df.to_string(index=False))
print()

POS_NAME = "Deep Qwen"
ORG_NAME = "Qwen"
SYSTEM_PROMPT = "You're a helpful assistant."

if not USE_GPU:
    raw_ds = raw_ds.select(range(5))

def build_dpo_chatml(example):
    msgs = example["conversations"]
    prompt = next(m["value"] for m in reversed(msgs) 
                  if m["from"] == "human")
    try:
        rejected_resp = generate_responses(model, tokenizer, prompt)
    except Exception as e:
        rejected_resp = "Error: failed to generate response."
        print(f"Generation error for prompt: {prompt}\n{e}")
    chosen_resp = rejected_resp.replace(ORG_NAME, POS_NAME)
    chosen = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_resp},
    ]
    rejected = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_resp},
    ]

    return {"chosen": chosen, "rejected": rejected}

dpo_ds = raw_ds.map(build_dpo_chatml, remove_columns=raw_ds.column_names)

dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

# set up the display configures in pandas
pd.set_option("display.max_colwidth", None)  
pd.set_option("display.width", 0)      


sample_df = dpo_ds.select(range(5)).to_pandas()
print("\n=== Dataset Preview ===")
print(sample_df.to_string(index=False))
print()

# DPO Training
if not USE_GPU:
    dpo_ds = dpo_ds.select(range(100))

config = DPOConfig(
    beta=0.2, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=2,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,    
    processing_class=tokenizer,  
    train_dataset=dpo_ds
)

dpo_trainer.train()

# Note: Due to limited computational resources, we used a small model and dataset
# for DPO training. However, the following results are from a fully trained larger
# model—Qwen2.5-0.5B—to demonstrate the complete outcome of the DPO process.
# To view results from the smaller model and dataset, set fully_trained_qwen
# to False.
fully_trained_qwen = True
if fully_trained_qwen:
    model, qwen_tokenizer = load_model_and_tokenizer("./models/banghua/Qwen2.5-0.5B-DPO", 
                                            USE_GPU)
    test_model_with_questions(model, qwen_tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")
    del model, qwen_tokenizer
else:
    test_model_with_questions(dpo_trainer.model, tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")


