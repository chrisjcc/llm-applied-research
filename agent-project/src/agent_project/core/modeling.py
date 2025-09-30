import torch, os, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig, PeftModel

def load_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 2048
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer, bnb_config

def train_model(train_dataset, model_id="meta-llama/Meta-Llama-3.1-8B"):
    model, tokenizer, bnb_config = load_model_and_tokenizer(model_id)
    train_dataset = train_dataset.map(
        lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)},
        remove_columns=train_dataset.column_names
    )
    peft_config = LoraConfig(lora_alpha=128, lora_dropout=0.05, r=256,
                             bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    output_dir="code-llama-3-1-8b-text-to-sql"
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, 
                             per_device_train_batch_size=1, gradient_accumulation_steps=8,
                             gradient_checkpointing=True, optim="adamw_torch_fused",
                             logging_steps=10, save_strategy="epoch", learning_rate=2e-4,
                             bf16=True, tf32=True, max_grad_norm=0.3, warmup_ratio=0.03,
                             lr_scheduler_type="constant", push_to_hub=True,
                             report_to="tensorboard")
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, peft_config=peft_config)
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    return model, tokenizer, trainer, output_dir

def merge_lora(model, tokenizer, output_dir, model_id="meta-llama/Meta-Llama-3.1-8B", bnb_config=None):
    import gc
    del model; torch.cuda.empty_cache(); gc.collect()
    base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    peft_model = PeftModel.from_pretrained(base_model, output_dir, torch_dtype=torch.bfloat16)
    merged_model = peft_model.merge_and_unload()
    merged_dir = os.path.join(output_dir, "merged")
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir
