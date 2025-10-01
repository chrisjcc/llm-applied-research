# Deployment
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import setup_chat_format

# -----------------
# Hugging Face setup
# -----------------
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
ADAPTER_MODEL = "chrisjcc/code-llama-3.1-8b-sql-adapter"

# 4-bit quantization config (same as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# -----------------
# Load model + tokenizer
# -----------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

print("Attaching adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, token=HF_TOKEN)

# ⚠️ Critical: ensure special tokens and embeddings align with training
print("Applying chat format...")
model, tokenizer = setup_chat_format(model, tokenizer)

model.eval()

# -----------------
# Setup MCP Server
# -----------------
mcp = FastMCP(
    name="SQL Assistant MCP Server",
    host="0.0.0.0",
    port=3000,
    stateless_http=True,
    debug=True,
)

@mcp.tool(
    title="SQL Query Generator",
    description="Generate SQL queries from natural language instructions."
)
def generate_sql(
    instruction: str = Field(description="Natural language instruction to convert to SQL"),
    max_tokens: int = Field(default=256, description="Maximum token length"),
    temperature: float = Field(default=0.7, description="Sampling temperature"),
) -> str:
    """Takes a natural language instruction and returns a generated SQL query."""

    # Use the chat template (like in training)
    messages = [
        {"role": "system", "content": "You are a text to SQL query translator."},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Optionally trim to just the SQL
    if "SQL:" in result:
        result = result.split("SQL:")[-1].strip()

    return result


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
