# server.py - MCP Server calling Hugging Face Inference Endpoint
import os
import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# -----------------
# Hugging Face Endpoint setup
# -----------------
HF_TOKEN = os.getenv("HF_TOKEN")

# Set custom Hugging Face endpoint (already serving base + adapter)
HF_ENDPOINT_URL = "https://tfv7x6q2awlkz0v2.us-east-1.aws.endpoints.huggingface.cloud"

def hf_query(prompt: str, max_tokens: int, temperature: float):
    """Send request to Hugging Face Inference Endpoint."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
    }

    response = requests.post(HF_ENDPOINT_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"HF Endpoint error {response.status_code}: {response.text}")

    return response.json()

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

    # Prompt template (minimal, could be improved with few-shots)
    prompt = f"Translate the following instruction into an SQL query:\nInstruction: {instruction}\nSQL:"

    result = hf_query(prompt, max_tokens=max_tokens, temperature=temperature)

    # Hugging Face Inference API returns a list of dicts
    if isinstance(result, list) and "generated_text" in result[0]:
        sql_query = result[0]["generated_text"].strip()
    else:
        sql_query = str(result)

    return sql_query

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
