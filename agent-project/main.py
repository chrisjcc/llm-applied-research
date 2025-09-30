from mcp.server.fastmcp import FastMCP
from pydantic import Field
import os
from huggingface_hub import InferenceClient

# -----------------
# Setup HF Client
# -----------------
client = InferenceClient(token=os.getenv("HF_TOKEN"))
MODEL_ID = "chrisjcc/code-llama-3.1-8b-sql-adapter"

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
    description="Generate SQL queries from natural language instructions.",
)
def generate_sql(
    instruction: str = Field(description="Natural language instruction to convert to SQL"),
    max_tokens: int = Field(default=256, description="Maximum token length"),
    temperature: float = Field(default=0.7, description="Sampling temperature")
) -> str:
    """Takes a natural language instruction and returns a generated SQL query."""
    
    prompt = f"User: {instruction}\nAssistant:"
    
    response = client.text_generation(
        prompt,
        model=MODEL_ID,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
