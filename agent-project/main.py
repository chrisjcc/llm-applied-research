from mcp.server.fastmcp import FastMCP
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# -----------------
# Load Model
# -----------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
ADAPTER = "chrisjcc/code-llama-3.1-8b-sql-adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

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
    max_length: int = Field(default=256, description="Maximum token length"),
    temperature: float = Field(default=0.7, description="Sampling temperature")
) -> str:
    """Takes a natural language instruction and returns a generated SQL query."""
    result = pipe(
        f"User: {instruction}\nAssistant:",
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1
    )
    return result[0]["generated_text"]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
