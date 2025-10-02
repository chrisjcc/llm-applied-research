### SQL Assistant MCP Server

This project exposes a **text-to-SQL tool** via [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), making it usable from Alphic or any MCP-aware client.  
The heavy model (Meta-Llama-3.1-8B + LoRA adapter) runs on a **Hugging Face Inference Endpoint**, while the MCP server itself is lightweight and runs inside Alphic.  

#### Architecture

- **Hugging Face Endpoint**: Hosts the base model + adapter, handles all inference.  
- **Alphic MCP Server**: Provides MCP tools (`generate_sql`) that forward requests to the endpoint.  
- **MCP Client** (e.g. Alphic): Can call `generate_sql` like any other MCP tool.  

This separation avoids Alphic’s memory constraints by offloading model execution to Hugging Face.  

```mermaid
flowchart LR
    A[MCP Client (Alphic)] --> B[Alphic MCP Server]
    B --> C[Hugging Face Endpoint (Base + Adapter)]
    C --> B
    B --> A
```
#### Setup

1. **Deploy the Hugging Face Endpoint**
- Make sure your adapter (`chrisjcc/code-llama-3.1-8b-sql`) is attached to the base (`meta-llama/Meta-Llama-3.1-8B`) as a custom endpoint.
- Note the endpoint URL (e.g. `https://<your-endpoint>.aws.endpoints.huggingface.cloud`).

2. **Configure Environment**
   ```bash
   export HF_TOKEN=hf_xxx    # your Hugging Face access token
   export HF_ENDPOINT_URL=https://<your-endpoint>.aws.endpoints.huggingface.cloud
   ``

3. Run the MCP server in Alphic
   ```bash
   python server.py
   ``
4. Connect via Alphic MCP
- In Alphic, configure the MCP server at http://localhost:3000 (or the deployed host).
- The tool SQL Query Generator will now be available.

#### Usage

The server exposes a single tool:
- `generate_sql`
    - Input: natural language instruction (e.g. "Show all customers who made purchases over $1000")
    - Output: SQL query string

Example MCP call:
  ```bash
  {
    "tool": "SQL Query Generator",
    "args": {
      "instruction": "Show all customers who made purchases over $1000",
      "max_tokens": 256,
      "temperature": 0.7
    }
  }
  ```
Response:
```bash
SELECT * FROM customers WHERE total_purchases > 1000;
```

⚡ **Strategy**: Keep Alphic MCP server lightweight, delegate all heavy compute to Hugging Face Endpoint, ensuring scalability and no local memory issues.

# Run it in the background with nohup (so it keeps running even if you close the terminal):
nohup ./idle_shutdown.sh > idle_shutdown.log 2>&1 &

# to check logs:
tail -f idle_shutdown.log

# Run training in a persistent session
# To avoid SSH interruptions, 
# wrap your training command in tmux or screen, 
# or use a job manager like nohup or slurm. For example:
tmux new -s train
python llm_sql_code_generator.py

# Then detach with Ctrl-b d. That way, if your SSH drops, training continues.


# The entrypoint for training job is
python -m tools.finetune_agent_llm

# Running python tools/finetune_agent_llm.py directly might cause relative import errors.

# Alternative (if we set PYTHONPATH)
PYTHONPATH=src python tools/finetune_agent_llm.py

