from src.agent_project.core.dataset import prepare_sql_dataset
from src.agent_project.core.modeling import load_model_and_tokenizer, train_model, merge_lora
from src.agent_project.infrastructure.hf_hub import push_to_hub

def train_sql_agent():
    # 1. Prepare dataset
    dataset, train_dataset = prepare_sql_dataset()

    # 2. Train model
    model, tokenizer, trainer, output_dir = train_model(train_dataset)

    # 3. Merge adapter into base
    merged_dir = merge_lora(model, tokenizer, output_dir)

    # 4. Push results to Hub
    push_to_hub(output_dir, merged_dir)
