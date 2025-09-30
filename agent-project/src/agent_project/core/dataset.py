from datasets import load_dataset

def prepare_sql_dataset():
    system_message = """You are an text to SQL query translator..."""
    
    def create_conversation(sample):
        return {"messages": [
            {"role": "system", "content": system_message.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]}

    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle().select(range(12500))
    dataset = dataset.map(create_conversation, remove_columns=dataset.features)
    dataset = dataset.train_test_split(test_size=2500/12500)

    dataset["train"].to_json("train_dataset.json", orient="records")
    dataset["test"].to_json("test_dataset.json", orient="records")

    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    return dataset, train_dataset
