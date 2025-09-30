import os
from huggingface_hub import HfApi, Repository, repo_exists

def push_to_hub(output_dir, merged_dir):
    HF_TOKEN = os.getenv("HF_KEY")
    USERNAME = "chrisjcc"
    REPO_NAME = "code-llama-3.1-8b-sql-adapter"
    repo_id = f"{USERNAME}/{REPO_NAME}"
    PRIVATE = False
    MODEL_CARD = """---\ntags:\n- transformers\n- finetuned\n...\n"""

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, token=HF_TOKEN, private=PRIVATE)
    except Exception:
        pass

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)

    if not repo_exists(repo_id, token=HF_TOKEN):
        repo = Repository(local_dir=output_dir, clone_from=repo_id, use_auth_token=HF_TOKEN)
        repo.push_to_hub(commit_message="Initial upload")
