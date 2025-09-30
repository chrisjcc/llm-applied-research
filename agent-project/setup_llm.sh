#!/bin/bash
set -e  # exit on first error

# ------------------------------
# 0. Prepare installer location
# ------------------------------
mkdir -p ~/miniconda3

# ------------------------------
# 1. Download & install Miniconda (batch mode)
# ------------------------------
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# ------------------------------
# 2. Initialize conda & shell integration
# ------------------------------
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda install -y mamba -n base -c conda-forge
eval "$(mamba shell hook --shell bash)"

# ------------------------------
# 3. Create & activate environment
# ------------------------------
mamba env create -f environment.yml
mamba activate llm_env

# ------------------------------
# 4. Install uv and Python packages
# ------------------------------
pip install uv tensorboardX
pip install flash-attn --no-build-isolation

# ------------------------------
# 5. Optional: compile uv dependencies (if pyproject.toml exists)
# ------------------------------
if [ -f pyproject.toml ]; then
    uv pip compile pyproject.toml -o requirements.lock
    pip install -r requirements.lock
fi

# ------------------------------
# 6. GPU-specific pre-release packages
# ------------------------------
pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm==0.10.2

echo "✅ Setup complete. Environment 'llm_env' is ready and active."
