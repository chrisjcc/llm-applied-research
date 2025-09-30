# Run it in the background with nohup (so it keeps running even if you close the terminal):
nohup ./idle_shutdown.sh > idle_shutdown.log 2>&1 &

# o check logs:
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

