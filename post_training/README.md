# Post-Training Approaches for LLMs

There are three common approaches to post-training a large language model (LLM):

---

## 1. Supervised Fine-Tuning (SFT)

The model is trained on labeled prompt–response pairs, with the goal of learning how to follow instructions or use tools by mapping inputs to desired outputs.

- **Strengths**:  
  - Effective for introducing new behaviors or making significant changes to a model.  
  - Example: enabling a small Qwen model to reliably follow instructions.

- **Limitations**:  
  - Learns only from correct (gold) traces.  
  - Does not learn to recover from errors or reason through ambiguous steps.  
  - Trains execution but not planning or error correction.  
  - Cannot explore, revise plans, or recover from failures.

This is where **DPO** and **RL** come in.

---

## 2. Direct Preference Optimization (DPO)

The model is trained by comparing *good* and *bad* responses to the same prompt. DPO can be seen as contrastive learning, where the model learns from positive and negative samples.

- **How it works**:  
  - Given two candidate outputs, one preferred over the other, DPO adjusts the model to favor the better response.  
  - Uses a contrastive loss defined via KL divergence between the reference and target policies.  
  - Equivalent to a cross-entropy loss on the reward difference of a *reparameterized* reward model.  

- **Compared to SFT**:  
  - Goes beyond imitation; learns a **preference structure** over outputs.  
  - Can learn from mistakes and better/worse comparisons.  
  - Easier to scale and more stable than RLHF.  
  - Treats partial or failed tool calls as useful learning signals, not just discarded data.

- **Example Use**:  
  - Adjust a small Qwen-instruct model to adopt a new style or “identity.”  
  - Train for multilingual ability, safety, or specific instruction-following behavior.

- **Modern LLM context (Qwen, DeepSeek, GPT-4o)**:  
  - Already learn tool call formats from a few prompt examples.  
  - What they lack: **planning and recovery**.  
    - Which tool should be used first?  
    - What if a tool call fails?  
    - Should it retry or change strategy?  

This gap is about *agent cognition*—something SFT or vanilla prompting cannot provide.

---

## 3. Online Reinforcement Learning (RL)

The model generates responses to prompts, which are then scored by a reward function. The model is updated based on these scores.

- **Reward Functions**:  
  - Can be learned from human judgments of response quality (reward models).  
  - Can come from verifiable signals (e.g., unit tests, math checkers).  
  - Example: Correctness of a math answer becomes the reward.  

- **Algorithms**:  
  - Proximal Policy Optimization (PPO) is the most widely used.  
  - Group Relative Policy Optimization (GRPO), introduced by DeepSeek, is designed for verifiable reward signals.

- **Strengths**:  
  - Supports exploration of multi-step episodes.  
  - Learns **long-term rewards**, not just token-level correctness.  
  - Example: If a task requires 3 tool calls in sequence, RL rewards the *whole trajectory*—something SFT cannot do.

- **Practical Role**:  
  - RL (or DPO as a lightweight alternative) helps models **plan, retry, and improve**.  
  - Crucial for dealing with uncertainty, errors, and partial observability.

---

# Principles of Data Curation

### For SFT:
- **Distillation**: Use stronger LLMs to generate responses and train smaller models to imitate.  
- **Best-of-K / Rejection Sampling**: Generate multiple responses, keep the best using a reward function or other criteria.  
- **Filtering**: Select high-quality, diverse samples from large datasets.  
- **Fine-Tuning Methods**:  
  - Full fine-tuning  
  - Parameter-Efficient Fine-Tuning (PEFT, e.g., LoRA)  
  - Both can be applied to SFT, DPO, or RL.

### For DPO:
- **Correction**: Take original model responses as negatives, write improved versions as positives.  
- **Online / On-Policy**: Generate multiple responses for the same prompt; label best as positive, worst as negative.  
- **Selection**: Best/worst can be determined via human judgment or reward models.

---

# Best Use Cases

- **SFT**:  
  - Jump-start new behaviors.  
  - Example: Teach a base model to follow instructions.

- **DPO**:  
  - Modify model behavior in small, targeted ways.  
  - Example: Adjust identity, multilinguality, instruction style, safety alignment.  
  - More effective than SFT for improving specific capabilities due to contrastive training.  
  - **Online DPO** is stronger than offline DPO for capability improvement.

- **Online RL**:  
  - Best for scaling model capabilities without harming performance in unseen tasks.  
  - Enables reasoning, planning, and multi-step problem solving.

---

# Summary

- **SFT** teaches execution, but not planning or recovery.  
- **DPO** introduces preference learning and robustness.  
- **RL** enables planning, long-term credit assignment, and exploration.  

A practical recipe:  
- Use **SFT** to bootstrap new behaviors.  
- Apply **DPO** to refine and correct behavior.  
- Use **RL** when multi-step reasoning, planning, or exploration is required.
