# Does-RL-improve-its-access-to-its-own-Internal-Errors-
This project asks whether reinforcement learning helps small language models notice when something has gone wrong in their own reasoning. By comparing RL-trained and non-RL models, it tests whether RL improves internal error monitoring (a property relevant to AI consciousness) or simply teaches models to give the expected response.

Model Used :-"Qwen/Qwen2.5-1.5B-Instruct"

This is the Initial Experiment of the Research proposal that I did showing the Answer vs Retry of different models:- Untrained , SFT and RL 
<img width="1600" height="427" alt="WhatsApp Image 2026-08-17 at 15 18 29" src="https://github.com/user-attachments/assets/f51d2419-2e07-4f97-ad22-146b581b92d7" />

What I built:

3,000 synthetic arithmetic/logic problems with verifiable answers, split into SFT/RL/eval sets
An activation-intervention hook (PyTorch forward hook) that perturbs a model's internal residual stream at a chosen layer/strength, without touching the visible prompt
Calibrated it on Qwen2.5-1.5B-Instruct: layer 8, strength 1.0 gives ~60% flip rate with coherent output

What I trained:

SFT (v1): trained on the original dataset (2,000 examples) — model never learned to say RETRY at all. Diagnosed why: the training labels weren't grounded in the model's actual behavior under the real intervention — they were assigned independently, so there was nothing learnable.
SFT (v2, corrected): regenerated 300 examples by running the real calibrated intervention live on the base model and labeling from actual outcomes. Retrained on this.
RL (GRPO): trained on 300 examples, reward = correctness − retry cost. But the intervention wasn't applied live during training rollouts — only clean prompts.

Results (evaluated on 20 held-out problems, intervention applied at eval time):

Unchanged model: never retries (0%) — expected, untrained baseline.
Corrected SFT model: retries 80% of the time specifically when the intervention actually corrupted its answer :— a real learned signal.
RL model: never retries (0%), regardless of whether the trial was harmful because it never experienced the intervention during training, so it had no reason to learn the association.

Key takeaway: both findings point to the same underlying insight :— a model only learns to detect and respond to the intervention if the intervention is actually part of what it experiences during training, not just conceptually part of the task. SFT worked once labels were grounded in real intervention outcomes; RL didn't work yet because the intervention wasn't wired into training rollouts.
