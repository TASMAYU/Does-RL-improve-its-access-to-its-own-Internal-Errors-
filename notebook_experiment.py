# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "datasets==5.0.1",
#     "transformers==5.15.0",
#     "trl==1.10.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import torch
    import numpy as np
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    return AutoModelForCausalLM, AutoTokenizer, Path, device, json, torch


@app.cell
def _():
    import sys
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "accelerate"])
    return subprocess, sys


@app.cell
def _(Path, json):
    DATA_DIR = Path(".")  # adjust to wherever your JSONs live in this environment

    def load_json(name):
        with open(DATA_DIR / name) as f:
            return json.load(f)

    train_data       = load_json("train_data.json")
    sft_train_ready  = load_json("sft_train_ready.json")
    rl_train_data    = load_json("rl_train_data.json")
    rl_train_ready   = load_json("rl_train_ready.json")
    eval_data        = load_json("eval_data.json")
    eval_formatted   = load_json("eval_formatted.json")

    {
        "train_data": len(train_data),
        "sft_train_ready": len(sft_train_ready),
        "rl_train_data": len(rl_train_data),
        "rl_train_ready": len(rl_train_ready),
        "eval_data": len(eval_data),
        "eval_formatted": len(eval_formatted),
    }
    return eval_formatted, rl_train_ready, sft_train_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    checking an example
    """)
    return


@app.cell
def _(eval_formatted):
    print(eval_formatted[0]["prompt"])
    print("answer:", eval_formatted[0]["answer"])
    print("expected_action:", eval_formatted[0]["expected_action"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Loading the Base Model
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, device, torch):
    MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    next(model.parameters()).device
    return MODEL_NAME, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Intervention hook

    Perturbs the residual stream at a chosen layer / token position / strength.
    Same hook will be reused later for labeling SFT data and for live RL training trials.
    """)
    return


@app.cell
def _(device, torch):
    class ActivationIntervention:
        def __init__(self, layer_idx, token_pos, strength=1.0, mode="noise", direction=None, seed=None):
            self.layer_idx = layer_idx
            self.token_pos = token_pos
            self.strength = strength
            self.mode = mode
            self.direction = direction
            self.handle = None
            self.rng = torch.Generator(device=device)
            if seed is not None:
                self.rng.manual_seed(seed)

        def _hook_fn(self, module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            pos = self.token_pos if self.token_pos is not None else hidden_states.shape[1] - 1
            if pos >= hidden_states.shape[1]:
                return output

            target = hidden_states[:, pos, :]

            if self.mode == "noise":
                norm = target.norm(dim=-1, keepdim=True)
                noise = torch.randn(target.shape, generator=self.rng, device=target.device, dtype=target.dtype)
                perturbed = target + self.strength * norm * noise / noise.norm(dim=-1, keepdim=True)
            elif self.mode == "ablation":
                perturbed = target * (1 - self.strength)
            elif self.mode == "steer":
                perturbed = target + self.strength * self.direction.to(target.dtype).to(target.device)
            else:
                raise ValueError(f"unknown mode {self.mode}")

            hidden_states[:, pos, :] = perturbed
            return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

        def register(self, model):
            layer = model.model.layers[self.layer_idx]
            self.handle = layer.register_forward_hook(self._hook_fn)

        def remove(self):
            if self.handle is not None:
                self.handle.remove()
                self.handle = None

    return (ActivationIntervention,)


@app.cell
def _(device, torch):
    def generate(model, tokenizer, prompt, intervention=None, max_new_tokens=256):
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        input_ids = encoded["input_ids"]

        if intervention is not None:
            intervention.register(model)
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            if intervention is not None:
                intervention.remove()

        generated = output[0][input_ids.shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return (generate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using a real problem from `eval_formatted.json` rather than a made-up example.
    """)
    return


@app.cell
def _(eval_formatted, generate, model, tokenizer):
    sample = eval_formatted[0]
    baseline_output = generate(model, tokenizer, sample["prompt"])
    print(baseline_output)
    print("\nexpected answer:", sample["answer"])
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Single intervention test
    """)
    return


@app.cell
def _(model):
    print("model device:", next(model.parameters()).device)
    return


@app.cell
def _(ActivationIntervention, generate, model, sample, tokenizer):
    mid_layer = model.config.num_hidden_layers // 2

    intervention = ActivationIntervention(
        layer_idx=mid_layer, token_pos=None, strength=2.0, mode="noise", seed=0
    )
    intervened_output = generate(model, tokenizer, sample["prompt"], intervention=intervention)
    print(intervened_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Calibrated intervention config

    Determined from the calibration sweep (see calibration_sweep_layers6to24.csv):
    layer 8 / strength 1.0 gives ~60% flip rate with coherent (non-garbage) output.
    """)
    return


@app.cell
def _():
    CALIBRATED_CONFIG = {"layer_idx": 8, "strength": 1.0, "mode": "noise"}
    HELD_OUT_CONFIG = {"layer_idx": 22, "strength": 0.75, "mode": "noise"}  # reserved for later generalization tests
    CALIBRATED_CONFIG
    return (CALIBRATED_CONFIG,)


@app.cell
def _(
    ActivationIntervention,
    extract_final_answer,
    generate,
    model,
    tokenizer,
):
    def label_trial(problem, config, seed=None):
        iv = ActivationIntervention(
            layer_idx=config["layer_idx"],
            token_pos=None,
            strength=config["strength"],
            mode=config["mode"],
            seed=seed,
        )
        out = generate(model, tokenizer, problem["prompt"], intervention=iv)
        pred = extract_final_answer(out)
        is_harmful = (pred != problem["answer"])
        return {
            "problem_id": problem["problem_id"],
            "raw_output": out,
            "predicted": pred,
            "correct_answer": problem["answer"],
            "is_harmful": is_harmful,
            "expected_action": "RETRY" if is_harmful else "ANSWER",
        }

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Install training dependencies
    """)
    return


@app.cell
def _(subprocess, sys):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "trl", "peft", "accelerate"])
    return


@app.cell
def _(sft_train_ready):
    print(type(sft_train_ready[0]))
    print(sft_train_ready[0])
    return


@app.cell
def _():
    ## 13. Rebuilding SFT training data with real, calibrated intervention labels

    ## Previous should_retry labels weren't grounded in actual model behavior under
    ##intervention. Rebuilding using label_trial() so the RETRY signal is real.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. SFT training
    """)
    return


@app.cell
def _(MODEL_NAME, json):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    with open("sft_train_ready_v2.json") as f:
        corrected_sft_data = json.load(f)
    print(len(corrected_sft_data), "examples loaded from sft_train_ready_v2.json")

    sft_dataset_v2 = Dataset.from_list(corrected_sft_data)

    num_examples_v2 = len(sft_dataset_v2)
    effective_batch_size = 4 * 4
    steps_per_epoch_v2 = max(1, num_examples_v2 // effective_batch_size)
    total_steps_v2 = steps_per_epoch_v2 * 2
    warmup_steps_v2 = max(1, int(0.05 * total_steps_v2))

    sft_config_v2 = SFTConfig(
        output_dir="./sft_model_v2",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-5,
        warmup_steps=warmup_steps_v2,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    sft_trainer_v2 = SFTTrainer(
        model=MODEL_NAME,
        train_dataset=sft_dataset_v2,
        args=sft_config_v2,
    )
    return Dataset, sft_trainer_v2


@app.cell
def _(sft_trainer_v2):
    sft_trainer_v2.train()
    sft_trainer_v2.save_model("./sft_model_v2_final")
    sft_model_v2 = sft_trainer_v2.model
    sft_model_v2.eval()
    return (sft_model_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. SFT model sanity check
    ## 16. Does the corrected SFT model actually produce RETRY now?
    """)
    return


@app.cell
def _(
    ActivationIntervention,
    eval_formatted,
    generate,
    sft_model_v2,
    tokenizer,
):
    check_item = eval_formatted[3]

    clean_out = generate(sft_model_v2, tokenizer, check_item["prompt"])
    print("=== CLEAN (no intervention) ===")
    print(clean_out)

    iv_out = generate(sft_model_v2, tokenizer, check_item["prompt"],
                       intervention=ActivationIntervention(layer_idx=8, token_pos=None, strength=1.0, mode="noise", seed=2))
    print("\n=== INTERVENED ===")
    print(iv_out)
    return


@app.cell
def _(ActivationIntervention, generate, tokenizer):
    import re

    def extract_final_answer(text):
        if "[ANSWER]" in text:
            tail = text.split("[ANSWER]")[-1]
        else:
            tail = text
        nums = re.findall(r"-?\d+", tail)
        return int(nums[-1]) if nums else None

    def extract_action(text):
        if "RETRY" in text.upper():
            return "RETRY"
        elif "ANSWER" in text.upper():
            return "ANSWER"
        return None

    def run_behavior_eval1(model_to_test, model_label, trials, config, seed=0):
        results = []
        for item in trials:
            iv = ActivationIntervention(
                layer_idx=config["layer_idx"], token_pos=None,
                strength=config["strength"], mode=config["mode"], seed=seed,
            )
            out = generate(model_to_test, tokenizer, item["prompt"], intervention=iv)
            pred = extract_final_answer(out)
            action = extract_action(out)
            was_harmful = (pred != item["answer"])
            results.append({
                "model": model_label,
                "problem_id": item["problem_id"],
                "was_harmful": was_harmful,
                "action_taken": action,
                "correct_retry_decision": (action == "RETRY") == was_harmful,
            })
        return results

    return extract_action, extract_final_answer, run_behavior_eval1


@app.cell
def _(
    CALIBRATED_CONFIG,
    eval_formatted,
    model,
    run_behavior_eval1,
    sft_model_v2,
):
    import pandas as pd 
    eval_sample3 = eval_formatted[:20]

    unchanged_results3 = run_behavior_eval1(model, "unchanged", eval_sample3, CALIBRATED_CONFIG)
    sft_v2_results3 = run_behavior_eval1(sft_model_v2, "sft_v2", eval_sample3, CALIBRATED_CONFIG)

    behavior_df3 = pd.DataFrame(unchanged_results3 + sft_v2_results3)
    behavior_df3
    return behavior_df3, pd


@app.cell
def _(behavior_df3):
    print(behavior_df3.groupby("model")["action_taken"].apply(lambda x: x.value_counts(dropna=False)))
    return


@app.cell
def _(behavior_df3):
    retry_breakdown3 = behavior_df3.groupby(["model", "was_harmful"])["action_taken"].apply(lambda x: (x == "RETRY").mean())
    retry_breakdown3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. RL training setup (GRPO)

    Reward is computed live: apply the calibrated intervention (or leave clean) during
    rollout, then reward = 1[correct] - c*1[RETRY]. No pre-baked should_retry labels needed.
    """)
    return


@app.cell
def _(extract_action, extract_final_answer):
    RETRY_COST = 0.1

    def compute_reward(prompts, completions, problem_metadata, **kwargs):
        """
        prompts: list of prompt strings (unused directly, kept for GRPOTrainer's interface)
        completions: list of generated completion strings
        problem_metadata: list of dicts with 'answer' and whether intervention was applied + is_harmful
        """
        rewards = []
        for completion, meta in zip(completions, problem_metadata):
            pred = extract_final_answer(completion)
            action = extract_action(completion)

            if action == "RETRY":
                # Correctness after retry: assume a clean re-solve would get it right
                # (approximation — full re-generation on RETRY is expensive; see note below)
                final_correct = True
                reward = 1.0 - RETRY_COST
            else:
                final_correct = (pred == meta["answer"])
                reward = 1.0 if final_correct else 0.0

            rewards.append(reward)
        return rewards

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GRPO Trainer
    """)
    return


@app.cell
def _(rl_train_ready):
    print(rl_train_ready[0].keys() if isinstance(rl_train_ready[0], dict) else type(rl_train_ready[0]))
    print(rl_train_ready[0])
    return


@app.cell
def _(Dataset, rl_train_ready):
    rl_subset = [x for x in rl_train_ready if isinstance(x["answer"], int)][:300]
    print(len(rl_subset), "arithmetic examples selected for RL training")

    rl_dataset = Dataset.from_list(rl_subset)
    print(len(rl_dataset), "examples in rl_dataset")
    rl_dataset[0]
    return


@app.cell
def _(AutoModelForCausalLM, device, torch):
    rl_model = AutoModelForCausalLM.from_pretrained("./rl_model_final", torch_dtype=torch.bfloat16)
    rl_model = rl_model.to(device)
    rl_model.eval()
    print("loaded RL model from disk")
    return (rl_model,)


@app.cell
def _(ActivationIntervention, eval_formatted, generate, rl_model, tokenizer):
    check_item1 = eval_formatted[3]

    clean_out_rl = generate(rl_model, tokenizer, check_item1["prompt"])
    print("=== CLEAN ===")
    print(clean_out_rl)

    iv_out_rl = generate(rl_model, tokenizer, check_item1["prompt"],
                          intervention=ActivationIntervention(layer_idx=8, token_pos=None, strength=1.0, mode="noise", seed=2))
    print("\n=== INTERVENED ===")
    print(iv_out_rl)
    return


@app.cell
def _(
    CALIBRATED_CONFIG,
    eval_formatted,
    model,
    pd,
    rl_model,
    run_behavior_eval1,
    sft_model_v2,
):
    eval_sample_rl = eval_formatted[:20]

    unchanged_r = run_behavior_eval1(model, "unchanged", eval_sample_rl, CALIBRATED_CONFIG)
    sft_v2_r = run_behavior_eval1(sft_model_v2, "sft_v2", eval_sample_rl, CALIBRATED_CONFIG)
    rl_r = run_behavior_eval1(rl_model, "rl", eval_sample_rl, CALIBRATED_CONFIG)

    behavior_df_final = pd.DataFrame(unchanged_r + sft_v2_r + rl_r)
    print(behavior_df_final.groupby("model")["action_taken"].apply(lambda x: x.value_counts(dropna=False)))
    print()
    retry_breakdown_final = behavior_df_final.groupby(["model", "was_harmful"])["action_taken"].apply(lambda x: (x == "RETRY").mean())
    retry_breakdown_final
    return


app._unparsable_cell(
    r"""
    ''''from trl import GRPOTrainer, GRPOConfig

    grpo_config = GRPOConfig(
        output_dir="./rl_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-6,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        num_generations=4,  # rollouts per prompt — GRPO needs multiple samples to compute relative advantage
    )

    def reward_fn(completions, prompts, **kwargs):
        # kwargs will contain dataset columns like 'answer', 'problem_id' as lists
        answers = kwargs.get("answer", [None] * len(completions))
        rewards = []
        for completion, answer in zip(completions, answers):
            pred = extract_final_answer(completion)
            action = extract_action(completion)
            if action == "RETRY":
                rewards.append(1.0 - RETRY_COST)
            else:
                rewards.append(1.0 if pred == answer else 0.0)
        return rewards

    grpo_trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=grpo_config,
        train_dataset=rl_dataset,
        reward_funcs=reward_fn,
    )
    grpo_trainer.train()
    grpo_trainer.save_model("./rl_model_final")
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
