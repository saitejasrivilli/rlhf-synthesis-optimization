"""
LLM-based synthesis policy: Llama-2 7B + LoRA (r=8, alpha=16).
Generates synthesis conditions as structured JSON text.
PPO actor-critic: LLM backbone for policy, value head on final hidden state.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
### Optimize pharmaceutical synthesis conditions

Molecule: {molecule}
CAS: {cas}
Objective: maximize yield and selectivity, minimize hazard and steps

### Optimal conditions (JSON):
"""

_FALLBACK_CONDITIONS = {
    "temperature_celsius": 80.0,
    "time_hours": 4.0,
    "catalyst_loading_M": 0.05,
    "solvent_ratio_ml_mmol": 3.0,
}


class ValueHead(nn.Module):
    """Critic head for PPO — predicts state value from LLM hidden state."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


class LLMSynthesisPolicy(nn.Module):
    """
    Llama-2 7B with LoRA adapters for synthesis optimization.

    Only LoRA weights + value head are trained; base model is frozen.
    LoRA targets: q_proj, k_proj, v_proj, o_proj  (all attention projections).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            from peft import LoraConfig, TaskType, get_peft_model

            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.model = get_peft_model(base, lora_cfg)
            self.model.print_trainable_parameters()
            logger.info(f"Loaded {model_name} with LoRA r={lora_r}")
        except ImportError:
            raise ImportError("pip install peft transformers accelerate")

        hidden_size = self.model.config.hidden_size  # 3584 for Qwen2.5-7B, 4096 for Llama-2-7B

        # Place value head on the same device as the model's output layer
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            _last_device = list(self.model.hf_device_map.values())[-1]
            _vhead_dev = _last_device if isinstance(_last_device, (str, torch.device)) else "cuda"
        else:
            _vhead_dev = next(self.model.parameters()).device
        self.value_head = ValueHead(hidden_size).to(_vhead_dev)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def build_prompt(self, trajectory: Dict) -> str:
        molecule = trajectory.get("molecule", "Unknown")
        cas = trajectory.get("cas_number", "N/A")
        return PROMPT_TEMPLATE.format(molecule=molecule, cas=cas)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        queries: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthesis conditions for a batch of prompts.

        Returns (query_ids, response_ids) — shapes [B, Q] and [B, R].
        """
        enc = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        out = self.model.generate(**enc, **gen_kwargs)
        query_ids = enc["input_ids"]
        response_ids = out[:, query_ids.shape[1]:]
        return query_ids, response_ids

    # ------------------------------------------------------------------
    # Log-probs + values (used during PPO update)
    # ------------------------------------------------------------------

    def get_logprobs_and_values(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-token log-probs over the response and scalar values.

        Returns:
            response_logprobs: [B, R]
            values:            [B]
        """
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        attn_mask = (full_ids != self.tokenizer.pad_token_id).long()

        out = self.model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        # Shift by 1 for next-token prediction
        logits = out.logits[:, :-1, :]          # [B, T-1, V]
        labels = full_ids[:, 1:]                 # [B, T-1]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        q_len = query_ids.shape[1]
        response_lp = token_lp[:, q_len - 1:]   # [B, R]

        last_hidden = out.hidden_states[-1][:, -1, :].float()
        values = self.value_head(last_hidden)    # [B]

        return response_lp, values

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_conditions(self, response_ids: torch.Tensor) -> List[Dict]:
        """
        Parse generated tokens into synthesis condition dicts.
        Clips out-of-range values to chemically feasible bounds.
        """
        from src.utils.reaction_constraints import validate_conditions

        texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        conditions = []
        for text in texts:
            try:
                m = re.search(r"\{[^}]+\}", text, re.DOTALL)
                raw = json.loads(m.group()) if m else _FALLBACK_CONDITIONS.copy()
            except (json.JSONDecodeError, AttributeError):
                raw = _FALLBACK_CONDITIONS.copy()
            cond, _ = validate_conditions(raw)
            conditions.append(cond)
        return conditions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")
        logger.info(f"Policy saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Load a saved LoRA checkpoint. LoRA weights remain trainable."""
        policy = cls(model_name=base_model)

        # Load saved adapter weights into the already-attached LoRA modules
        policy.model.load_adapter(path, adapter_name="default", is_trainable=True)

        vh_path = f"{path}/value_head.pt"
        if Path(vh_path).exists():
            policy.value_head.load_state_dict(
                torch.load(vh_path, map_location="cpu")
            )
        logger.info(f"Policy loaded from {path}")
        return policy
