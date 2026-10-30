"""Run the exported checkpoint in stock transformers (Qwen3_5ForCausalLM, bf16) and check its
greedy predictions against the reference trace, teacher-forced."""

import json
import sys
import time
from pathlib import Path

import numpy as np


def hf_check(model_dir: str, prg: bytes, toks: list[int], pc=None, verbose=True):
    import torch
    from transformers import Qwen3_5ForCausalLM

    from nesformer import core
    from nesformer import spec as S
    from nesformer.cpu import Cpu
    from nesformer.run import injected_positions

    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    meta = json.loads((Path(model_dir) / "nesformer.json").read_text())
    rows = core.state_rows(Cpu.boot(prg, pc))
    inj = injected_positions(toks)
    bad, checked = [], 0
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[S.PAD]]), use_cache=True)
        cache = out.past_key_values
        for name, (L, j, nvh) in meta["heads"].items():
            st = cache.get_recurrent_state(L, state_idx=0) if hasattr(
                cache, "get_recurrent_state") else cache.layers[L].recurrent_states[0]
            for x in range(nvh):
                Sx = st[0, 2 * j + x]
                Sx.zero_()
                for r, vec in rows[name][x].items():
                    Sx[r] = torch.tensor(vec, dtype=Sx.dtype)
        for i, t in enumerate(toks[:-1]):
            o = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            pred = int(o.logits[0, -1].float().argmax())
            if (i + 1) in inj:
                continue
            checked += 1
            if pred != toks[i + 1]:
                bad.append((i, S.name(t), S.name(toks[i + 1]),
                            S.name(pred) if pred < S.VOCAB else pred))
            if verbose and (i + 1) % 200 == 0:
                print(f"  {i + 1} tokens, {len(bad)} mismatches, {time.time() - t0:.0f}s",
                      flush=True)
    return checked, bad


class HFEngine:
    """The CLI's --engine hf: step(token) through stock transformers' Qwen3_5ForCausalLM (bf16),
    with the snapshot written into its DeltaNet cache. Slow on a CPU (~10 tokens/s)."""

    def __init__(self, model_dir: str):
        import torch
        from transformers import Qwen3_5ForCausalLM

        self.torch = torch
        self.model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
        self.meta = json.loads((Path(model_dir) / "nesformer.json").read_text())
        self.cache = None
        self.steps = 0

    def load_state(self, rows: dict) -> None:
        torch = self.torch
        from nesformer import spec as S
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([[S.PAD]]), use_cache=True)
            cache = out.past_key_values
            for name, (L, j, nvh) in self.meta["heads"].items():
                st = cache.get_recurrent_state(L, state_idx=0) if hasattr(
                    cache, "get_recurrent_state") else cache.layers[L].recurrent_states[0]
                for x in range(nvh):
                    Sx = st[0, 2 * j + x]
                    Sx.zero_()
                    for r, vec in rows[name][x].items():
                        Sx[r] = torch.tensor(vec, dtype=Sx.dtype)
        self.cache = cache

    def step(self, t: int) -> int:
        torch = self.torch
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([[t]]), past_key_values=self.cache,
                           use_cache=True)
        self.cache = o.past_key_values
        self.steps += 1
        return int(o.logits[0, -1].float().argmax())
