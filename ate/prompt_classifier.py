"""Predict the PIE edit type from the (source, target) prompt pair only.

Two predictors: a zero-cost keyword/word-diff rule, and an optional small instruct LLM (default Qwen2.5-1.5B).
At test time neither sees PIE's category label, the edit instruction, or the [bracket] markers.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .pie import CATEGORY_NAMES, clean_prompt

COLORS = set("red orange yellow green blue purple pink brown black white gray grey golden silver beige "
             "violet cyan magenta teal turquoise navy maroon crimson".split())
MATERIALS = set("wooden wood metal metallic glass plastic stone marble paper ceramic leather steel iron "
                "bronze copper clay rubber woolen wool knitted crystal gold".split())
STYLE_WORDS = set("painting drawing sketch watercolor cartoon anime illustration style oil pixel lego "
                  "origami pencil impressionist cubist ukiyo-e mosaic sculpture 3d render vangogh monet".split())
POSE_WORDS = set("sitting standing running jumping lying laying flying walking dancing sleeping smiling "
                 "crying laughing raising opening closing holding swimming kneeling".split())
BACKGROUND_WORDS = set("background beach forest desert snow city street sky mountain ocean sea space "
                       "night sunset room field garden".split())


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", clean_prompt(text).lower())


def keyword_category(source: str, target: str) -> int:
    src, tgt = _words(source), _words(target)
    s_set, t_set = set(src), set(tgt)
    added, removed = t_set - s_set, s_set - t_set
    if added & STYLE_WORDS:
        return 9
    if added & COLORS:
        return 6
    if added & MATERIALS:
        return 7
    if added & POSE_WORDS or removed & POSE_WORDS:
        return 5
    if added & BACKGROUND_WORDS and ("background" in t_set or "in" in t_set or "on" in t_set):
        return 8
    if added and not removed:
        return 2
    if removed and not added:
        return 3
    if len(added) == 1 and len(removed) == 1:
        return 1
    return 4 if added else 0


_SYSTEM = (
    "You classify image-editing requests. Given a SOURCE caption and a TARGET caption, answer with the single "
    "digit of the edit type:\n" + "\n".join(f"{k}: {v}" for k, v in CATEGORY_NAMES.items()) +
    "\nAnswer with the digit only."
)
_FEWSHOT: Sequence[Tuple[str, str, int]] = (
    ("a cat sitting on a wooden chair", "a dog sitting on a wooden chair", 1),
    ("a vase on a table", "a vase with flowers on a table", 2),
    ("a man with a hat standing in a park", "a man standing in a park", 3),
    ("a red car parked on the street", "a blue car parked on the street", 6),
    ("a dog running on the grass", "a dog sitting on the grass", 5),
    ("a wooden table", "a glass table", 7),
    ("a horse in a field", "a horse on the beach", 8),
    ("a photo of a lighthouse", "a watercolor painting of a lighthouse", 9),
)


class LLMClassifier:
    def __init__(self, model_name: str, device: Optional[str] = None, max_new_tokens: int = 4) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def _messages(self, source: str, target: str):
        msgs = [{"role": "system", "content": _SYSTEM}]
        for s, t, y in _FEWSHOT:
            msgs.append({"role": "user", "content": f"SOURCE: {s}\nTARGET: {t}"})
            msgs.append({"role": "assistant", "content": str(y)})
        msgs.append({"role": "user", "content": f"SOURCE: {clean_prompt(source)}\nTARGET: {clean_prompt(target)}"})
        return msgs

    def predict(self, source: str, target: str) -> Optional[int]:
        import torch

        text = self.tok.apply_chat_template(self._messages(source, target), tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        answer = self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        match = re.search(r"\d", answer)
        return int(match.group(0)) if match else None


def classify_all(items, use_llm: bool, model_name: str, max_new_tokens: int = 4) -> List[Dict]:
    llm = LLMClassifier(model_name, max_new_tokens=max_new_tokens) if use_llm else None
    rows = []
    for it in items:
        kw = keyword_category(it.source_prompt, it.target_prompt)
        pred = llm.predict(it.source_prompt, it.target_prompt) if llm else None
        rows.append({
            "sample_id": it.sample_id,
            "true_category": it.category,
            "keyword_category": kw,
            "llm_category": pred if pred is not None else kw,
            "llm_parsed": pred is not None,
        })
    return rows
