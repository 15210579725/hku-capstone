"""src/extract.py — 用 RepE rep-reading pipeline 抽 6 情绪「全层 LAT 方向」。

逐字复刻 examples/primary_emotions/emotion_concept.ipynb 的 reading 流程：
  - hidden_layers = list(range(-1, -num_hidden_layers, -1))  # 全 transformer 层（负索引口径）
  - rep_token=-1, n_difference=1, direction_method='pca'
  - 每情绪：get_directions(train) → rep_reader（.directions[layer] (n_comp,hidden)、
    .direction_signs[layer] (n_comp,)、.H_train_means[layer] (1,hidden)）。
  - 在 test 刺激上跑 pipeline 得 H_tests（供 select_layer 算各层分类准确率）。

严格复用：
  - stimuli.build_concept_dataset / _zh_chat_prefix —— 中文成对刺激 + Qwen chat 模板。
  - repe_compat.apply_compat —— device-agnostic 补丁 + 注册 rep-reading pipeline（幂等）。
本文件只做「抽取」；选层在 select_layer.py，落盘契约在 io_artifact.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import repe_compat, stimuli  # noqa: E402

EMOTIONS = config.EMOTIONS


def load_model_and_tokenizer(model_name_or_path: str, dtype=torch.float16):
    """加载抽向量用模型（不量化，fp16）+ tokenizer（left padding，复刻 notebook 口径）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", legacy=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_pipeline(model, tokenizer):
    """注册并构造 RepE rep-reading pipeline（device-agnostic 补丁已在内部打好）。"""
    repe_compat.apply_compat(register_pipeline=True)
    from transformers import pipeline

    return pipeline("rep-reading", model=model, tokenizer=tokenizer)


def all_hidden_layers(model) -> list:
    """全 transformer 层的负索引列表：[-1, -2, ..., -(num_hidden_layers-1)]（同 notebook）。"""
    return list(range(-1, -model.config.num_hidden_layers, -1))


def extract_all(
    model,
    tokenizer,
    data_dir: str,
    *,
    lang: str = "zh",
    use_chat_template: bool = True,
    cap: int = 200,
    rep_token: int = -1,
    n_difference: int = 1,
    direction_method: str = "pca",
    batch_size: int = 16,
    seed: int = 0,
):
    """对 6 情绪逐个抽全层方向 + 在 test 集拿投影分。

    Returns:
        rep_readers: {emotion: RepReader}（含 directions / direction_signs / H_train_means）
        H_tests:     {emotion: list[dict{layer: np.ndarray(1,)}]}（每个 test 样本一项）
        hidden_layers: list[int]
        meta: dict（模型/模板/层数等，写进 artifact）
    """
    rep_pipe = build_pipeline(model, tokenizer)
    hidden_layers = all_hidden_layers(model)

    dataset = stimuli.build_concept_dataset(
        data_dir,
        tokenizer=tokenizer,
        lang=lang,
        use_chat_template=use_chat_template,
        cap=cap,
        seed=seed,
    )

    rep_readers = {}
    H_tests = {}
    for emo in EMOTIONS:
        train = dataset[emo]["train"]
        test = dataset[emo]["test"]

        rep_reader = rep_pipe.get_directions(
            train["data"],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            n_difference=n_difference,
            train_labels=train["labels"],
            direction_method=direction_method,
            batch_size=batch_size,
        )
        H_test = rep_pipe(
            test["data"],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            rep_reader=rep_reader,
            batch_size=batch_size,
        )
        rep_readers[emo] = rep_reader
        H_tests[emo] = H_test
        print(f"[extract] {emo}: directions over {len(hidden_layers)} layers, "
              f"{len(H_test)} test samples", flush=True)

    sig = stimuli.chat_template_signature(tokenizer, lang=lang) if use_chat_template else stimuli.EN_TEMPLATE
    meta = {
        "model_name": getattr(model.config, "_name_or_path", str(model.config.model_type)),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "rep_token": rep_token,
        "stimulus_lang": lang,
        "use_chat_template": use_chat_template,
        "chat_template_signature": sig,
        "patch_info": repe_compat.patched_signatures(),
        "hidden_layers": hidden_layers,
    }
    return rep_readers, H_tests, hidden_layers, meta


def reader_directions_to_dicts(rep_readers, hidden_layers):
    """把 RepReader 拆成 io_artifact.save_artifact 需要的三个嵌套 dict。

    directions[emo][layer]      = PCA comp0 (hidden,)
    direction_signs[emo][layer] = int(±1)
    H_train_means[emo][layer]   = (1, hidden)  recenter 复现用
    """
    directions, signs, means = {}, {}, {}
    for emo, rr in rep_readers.items():
        directions[emo], signs[emo], means[emo] = {}, {}, {}
        for layer in hidden_layers:
            comp0 = np.asarray(rr.directions[layer][0], dtype=np.float32).reshape(-1)
            directions[emo][int(layer)] = comp0
            s = rr.direction_signs[layer]
            s0 = float(np.asarray(s).reshape(-1)[0])
            signs[emo][int(layer)] = int(np.sign(s0) or 1)
            means[emo][int(layer)] = np.asarray(
                rr.H_train_means[layer], dtype=np.float32
            ).reshape(1, -1)
    return directions, signs, means
