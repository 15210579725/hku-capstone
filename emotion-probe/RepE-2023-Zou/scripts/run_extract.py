"""scripts/run_extract.py — GPU 抽取 6 情绪全层方向 + 选层 → emotion_vectors.pt。

在 AutoDL qwen 环境跑：
  conda activate qwen && python scripts/run_extract.py \
      --model /root/autodl-tmp/Qwen2.5-3B-Instruct \
      --data_dir data/emotions_zh \
      --out_artifact outputs/directions/emotion_vectors.pt \
      --out_select outputs/select_layer
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import extract, select_layer, io_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--data_dir", default=os.path.join(config.PROJECT_ROOT, "data", "emotions_zh"))
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_artifact", default=config.AGENT_B_DIRECTIONS)
    ap.add_argument("--out_select", default=os.path.join(config.PROJECT_ROOT, "outputs", "select_layer"))
    args = ap.parse_args()

    print(f"[run_extract] loading model {args.model} ...", flush=True)
    model, tokenizer = extract.load_model_and_tokenizer(args.model)

    rep_readers, H_tests, hidden_layers, meta = extract.extract_all(
        model, tokenizer, args.data_dir,
        lang=args.lang, use_chat_template=True, cap=args.cap,
        batch_size=args.batch_size,
    )

    sel = select_layer.run_select(rep_readers, H_tests, hidden_layers, args.out_select)

    directions, signs, means = extract.reader_directions_to_dicts(rep_readers, hidden_layers)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_artifact)), exist_ok=True)
    payload = io_artifact.save_artifact(
        args.out_artifact,
        model_name=meta["model_name"],
        num_hidden_layers=meta["num_hidden_layers"],
        hidden_size=meta["hidden_size"],
        stimulus_lang=meta["stimulus_lang"],
        use_chat_template=meta["use_chat_template"],
        chat_template_signature=meta["chat_template_signature"],
        patch_info=meta["patch_info"],
        emotions=config.EMOTIONS,
        layers=hidden_layers,
        directions=directions,
        direction_signs=signs,
        best_layer=sel["best_layer"],
        best_layer_acc=sel["best_layer_acc"],
        H_train_means=means,
        rep_token=meta["rep_token"],
    )
    manifest_path = os.path.join(os.path.dirname(args.out_artifact), "manifest.json")
    io_artifact.write_manifest(manifest_path, payload)
    print(f"[run_extract] saved artifact -> {args.out_artifact}", flush=True)
    print(f"[run_extract] best_layer={sel['best_layer']}", flush=True)
    print(f"[run_extract] best_acc={ {e: round(v,3) for e,v in sel['best_layer_acc'].items()} }", flush=True)


if __name__ == "__main__":
    main()
