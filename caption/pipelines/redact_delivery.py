#!/usr/bin/env python3
"""Build an independent, privacy-redacted caption delivery copy.

The source is read-only.  JSONL lines are parsed only for validation and
top-level ``ok`` filtering, then sanitized in-place so nested ``content_raw``,
``parsed``, ``pass1`` and ``pass2`` strings are covered without retaining huge
duplicate Python objects.  Audit output contains counts and location hashes,
never matched values or source excerpts.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

MASK = "XXX"
EMAIL = re.compile(r"(?i)(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9_-])")
API = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{20,}|pk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})")
BEARER = re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{16,}")
CRED = re.compile(r"(?ix)(\b(?:password|passwd|passcode|api[ _-]?key|secret|access[ _-]?token|refresh[ _-]?token|bearer)\b\s*[:=]\s*)([^\s,;}'\"\]]{3,})")
CN_CRED = re.compile(r"((?:密码|密碼|验证码|驗證碼|口令|门禁(?:码|号|號)?|門禁(?:碼|號)?|PIN|OTP)(?:\s*(?:是|为|係|為))?\s*[:：=#]?\s*)([A-Za-z0-9][A-Za-z0-9_.-]{3,})", re.I)
PHONE = re.compile(r"(?<![A-Za-z0-9])(?:\+?(?:86|852)|00?852|0[1-9][0-9]{2})[ ()-]*[0-9]{3,4}[ -][0-9]{3,4}(?![A-Za-z0-9])")
MOBILE = re.compile(r"(?<![A-Za-z0-9])1[3-9][0-9]{9}(?![A-Za-z0-9])")
HKPHONE = re.compile(r"(?<![A-Za-z0-9])[2-9][0-9]{3}[ -][0-9]{4}(?![A-Za-z0-9])")
CARD = re.compile(r"(?<![A-Za-z0-9])(?:[0-9]{4}[ -]?){3,4}[0-9]{1,4}(?![A-Za-z0-9])")
URL = re.compile(r"(?i)\bhttps?://[^\s\"<>]+")
LOCAL = re.compile(r"/(?:Users|home)/[^\s\"'<>\\]+")
MPLATE = re.compile(r"(?<![A-Za-z0-9])[\u4e00-\u9fff][A-Z][A-Z0-9]{5}(?![A-Za-z0-9])")
PARTICIPANT = re.compile(r"(?<![A-Za-z0-9])A[0-9]+_[A-Z][A-Z0-9_-]+(?![A-Za-z0-9])")
ANON = re.compile(r"(?<![A-Za-z0-9])User[ _-]?[A-Z0-9]+(?![A-Za-z0-9])", re.I)
JSON_NAME = re.compile(r'(?i)("(?:speaker|name|participant|participant_id|author|sender|recipient|contact_name|account_name)"\s*:\s*)"(?:\\.|[^"\\])*"')
PAYMENT = re.compile(r"((?:商户|收款方|付款方|支付金额|付款金额|交易金额|金额)\s*[:：]?\s*)([^\s,;}'\"\]]{1,32})")
ADDRESS = re.compile(r"((?:地址|住址|现居住地址|收货地址|邮编|邮政编码)\s*[:：]?\s*)([^,。；\]}\"']{2,80})")
CURRENCY = re.compile(r"(?:HK\$|RMB|[$¥￥])\s*[0-9]+(?:\.[0-9]{1,2})?(?![A-Za-z0-9])", re.I)

class Audit:
    def __init__(self):
        self.counts = Counter(); self.files = defaultdict(set); self.locs = defaultdict(set)
    def add(self, cat, n, rel, line):
        if n:
            self.counts[cat] += n; self.files[cat].add(rel)
            self.locs[cat].add(hashlib.sha256(f"{rel}:{line}".encode()).hexdigest()[:16])
    def sub(self, cat, pat, text, rel, line):
        ms = list(pat.finditer(text)); self.add(cat, len(ms), rel, line)
        return pat.sub(MASK, text)
    def pair(self, cat, pat, text, rel, line):
        def r(m): self.add(cat, 1, rel, line); return m.group(1) + MASK
        return pat.sub(r, text)
    def text(self, text, rel, line):
        text = self.pair("credential_value", CRED, text, rel, line)
        text = self.pair("credential_value", CN_CRED, text, rel, line)
        for cat, pat in [("email",EMAIL),("api_token",API),("bearer_token",BEARER),("phone_formatted",PHONE),("phone_mobile",MOBILE),("phone_plain_hk",HKPHONE),("credit_card",CARD),("url",URL),("local_path",LOCAL),("license_plate",MPLATE),("participant_id",PARTICIPANT),("anonymized_user",ANON),("payment_amount",CURRENCY)]:
            text = self.sub(cat, pat, text, rel, line)
        text = self.pair("payment_merchant", PAYMENT, text, rel, line)
        text = self.pair("address_value", ADDRESS, text, rel, line)
        return text

SENSITIVE_NAME_KEYS = {
    "speaker", "name", "participant", "participant_id", "author", "sender",
    "recipient", "contact_name", "account_name",
}
RAW_DUPLICATE_KEYS = {"content_raw"}

SPEAKER_PREFIX_NAME = re.compile(
    r"(?<![A-Za-z])([A-Z][A-Za-z.'-]{1,30})(?=\s*[:：])"
)
NON_PERSON_LABELS = {
    "action", "ai", "ambient", "assistant", "caption", "densecaption",
    "details", "environment", "hkt", "model", "ocr", "scene", "screen",
    "self", "speech", "summary", "system", "text", "transcript", "user",
    "other", "others", "unknown", "male", "female", "man", "woman",
    "person", "speaker", "narrator", "student", "teacher", "me",
    "companion", "chinese", "friend", "colleague", "roommate",
    "passenger", "driver", "instructor", "lecturer", "caller",
    "gemini", "chatgpt", "claude", "deepseek", "whatsapp", "wechat",
    "telegram", "instagram", "messenger", "facebook", "youtube", "lite",
}
# One frequent source-only speaker role appeared in 530 clips across 80
# recordings. Keep only its hash here so the source value itself is not copied
# into code or logs.
NON_PERSON_LABEL_HASHES = {"5777ff8f6f1f"}
SPEAKER_CONTEXT_KEYS = {
    "speech", "speech_summary", "transcript", "dialogue", "conversation",
    "speaker_turns",
}
ATTRIBUTED_NAME = re.compile(
    r"(?<![A-Za-z])([A-Z][a-z][A-Za-z.'-]{0,29})(?=\s*(?:说|说道|表示|问|回答|告诉|喊|提到|建议|解释))"
)
ENGLISH_ATTRIBUTED_NAME = re.compile(
    r"(?<![A-Za-z])([A-Z][a-z][A-Za-z.'-]{0,29})(?=\s+(?i:says|said|asks|asked|replies|replied|tells|told|speaks|spoke)\b)",
)


def is_person_name_candidate(value):
    return (
        bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,79}", value))
        and value.casefold() not in NON_PERSON_LABELS
        and hashlib.sha256(value.casefold().encode()).hexdigest()[:12]
        not in NON_PERSON_LABEL_HASHES
    )


def collect_person_names(obj, key=None):
    """Collect explicit person-name signals within one caption record."""
    names = set()
    if isinstance(obj, str):
        value = obj.strip()
        if (
            key
            and str(key).lower() in SENSITIVE_NAME_KEYS
            and is_person_name_candidate(value)
        ):
            names.add(value)
        if key and str(key).lower() in SPEAKER_CONTEXT_KEYS:
            for match in SPEAKER_PREFIX_NAME.finditer(obj):
                value = match.group(1)
                if value.casefold() not in NON_PERSON_LABELS:
                    names.add(value)
        for pattern in (ATTRIBUTED_NAME, ENGLISH_ATTRIBUTED_NAME):
            for match in pattern.finditer(obj):
                value = match.group(1)
                if is_person_name_candidate(value):
                    names.add(value)
    elif isinstance(obj, list):
        for value in obj:
            names.update(collect_person_names(value, key))
    elif isinstance(obj, dict):
        for child_key, value in obj.items():
            names.update(collect_person_names(value, child_key))
    return names


def collect_corpus_person_names(obj, key=None):
    """Collect only strong signals safe to propagate across recordings."""
    names = set()
    if isinstance(obj, str):
        value = obj.strip()
        if key and str(key).lower() in SENSITIVE_NAME_KEYS and is_person_name_candidate(value):
            names.add(value)
        for pattern in (ATTRIBUTED_NAME, ENGLISH_ATTRIBUTED_NAME):
            for match in pattern.finditer(obj):
                value = match.group(1)
                if is_person_name_candidate(value):
                    names.add(value)
    elif isinstance(obj, list):
        for value in obj:
            names.update(collect_corpus_person_names(value, key))
    elif isinstance(obj, dict):
        for child_key, value in obj.items():
            names.update(collect_corpus_person_names(value, child_key))
    return names


@lru_cache(maxsize=128)
def person_names_pattern(names):
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(rf"(?<![A-Za-z])(?:{alternatives})(?![A-Za-z])", re.I)


def redact_person_names(text, names, rel, line, audit):
    names_key = tuple(sorted(set(names), key=lambda name: (-len(name), name.casefold())))
    pattern = person_names_pattern(names_key)
    if pattern is None:
        return text
    return audit.sub("person_name_in_text", pattern, text, rel, line)

def sanitize_obj(obj, rel, line, a, key=None, person_names=None):
    """Recursively sanitize parsed JSON values without touching JSON syntax."""
    if person_names is None:
        person_names = collect_person_names(obj, key)
    if isinstance(obj, str):
        if key and str(key).lower() in RAW_DUPLICATE_KEYS:
            a.add("raw_duplicate_removed", 1, rel, line)
            return MASK
        if key and str(key).lower() in SENSITIVE_NAME_KEYS:
            a.add("person_name", 1, rel, line)
            return MASK
        return redact_person_names(a.text(obj, rel, line), person_names, rel, line, a)
    if isinstance(obj, list):
        return [sanitize_obj(v, rel, line, a, person_names=person_names) for v in obj]
    if isinstance(obj, dict):
        return {
            k: sanitize_obj(v, rel, line, a, k, person_names)
            for k, v in obj.items()
        }
    return obj

def jsonl(src, dst, rel, a, corpus_person_names=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept_ids = set(); person_names_by_clip = {}; kept=failed=bad=segments=0; lines=0; models=Counter()
    with src.open(encoding="utf-8", errors="replace") as fi, dst.open("w",encoding="utf-8",newline="") as fo:
        for no,line in enumerate(fi,1):
            lines = no
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Never emit a transformed raw line: that was the source of the
                # previous malformed-output bug. Invalid input is excluded.
                bad += 1
                continue
            if not isinstance(obj, dict) or obj.get("ok") is not True:
                failed += 1
                continue
            clip_id = obj.get("clip_id")
            if isinstance(clip_id, int):
                kept_ids.add(clip_id)
            p=obj.get("merged") or obj.get("parsed")
            if isinstance(p,dict) and isinstance(p.get("segments"),list):
                segments += len(p["segments"])
            model = obj.get("model")
            if isinstance(model, str) and model:
                models[model] += 1
            person_names = set(corpus_person_names or ())
            person_names.update(collect_person_names(obj))
            if isinstance(clip_id, int):
                person_names_by_clip[clip_id] = person_names
            clean = sanitize_obj(obj, rel, no, a, person_names=person_names)
            fo.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    return {"lines":lines,"kept":kept,"filtered_failed":failed,"parse_errors":bad,"segments":segments,"models":dict(models),"kept_ids":kept_ids,"person_names_by_clip":person_names_by_clip}

def txt(src,dst,rel,a,kept_ids=None,person_names=None,person_names_by_clip=None):
    dst.parent.mkdir(parents=True,exist_ok=True); lines=empty=0; block=[]
    header_re = re.compile(r"^===== clip\s+(\d+)\b")
    def flush(fo):
        nonlocal empty
        if not block:return
        m = header_re.match(block[0][1])
        if m and kept_ids is not None and int(m.group(1)) not in kept_ids:
            empty += 1
            return
        if m and not any(s.strip() for _,s in block[1:]): empty+=1; return
        block_person_names = set(person_names or ())
        if m and person_names_by_clip is not None:
            block_person_names.update(person_names_by_clip.get(int(m.group(1)), ()))
        for no,s in block:
            clean = a.text(s.rstrip(), rel, no)
            clean = redact_person_names(clean, block_person_names, rel, no, a)
            fo.write(clean + "\n")
    with src.open(encoding="utf-8",errors="replace") as fi,dst.open("w",encoding="utf-8",newline="") as fo:
        for no,line in enumerate(fi,1):
            lines+=1
            if line.startswith("===== clip"): flush(fo); block=[(no,line)]
            else:block.append((no,line))
        flush(fo)
    normalized = "\n".join(
        line.rstrip() for line in dst.read_text(encoding="utf-8").splitlines()
    ).rstrip()
    dst.write_text(normalized + ("\n" if normalized else ""), encoding="utf-8")
    return {"lines":lines,"filtered_empty":empty}


def recompute_index(index, stats_by_recording, redaction_meta):
    for recording, stats in stats_by_recording.items():
        entry = index.get(recording)
        if not isinstance(entry, dict):
            continue
        kept = int(stats["kept"])
        source_planned_target = entry.get("target_clips")
        selected_candidates = int(stats["lines"])
        entry["source_planned_target_clips"] = source_planned_target
        entry["selected_candidate_clips"] = selected_candidates
        # For the published snapshot, coverage means successful rows among the
        # candidate rows actually pulled from HF. The source planning target is
        # retained separately because it can become stale as HF gains outputs.
        entry["target_clips"] = selected_candidates
        entry["usable_clips"] = kept
        entry["coverage"] = round(kept / selected_candidates, 3) if selected_candidates else None
        entry["models"] = dict(stats["models"])
        entry["segments"] = int(stats["segments"])

    recordings = [
        value
        for key, value in index.items()
        if not key.startswith("_") and isinstance(value, dict) and "usable_clips" in value
    ]
    first_100h = [value for value in recordings if value.get("in_first_100h")]
    first_target = sum(
        int(value["selected_candidate_clips"])
        for value in first_100h
        if isinstance(value.get("selected_candidate_clips"), (int, float))
    )
    first_source_planned_target = sum(
        int(value["source_planned_target_clips"])
        for value in first_100h
        if isinstance(value.get("source_planned_target_clips"), (int, float))
    )
    first_usable = sum(int(value["usable_clips"]) for value in first_100h)
    selected_total = sum(int(value["selected_candidate_clips"]) for value in recordings)
    usable_total = sum(int(value["usable_clips"]) for value in recordings)
    previous_summary = index.get("_summary")
    summary = dict(previous_summary) if isinstance(previous_summary, dict) else {}
    summary.update({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "usable_clips_total": usable_total,
        "selected_candidate_clips_total": selected_total,
        "selected_candidate_coverage": round(usable_total / selected_total, 3) if selected_total else None,
        "filtered_or_unparseable_clips_total": selected_total - usable_total,
        "first_100h_usable": first_usable,
        "first_100h_target": first_target,
        "first_100h_coverage": round(first_usable / first_target, 3) if first_target else None,
        "source_planned_target_clips": first_source_planned_target,
        "recordings_with_output": sum(int(value["usable_clips"]) > 0 for value in recordings),
        "note": "只含顶层 ok=true 的 clip；coverage 按本次 HF 已选中候选计算；旧计划目标保存在 source_planned_target_clips；失败或无法解析记录已过滤；敏感文本已替换为 XXX。clip_index 可能稀疏。",
    })
    index["_summary"] = summary
    index["_redaction"] = redaction_meta
    return index

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",type=Path,required=True); ap.add_argument("--dst",type=Path,required=True); ap.add_argument("--report",type=Path,required=True); args=ap.parse_args()
    src=args.src.resolve(); dst=args.dst.resolve(); report=args.report.resolve()
    if not src.is_dir(): raise SystemExit(f"missing source: {src}")
    if dst.exists(): raise SystemExit(f"refusing overwrite: {dst}")
    dst.mkdir(parents=True); a=Audit(); js={"files":0,"lines":0,"kept":0,"filtered_failed":0,"parse_errors":0,"segments":0}; ts={"files":0,"lines":0,"filtered_empty":0}; copied=0
    kept_by_recording = {}; names_by_recording = {}; stats_by_recording = {}
    all_files = sorted(p for p in src.rglob("*") if p.is_file())
    jsonl_source_files = [p for p in all_files if p.name == "captions.jsonl"]
    # Strong signals from explicit name fields or spoken attribution remain
    # sensitive when repeated later in OCR/text_visible content.
    corpus_person_names = set()
    for f in jsonl_source_files:
        with f.open(encoding="utf-8", errors="replace") as fi:
            for line in fi:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("ok") is True:
                    corpus_person_names.update(collect_corpus_person_names(obj))
    # Process JSONL first so TXT blocks can be synchronized with successful IDs.
    for f in jsonl_source_files:
        rel=f.relative_to(src); rs=rel.as_posix(); out=dst/rel
        s=jsonl(f,out,rs,a,corpus_person_names)
        kept_by_recording[f.parent] = s.pop("kept_ids", set())
        names_by_recording[f.parent] = s.pop("person_names_by_clip", {})
        stats_by_recording[rel.parts[0]] = dict(s)
        js["files"] += 1
        for k in js:
            js[k] += int(s.get(k,0))
    for f in all_files:
        rel=f.relative_to(src); rs=rel.as_posix(); out=dst/rel
        if rs=="index.json": continue
        if f.name=="captions.jsonl":
            continue
        elif f.name=="captions.txt":
            s=txt(f,out,rs,a,kept_by_recording.get(f.parent),corpus_person_names,names_by_recording.get(f.parent)); ts["files"]+=1; ts["lines"]+=s["lines"]; ts["filtered_empty"]+=s["filtered_empty"]
        else: out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,out); copied+=1
    meta={"version":3,"mask":MASK,"source_name":src.name,"corpus_person_name_signals":len(corpus_person_names),"match_counts":dict(sorted(a.counts.items())),"files_with_matches":{k:len(v) for k,v in sorted(a.files.items())},"kept_success_clips":js["kept"],"filtered_failed_clips":js["filtered_failed"],"filtered_empty_txt_blocks":ts["filtered_empty"]}
    idx=json.loads((src/"index.json").read_text(encoding="utf-8")); recompute_index(idx,stats_by_recording,meta); (dst/"index.json").write_text(json.dumps(idx,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rep={"version":3,"source_name":src.name,"destination_name":dst.name,"mask":MASK,"corpus_person_name_signals":len(corpus_person_names),"input":{"jsonl":js,"txt":ts,"copied_non_caption_files":copied},"match_counts":dict(sorted(a.counts.items())),"files_with_matches":{k:len(v) for k,v in sorted(a.files.items())},"location_hashes":{k:sorted(v) for k,v in sorted(a.locs.items())},"verification":{"jsonl_parse_errors":js["parse_errors"]}}
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"destination":str(dst),"report":str(report),"input":rep["input"],"match_counts":rep["match_counts"]},ensure_ascii=False))
if __name__=="__main__": main()
