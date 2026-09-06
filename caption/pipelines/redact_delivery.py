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
from pathlib import Path

MASK = "XXX"
EMAIL = re.compile(r"(?i)(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
API = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{20,}|pk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})")
BEARER = re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{16,}")
CRED = re.compile(r"(?ix)(\b(?:password|passwd|passcode|api[ _-]?key|secret|access[ _-]?token|refresh[ _-]?token|bearer)\b\s*[:=]\s*)([^\s,;}'\"\]]{3,})")
CN_CRED = re.compile(r"((?:密码|密碼|验证码|驗證碼|口令|门禁(?:码|号|號)?|門禁(?:碼|號)?|PIN|OTP)(?:\s*(?:是|为|係|為))?\s*[:：=#]?\s*)([A-Za-z0-9][A-Za-z0-9_.-]{3,})", re.I)
PHONE = re.compile(r"(?<![A-Za-z0-9])(?:\+?(?:86|852)|00?852|0[1-9][0-9]{2})[ ()-]*[0-9]{3,4}[ -][0-9]{3,4}(?![A-Za-z0-9])")
MOBILE = re.compile(r"(?<![A-Za-z0-9])1[3-9][0-9]{9}(?![A-Za-z0-9])")
HKPHONE = re.compile(r"(?<![A-Za-z0-9])[2-9][0-9]{3}[ -][0-9]{4}(?![A-Za-z0-9])")
CARD = re.compile(r"(?<![A-Za-z0-9])(?:[0-9]{4}[ -]?){3,4}[0-9]{1,4}(?![A-Za-z0-9])")
URL = re.compile(r"(?i)\bhttps?://[^\s\"<>]+")
LOCAL = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^\s\"'<>\\]+")
MPLATE = re.compile(r"(?<![A-Za-z0-9])[\u4e00-\u9fff][A-Z][A-Z0-9]{5}(?![A-Za-z0-9])")
PARTICIPANT = re.compile(r"(?<![A-Za-z0-9])A[0-9]+_[A-Z][A-Z0-9_-]+(?![A-Za-z0-9])")
ANON = re.compile(r"(?<![A-Za-z0-9])User[ _-]?[A-Z0-9]+(?![A-Za-z0-9])", re.I)
JSON_NAME = re.compile(r'(?i)("(?:speaker|name|participant|participant_id|author|sender|recipient|contact_name|account_name)"\s*:\s*)"(?:\\.|[^"\\])*"')
PAYMENT = re.compile(r"((?:商户|收款方|付款方|支付金额|付款金额|交易金额|金额)\s*[:：]?\s*)([^\s,;}'\"\]]{1,32})")
ADDRESS = re.compile(r"((?:地址|住址|现居住地址|收货地址|邮编|邮政编码)\s*[:：]?\s*)([^,。；\]}\"']{2,80})")
CURRENCY = re.compile(r"(?<![A-Za-z0-9])(?:HK\$|RMB|[$¥￥])\s*[0-9]+(?:\.[0-9]{1,2})?(?![A-Za-z0-9])", re.I)

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

def sanitize_obj(obj, rel, line, a, key=None):
    """Recursively sanitize parsed JSON values without touching JSON syntax."""
    if isinstance(obj, str):
        if key and str(key).lower() in SENSITIVE_NAME_KEYS:
            a.add("person_name", 1, rel, line)
            return MASK
        return a.text(obj, rel, line)
    if isinstance(obj, list):
        return [sanitize_obj(v, rel, line, a) for v in obj]
    if isinstance(obj, dict):
        return {k: sanitize_obj(v, rel, line, a, k) for k, v in obj.items()}
    return obj

def jsonl(src, dst, rel, a):
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept_ids = set(); kept=failed=bad=segments=0; lines=0
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
            p=obj.get("parsed")
            if isinstance(p,dict) and isinstance(p.get("segments"),list):
                segments += len(p["segments"])
            clean = sanitize_obj(obj, rel, no, a)
            fo.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    return {"lines":lines,"kept":kept,"filtered_failed":failed,"parse_errors":bad,"segments":segments,"kept_ids":kept_ids}

def txt(src,dst,rel,a,kept_ids=None):
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
        for no,s in block: fo.write(a.text(s.rstrip("\n"),rel,no)+"\n")
    with src.open(encoding="utf-8",errors="replace") as fi,dst.open("w",encoding="utf-8",newline="") as fo:
        for no,line in enumerate(fi,1):
            lines+=1
            if line.startswith("===== clip"): flush(fo); block=[(no,line)]
            else:block.append((no,line))
        flush(fo)
    return {"lines":lines,"filtered_empty":empty}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",type=Path,required=True); ap.add_argument("--dst",type=Path,required=True); ap.add_argument("--report",type=Path,required=True); args=ap.parse_args()
    src=args.src.resolve(); dst=args.dst.resolve(); report=args.report.resolve()
    if not src.is_dir(): raise SystemExit(f"missing source: {src}")
    if dst.exists(): raise SystemExit(f"refusing overwrite: {dst}")
    dst.mkdir(parents=True); a=Audit(); js={"files":0,"lines":0,"kept":0,"filtered_failed":0,"parse_errors":0,"segments":0}; ts={"files":0,"lines":0,"filtered_empty":0}; copied=0
    kept_by_recording = {}
    all_files = sorted(p for p in src.rglob("*") if p.is_file())
    # Process JSONL first so TXT blocks can be synchronized with successful IDs.
    for f in [p for p in all_files if p.name == "captions.jsonl"]:
        rel=f.relative_to(src); rs=rel.as_posix(); out=dst/rel
        s=jsonl(f,out,rs,a); kept_by_recording[f.parent] = s.pop("kept_ids", set())
        js["files"] += 1
        for k in js:
            js[k] += int(s.get(k,0))
    for f in all_files:
        rel=f.relative_to(src); rs=rel.as_posix(); out=dst/rel
        if rs=="index.json": continue
        if f.name=="captions.jsonl":
            continue
        elif f.name=="captions.txt":
            s=txt(f,out,rs,a, kept_by_recording.get(f.parent)); ts["files"]+=1; ts["lines"]+=s["lines"]; ts["filtered_empty"]+=s["filtered_empty"]
        else: out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,out); copied+=1
    meta={"version":3,"mask":MASK,"source_name":src.name,"match_counts":dict(sorted(a.counts.items())),"files_with_matches":{k:len(v) for k,v in sorted(a.files.items())},"kept_success_clips":js["kept"],"filtered_failed_clips":js["filtered_failed"],"filtered_empty_txt_blocks":ts["filtered_empty"]}
    idx=json.loads((src/"index.json").read_text(encoding="utf-8")); idx["_redaction"]=meta; (dst/"index.json").write_text(json.dumps(idx,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rep={"version":3,"source_name":src.name,"destination_name":dst.name,"mask":MASK,"input":{"jsonl":js,"txt":ts,"copied_non_caption_files":copied},"match_counts":dict(sorted(a.counts.items())),"files_with_matches":{k:len(v) for k,v in sorted(a.files.items())},"location_hashes":{k:sorted(v) for k,v in sorted(a.locs.items())},"verification":{"jsonl_parse_errors":js["parse_errors"]}}
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"destination":str(dst),"report":str(report),"input":rep["input"],"match_counts":rep["match_counts"]},ensure_ascii=False))
if __name__=="__main__": main()
