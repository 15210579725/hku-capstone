#!/usr/bin/env python3
"""Create a privacy-redacted copy of a caption delivery directory.

The source directory is never edited.  JSONL records are parsed and sanitized
recursively (schema keys remain unchanged); human-readable TXT files are
sanitized as text.  The audit report contains counts and salted-free location
hashes only, never matched values or surrounding source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# Keep these deliberately conservative around boundaries so timestamps, UUIDs,
# clip IDs, and hexadecimal file hashes are not treated as phone numbers.
EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"
)
API_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16})"
)
BEARER_RE = re.compile(
    r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{16,}"
)
CREDENTIAL_PAIR_RE = re.compile(
    r"(?ix)(\b(?:password|passwd|passcode|api[ _-]?key|secret|"
    r"access[ _-]?token|refresh[ _-]?token|bearer)\b\s*[:=]\s*)"
    r"([^\s,;}'\"\]]{3,})"
)
CN_CREDENTIAL_PAIR_RE = re.compile(
    r"((?:密码|密碼|验证码|驗證碼|口令|门禁(?:码|號|号)?|門禁(?:碼|號)?|"
    r"PIN|OTP)\s*[:：=#]?\s*)([A-Za-z0-9][A-Za-z0-9_.-]{3,})",
    re.IGNORECASE,
)
# Explicitly formatted business/personal telephone numbers and mainland mobile
# numbers.  The alphanumeric boundaries prevent matching a numeric substring
# inside a SHA/UUID filename.
PHONE_SEP_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\+?(?:86|852)|00?852|0[1-9][0-9]{2})"
    r"[ ()-]*[0-9]{3,4}[ -][0-9]{3,4}(?![A-Za-z0-9])"
)
CN_MOBILE_RE = re.compile(
    r"(?<![A-Za-z0-9])1[3-9][0-9]{9}(?![A-Za-z0-9])"
)
CREDIT_CARD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9]{4}[ -]?){3,4}[0-9]{1,4}(?![A-Za-z0-9])"
)
URL_RE = re.compile(r"(?i)\bhttps?://[^\s\"<>]+")
LOCAL_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^\s\"'<>\\]+")
MEETING_URL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:https?://)?(?:zoom\.us|meet\.google\.com|teams\.microsoft\.com|feishu\.cn)/[^\s\"<>]+"
)
CURRENCY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:HK\$|RMB|[$¥￥])\s*[0-9]+(?:\.[0-9]{1,2})?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PAYMENT_PAIR_RE = re.compile(
    r"((?:商户|收款方|付款方|支付金额|付款金额|交易金额|金额)\s*[:：]?\s*)([^\s,;}'\"\]]{1,32})"
)
ADDRESS_PAIR_RE = re.compile(
    r"((?:地址|住址|现居住地址|收货地址|邮编|邮政编码)\s*[:：]?\s*)([^,。；\]}\"']{2,80})"
)
LICENSE_PLATE_RE = re.compile(
    r"(?<![A-Za-z0-9])[\u4e00-\u9fff][A-Z][A-Z0-9]{5}(?![A-Za-z0-9])"
)
# Hong Kong numbers are often rendered as a bare 4-4 pair in screen text
# (for example ``2605 3928``), without a +852 prefix.  Keep boundaries tight
# so UUIDs, timestamps, and frame indices are not touched.
PHONE_PLAIN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[2-9][0-9]{3}[ -][0-9]{4})(?![A-Za-z0-9])"
)
HKID_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]{1,2}[0-9]{6}\([0-9A]\)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Participant identifiers in this project and explicit speaker/name fields are
# identifiers, not schema keys.  Do not run broad capitalized-word replacement;
# it would damage normal prose such as "Physical space" and model names.
PARTICIPANT_ID_RE = re.compile(r"(?<![A-Za-z0-9])A[0-9]+_[A-Z][A-Z0-9_-]+(?![A-Za-z0-9])")
ANON_USER_RE = re.compile(r"(?<![A-Za-z0-9])User[ _-]?[A-Z0-9]+(?![A-Za-z0-9])", re.IGNORECASE)
# Shell prompts and screenshots sometimes expose a local account/host pair
# rather than an email address (for example ``mac@MacdeMacBook-Pro-1144``).
USER_AT_HOST_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[a-z][A-Za-z0-9._-]{1,31}@[A-Za-z][A-Za-z0-9._-]{1,63}(?![A-Za-z0-9._-])"
)
TITLE_NAME_RE = re.compile(
    r"\b(?:Prof\.|Dr\.|Professor|Mr\.|Ms\.|Mrs\.)\s+"
    r"(?!X\b|User\b|Person\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
)
CN_SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
)
CN_TITLE_NAME_RE = re.compile(
    rf"[{CN_SURNAMES}](?:老师|教授|同学|博士|师兄|师姐|导师|学长|学姐)"
)
CN_LABEL_VALUE_RE = re.compile(
    r"((?:姓名|名字|联系人|收件人|发送者|用户名|账号名|用户名称)\s*[:：=]\s*)"
    r"([^\s,;，；}]{1,40})"
)
JSON_NAME_FIELD_RE = re.compile(
    r'(?i)("(?:speaker|name|participant|participant_id|author|sender|recipient|contact_name|account_name)"\s*:\s*)"(?:\\.|[^"\\])*"'
)
INSTITUTION_RE = re.compile(
    r"(?i)\bUniversity of Hong Kong\b|\bHKU(?:L|-SPACE)?\b|"
    r"香港大学|港大|港大图书馆|\bTongji University\b|同济大学|同济"
)

MASK = "XXX"
SENSITIVE_KEYS = {
    "speaker",
    "name",
    "participant",
    "participant_id",
    "author",
    "sender",
    "recipient",
    "contact_name",
    "account_name",
}


class Sanitizer:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.files: defaultdict[str, set[str]] = defaultdict(set)
        self.locations: defaultdict[str, set[str]] = defaultdict(set)

    def _location_hash(self, rel: str, line_no: int) -> str:
        # A deterministic location digest is useful for review without putting
        # source filenames/clip IDs or matched values into the audit artifact.
        return hashlib.sha256(f"{rel}:{line_no}".encode("utf-8")).hexdigest()[:16]

    def _record(self, category: str, n: int, rel: str, line_no: int) -> None:
        if n:
            self.counts[category] += n
            self.files[category].add(rel)
            self.locations[category].add(self._location_hash(rel, line_no))

    def _sub(self, category: str, pattern: re.Pattern[str], text: str, rel: str, line_no: int) -> str:
        matches = list(pattern.finditer(text))
        self._record(category, len(matches), rel, line_no)
        return pattern.sub(MASK, text)

    def sanitize_text(self, text: str, rel: str, line_no: int) -> str:
        # Pairs first: preserve a non-sensitive label but remove its value.
        def cred_repl(m: re.Match[str]) -> str:
            self._record("credential_value", 1, rel, line_no)
            return m.group(1) + MASK

        text = CREDENTIAL_PAIR_RE.sub(cred_repl, text)

        def cn_cred_repl(m: re.Match[str]) -> str:
            self._record("credential_value", 1, rel, line_no)
            return m.group(1) + MASK

        text = CN_CREDENTIAL_PAIR_RE.sub(cn_cred_repl, text)
        text = self._sub("email", EMAIL_RE, text, rel, line_no)
        text = self._sub("api_token", API_RE, text, rel, line_no)
        text = self._sub("bearer_token", BEARER_RE, text, rel, line_no)
        text = self._sub("credit_card", CREDIT_CARD_RE, text, rel, line_no)
        text = self._sub("meeting_url", MEETING_URL_RE, text, rel, line_no)
        text = self._sub("url", URL_RE, text, rel, line_no)
        text = self._sub("local_path", LOCAL_PATH_RE, text, rel, line_no)
        text = self._sub("payment_amount", CURRENCY_RE, text, rel, line_no)
        text = self._sub("license_plate", LICENSE_PLATE_RE, text, rel, line_no)
        text = self._sub("phone_formatted", PHONE_SEP_RE, text, rel, line_no)
        text = self._sub("phone_plain_hk", PHONE_PLAIN_RE, text, rel, line_no)
        text = self._sub("phone_mobile", CN_MOBILE_RE, text, rel, line_no)
        text = self._sub("hkid", HKID_RE, text, rel, line_no)
        text = self._sub("participant_id", PARTICIPANT_ID_RE, text, rel, line_no)
        text = self._sub("anonymized_user", ANON_USER_RE, text, rel, line_no)
        text = self._sub("user_at_host", USER_AT_HOST_RE, text, rel, line_no)
        text = self._sub("institution", INSTITUTION_RE, text, rel, line_no)
        text = self._sub("title_name", TITLE_NAME_RE, text, rel, line_no)
        text = self._sub("cn_title_name", CN_TITLE_NAME_RE, text, rel, line_no)

        def cn_name_repl(m: re.Match[str]) -> str:
            self._record("name_label_value", 1, rel, line_no)
            return m.group(1) + MASK

        text = CN_LABEL_VALUE_RE.sub(cn_name_repl, text)
        def context_repl(category: str):
            def repl(m: re.Match[str]) -> str:
                self._record(category, 1, rel, line_no)
                return m.group(1) + MASK
            return repl
        text = PAYMENT_PAIR_RE.sub(context_repl("payment_merchant"), text)
        text = ADDRESS_PAIR_RE.sub(context_repl("address_value"), text)
        return text

    def sanitize_value(self, value: Any, rel: str, line_no: int, key: str = "") -> Any:
        if isinstance(value, str):
            if key.lower() in SENSITIVE_KEYS:
                self._record("name_field", 1, rel, line_no)
                return MASK
            return self.sanitize_text(value, rel, line_no)
        if isinstance(value, list):
            return [self.sanitize_value(v, rel, line_no, key) for v in value]
        if isinstance(value, dict):
            return {k: self.sanitize_value(v, rel, line_no, str(k)) for k, v in value.items()}
        return value


def iter_source_files(src: Path) -> Iterable[Path]:
    yield from sorted(p for p in src.rglob("*") if p.is_file())


def process_jsonl(src: Path, dst: Path, rel: str, sanitizer: Sanitizer) -> dict[str, int]:
    stats: dict[str, int] = {"lines": 0, "parse_errors": 0, "kept": 0, "filtered_failed": 0, "segments": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                fout.write(sanitizer.sanitize_text(line.rstrip("\n"), rel, line_no) + "\n")
                continue
            if obj.get("ok") is False:
                stats["filtered_failed"] = int(stats["filtered_failed"]) + 1
                continue
            parsed = obj.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
                stats["segments"] += len(parsed["segments"])
            # Parse only for validity/filtering; redact the original JSON line
            # in-place to avoid traversing duplicated content_raw/pass1/pass2
            # trees (which made the first implementation prohibitively slow).
            clean_line = JSON_NAME_FIELD_RE.sub(r'\1"XXX"', line.rstrip("\n"))
            clean_line = sanitizer.sanitize_text(clean_line, rel, line_no)
            fout.write(clean_line + "\n")
            stats["kept"] = int(stats["kept"]) + 1
    return stats


def process_txt(src: Path, dst: Path, rel: str, sanitizer: Sanitizer) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            lines += 1
            fout.write(sanitizer.sanitize_text(line.rstrip("\n"), rel, line_no) + "\n")
    return lines


def build_index(src_index: Path, dst_index: Path, metadata: dict[str, Any], rec_stats: dict[str, dict[str, Any]]) -> None:
    with src_index.open("r", encoding="utf-8") as f:
        index = json.load(f)
    # Recompute counts from the actual kept rows.  The source index was written
    # before late API failures were removed and therefore cannot be copied as-is.
    for rec, entry in index.items():
        if rec.startswith("_") or not isinstance(entry, dict):
            continue
        s = rec_stats.get(rec, {})
        entry["usable_clips"] = int(s.get("kept", 0))
        entry["segments"] = int(s.get("segments", 0))
        entry["filtered_failed_clips"] = int(s.get("filtered_failed", 0))
        target = int(entry.get("target_clips", entry.get("usable_clips", 0)) or 0)
        entry["coverage"] = round(entry["usable_clips"] / target, 3) if target else 0.0
    rec_entries = [v for k, v in index.items() if not k.startswith("_") and isinstance(v, dict)]
    first = [v for v in rec_entries if v.get("in_first_100h")]
    first_target = sum(int(v.get("target_clips", 0) or 0) for v in first)
    first_usable = sum(int(v.get("usable_clips", 0) or 0) for v in first)
    index["_summary"] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "usable_clips_total": sum(int(v.get("usable_clips", 0) or 0) for v in rec_entries),
        "first_100h_usable": first_usable,
        "first_100h_target": first_target,
        "first_100h_coverage": round(first_usable / first_target, 3) if first_target else 0.0,
        "recordings_with_output": sum(1 for v in rec_entries if int(v.get("usable_clips", 0) or 0) > 0),
        "note": "只含顶层 ok=true 的 clip；失败记录已过滤；敏感文本已替换为 XXX。",
    }
    index["_redaction"] = metadata
    dst_index.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def redact(src: Path, dst: Path, report_path: Path) -> dict[str, Any]:
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory does not exist: {src}")
    if dst.exists():
        raise SystemExit(f"destination exists; choose a new path (refusing overwrite): {dst}")
    dst.mkdir(parents=True)
    sanitizer = Sanitizer()
    jsonl_stats = {"files": 0, "lines": 0, "parse_errors": 0, "kept": 0, "filtered_failed": 0}
    rec_stats: dict[str, dict[str, Any]] = {}
    txt_stats = {"files": 0, "lines": 0}
    copied = 0
    for src_file in iter_source_files(src):
        rel_path = src_file.relative_to(src)
        rel = rel_path.as_posix()
        dst_file = dst / rel_path
        if rel == "index.json":
            continue
        if src_file.name == "captions.jsonl":
            s = process_jsonl(src_file, dst_file, rel, sanitizer)
            jsonl_stats["files"] += 1
            jsonl_stats["lines"] += s["lines"]
            jsonl_stats["parse_errors"] += s["parse_errors"]
            jsonl_stats["kept"] += int(s["kept"])
            jsonl_stats["filtered_failed"] += int(s["filtered_failed"])
            rec_stats[src_file.parent.name] = {
                "kept": int(s["kept"]),
                "filtered_failed": int(s["filtered_failed"]),
                "segments": int(s["segments"]),
            }
        elif src_file.name == "captions.txt":
            txt_stats["files"] += 1
            txt_stats["lines"] += process_txt(src_file, dst_file, rel, sanitizer)
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied += 1

    counts = {k: int(v) for k, v in sorted(sanitizer.counts.items())}
    files_hit = {k: len(v) for k, v in sorted(sanitizer.files.items())}
    location_hashes = {
        k: sorted(v) for k, v in sorted(sanitizer.locations.items())
    }
    metadata = {
        "version": 2,
        "mask": MASK,
        "source_name": src.name,
        "redacted_files": jsonl_stats["files"] + txt_stats["files"],
        "filtered_failed_clips": jsonl_stats["filtered_failed"],
        "kept_success_clips": jsonl_stats["kept"],
        "match_counts": counts,
        "files_with_matches": files_hit,
    }
    build_index(src / "index.json", dst / "index.json", metadata, rec_stats)
    report = {
        "version": 2,
        "source_name": src.name,
        "destination_name": dst.name,
        "mask": MASK,
        "input": {
            "jsonl": jsonl_stats,
            "txt": txt_stats,
            "copied_non_caption_files": copied,
            "kept": jsonl_stats["kept"],
            "filtered_failed": jsonl_stats["filtered_failed"],
        },
        "match_counts": counts,
        "files_with_matches": files_hit,
        "location_hashes": location_hashes,
        "verification": {"jsonl_parse_errors_during_copy": jsonl_stats["parse_errors"]},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = redact(args.src, args.dst, args.report)
    print(json.dumps({"destination": str(args.dst), "report": str(args.report), "match_counts": report["match_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
