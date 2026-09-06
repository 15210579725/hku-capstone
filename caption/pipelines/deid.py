#!/usr/bin/env python3
"""caption 公开前的文本层脱敏。

原则（用户 2026-08 定的）：**别遮一切**。只替换结构化的身份标识符，
浏览内容、GPT 对话、工作内容全部保持可读 —— 遮掉就没有研究价值了。

prompt Rule 8 让模型自己匿名化，但实测 2260 个 clip 里仍漏了 3462 处 HKU、
289 处「X老师」、47 处「Prof. 名」、9 个真实邮箱、14 个手机号。这一遍是兜底。

用法: python deid.py <输入目录> <输出目录>
"""
from __future__ import annotations

import json, re, sys, collections
from pathlib import Path

# 常见姓氏：只有「姓 + 称谓」才替换，避免把「以及老师」「所有同学」误伤
SURNAMES = ("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
            "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
            "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
            "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
            "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍")
TITLES = "老师|教授|同学|博士|师兄|师姐|导师|学长|学姐"

# 私有/回环 IP 不算敏感（127.x、10.x、192.168.x、172.16-31.x）
PRIVATE_IP = re.compile(r"^(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)")


def _ip(m):
    return m.group(0) if PRIVATE_IP.match(m.group(0)) else "0.0.0.0"


def _digits(m):
    """4-4(-4) 数字组：录制名里的 UUID 片段（4871-9269）会误命中，先排除。"""
    s = m.group(0)
    lo = m.string[max(0, m.start() - 1):m.start()]
    hi = m.string[m.end():m.end() + 1]
    if re.match(r"[0-9a-fA-F-]", lo) or re.match(r"[0-9a-fA-F-]", hi):
        return s
    return "XXX-XXXX-XXXX"


def _github(m):
    return f"https://github.com/User_X/{m.group(2)}"


RULES = [
    # —— 密钥/凭据：整段抹掉，最优先
    (re.compile(r"\b(?:sk-|pk-|AIza|ghp_|gho_|AQ\.)[A-Za-z0-9_\-]{16,}"), "X"),
    # —— 邮箱
    (re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
     "email_x@example.com"),
    # —— 手机号
    (re.compile(r"\b(?:\+?86)?1[3-9]\d{9}\b"), "XXX-XXXX-XXXX"),
    # —— 身份证
    (re.compile(r"\b\d{17}[\dXx]\b"), "XXXXXXXXXXXXXXXXXX"),
    # —— 4-4(-4) 数字组（座机/卡号），排除 UUID 片段
    (re.compile(r"\b\d{4}[- ]\d{4}(?:[- ]\d{4})?\b"), _digits),
    # —— 公网 IP
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _ip),
    # —— 本机路径里的用户名
    (re.compile(r"/Users/(?!UserX\b)[A-Za-z0-9_.-]+"), "/Users/UserX"),
    (re.compile(r"/home/(?!userx\b)[A-Za-z0-9_.-]+"), "/home/userx"),
    # —— github / 社交平台上的真实 handle
    (re.compile(r"https?://github\.com/(?!User_X\b)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"),
     _github),
    (re.compile(r"https?://(?:www\.)?(?:linkedin\.com/in|twitter\.com|x\.com|"
                r"instagram\.com|facebook\.com)/[A-Za-z0-9_.-]+"),
     "https://social.example.com/User_X"),
    # —— 微信/QQ 号（要有 id/号 + 分隔符才算，"WeChat messages" 不是）
    (re.compile(r"(?i)\b(wechat|weixin|微信|qq)\s*(?:id|号)\s*[:：]\s*[A-Za-z0-9_-]{4,}"),
     lambda m: f"{m.group(1)} id: User_X"),
    # —— 机构：用户本人的两个学校
    # HKUL（港大图书馆）这类后缀变体 \bHKU\b 匹配不到，2026-09-05 复核时漏网过。
    # HKUST 是另一所学校、不是用户单位，不动它。
    (re.compile(r"(?i)\buniversity of hong kong\b|\bHKU(?:L|-SPACE)?\b|"
                r"香港大学|港大|港大图书馆"), "University X"),
    (re.compile(r"(?i)\btongji\s+university\b|同济大学|同济"), "University Y"),
    # —— 英文人名：Prof./Dr. + 名（已匿名的 User/X 不动）
    (re.compile(r"\b(Prof\.|Dr\.|Professor|Mr\.|Ms\.|Mrs\.)\s+"
                r"(?!User\b|X\b|Person\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
     lambda m: f"{m.group(1)} X"),
    # —— 中文「姓 + 称谓」
    (re.compile(rf"[{SURNAMES}](?:{TITLES})"), lambda m: "X" + m.group(0)[1:]),
]

AUDIT = {
    "机构_HKU": r"(?i)\buniversity of hong kong\b|\bHKU(?:L|-SPACE)?\b|港大|香港大学",
    "机构_同济": r"(?i)\btongji\b|同济",
    "邮箱": r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "手机": r"\b(?:\+?86)?1[3-9]\d{9}\b",
    "身份证": r"\b\d{17}[\dXx]\b",
    "公网IP": r"\b(?!127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)(?:\d{1,3}\.){3}\d{1,3}\b",
    "本机路径": r"/Users/(?!UserX\b)[A-Za-z0-9_.-]+",
    "密钥": r"\b(?:sk-|pk-|AIza|ghp_|gho_|AQ\.)[A-Za-z0-9_\-]{16,}",
    "github真实handle": r"https?://github\.com/(?!User_X\b)[A-Za-z0-9_.-]+/",
    "英文人名": r"\b(?:Prof\.|Dr\.|Professor|Mr\.|Ms\.|Mrs\.)\s+(?!User\b|X\b|Person\b)[A-Z][a-z]+",
    "中文姓名": rf"[{SURNAMES}](?:{TITLES})",
}


def scrub(x):
    if isinstance(x, str):
        for pat, rep in RULES:
            x = pat.sub(rep, x)
        return x
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, dict):
        # 录制名/文件名不动，动了对不上 HF 和 GCS 的路径
        return {k: (v if k in ("recording", "clip_index", "run_id") else scrub(v))
                for k, v in x.items()}
    return x


def count(blob, table):
    return {k: len(re.findall(p, blob)) for k, p in table.items()}


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    before, after = collections.Counter(), collections.Counter()
    nclip = nrec = 0
    for p in sorted(src.glob("*/captions.jsonl")):
        rec = p.parent.name
        outd = dst / rec
        outd.mkdir(exist_ok=True)
        rows = []
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            before.update(count(ln, AUDIT))
            o2 = scrub(o)
            s2 = json.dumps(o2, ensure_ascii=False)
            after.update(count(s2, AUDIT))
            rows.append((o2, s2))
            nclip += 1
        with (outd / "captions.jsonl").open("w") as fh:
            for _, s2 in rows:
                fh.write(s2 + "\n")
        # 可读版同样脱敏后重新渲染
        with (outd / "captions.txt").open("w") as fh:
            for o2, _ in rows:
                m = o2.get("merged") or o2.get("parsed") or {}
                fh.write(f"===== clip {o2.get('clip_id')}  {o2.get('time_range','')}  "
                         f"[{o2.get('model','?')}] =====\n")
                if m.get("scene_summary"):
                    fh.write(f"场景: {m['scene_summary']}\n")
                for s in (m.get("segments") or []):
                    fh.write(f"  {s.get('time_range','')}  {s.get('action','')}\n")
                    if s.get("speech"):
                        fh.write(f"      语音: {s['speech']}\n")
                    if s.get("screen_text_detail"):
                        fh.write(f"      屏幕: {str(s['screen_text_detail'])[:400]}\n")
                fh.write("\n")
        nrec += 1
    idx = src / "index.json"
    if idx.exists():
        (dst / "index.json").write_text(idx.read_text())
    rep = {"clips": nclip, "recordings": nrec,
           "before": dict(before), "after": dict(after)}
    (dst / "pii_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"脱敏 {nrec} 条录制 / {nclip} 个 clip")
    print(f"{'类别':<20}{'脱敏前':>8}{'脱敏后':>8}")
    for k in AUDIT:
        print(f"{k:<20}{before[k]:>8}{after[k]:>8}")
    return rep


if __name__ == "__main__":
    main()
