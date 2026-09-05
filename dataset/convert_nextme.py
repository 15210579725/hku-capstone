#!/usr/bin/env python3
"""Convert Next-Me caption-result JSONL to unified events.txt format."""
import json, os, re, sys

CAPTION_DIR = os.path.join(os.path.dirname(__file__), '..', 'caption-result')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'nextme')

def parse_time_range(tr: str):
    """Parse '18:43:27–18:43:33' or '18:43:27-18:43:33' into (start, end)."""
    parts = re.split(r'[–\-]', tr.strip(), maxsplit=1)
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), parts[1].strip()

def convert_recording(rec_dir: str, out_dir: str):
    jf = os.path.join(rec_dir, 'captions.jsonl')
    if not os.path.exists(jf):
        return 0

    lines = []
    with open(jf) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            clip = json.loads(raw)
            if not clip.get('ok', False):
                continue
            for seg in (clip.get('segments') or []):
                tr = seg.get('time_range', '')
                start, end = parse_time_range(tr)
                if not start or not end:
                    continue
                action = seg.get('action', '').strip()
                if not action:
                    continue
                lines.append((start, end, action))

    lines.sort(key=lambda x: x[0])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'events.txt'), 'w') as f:
        for start, end, action in lines:
            f.write(f'[{start} -> {end}] {action}\n')

    return len(lines)

def main():
    total = 0
    recs = sorted([d for d in os.listdir(CAPTION_DIR)
                   if os.path.isdir(os.path.join(CAPTION_DIR, d))])

    for rec in recs:
        rec_path = os.path.join(CAPTION_DIR, rec)
        out_path = os.path.join(OUT_DIR, rec)
        n = convert_recording(rec_path, out_path)
        if n > 0:
            print(f'  {rec}: {n} events')
            total += n

    print(f'\nTotal: {len(recs)} recordings, {total} events')

if __name__ == '__main__':
    main()
