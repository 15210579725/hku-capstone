#!/usr/bin/env python3
"""Vertex Batch single-pass worker (MEDIUM full frames + MEDIUM gaze crops).

This intentionally keeps one request per clip and reuses the proven tar index,
GCS upload, Batch submit/wait, and HF commit helpers from batch_worker.py.
"""
from __future__ import annotations
import json, os, time, io
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
import caption_core as cc
import caption_v9 as v9
import cloud_worker as cw
import batch_worker as bw

MEDIUM = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM
MODEL = os.environ.get("MODEL", "gemini-3.5-flash")

def crop_bytes(src, meta, fn, gaze):
    im = Image.open(io.BytesIO(src.read(meta[fn]["off"], meta[fn]["size"]))).convert("RGB")
    w, h = im.size; cx, cy = w/2, h/2
    row = gaze.get(fn)
    if row:
        try: cx, cy = float(row.get("gaze_x"))*w/2880, float(row.get("gaze_y"))*h/2880
        except Exception: pass
    side = int(min(w,h)*.62); l=max(0,min(w-side,int(cx-side/2))); t=max(0,min(h-side,int(cy-side/2)))
    out=io.BytesIO(); im.crop((l,t,l+side,t+side)).resize((1440,1440),Image.LANCZOS).save(out,format="JPEG",quality=70)
    return out.getvalue()

def main():
    c=cw.cfg(); c["model"]=MODEL; c["caption_version"]="v10"; c["run_id"]=os.environ.get("RUN_ID",time.strftime("%Y%m%d"))
    c["bucket"]=os.environ.get("GCS_BUCKET",v9.GCS_BUCKET); c["batch_timeout_min"]=float(os.environ.get("BATCH_TIMEOUT_MIN","240")); c["batch_poll_sec"]=float(os.environ.get("BATCH_POLL_SEC","30")); c["upload_workers"]=int(os.environ.get("UPLOAD_WORKERS","8")); c["render_workers"]=int(os.environ.get("RENDER_WORKERS","8"))
    cw.setup_adc(); api=__import__('huggingface_hub').HfApi(token=c["token"]); rec=c["rec"]
    src=bw.rt.open_source(c["tar"],mount_root=c["mount"],token=c["token"]); idx=bw.index_tar(src)
    gaze=bw.read_gaze(src,idx["gaze"]); trails=cc.build_trails(gaze) if gaze else {}; ocr=bw.read_ocr(src,idx["ocr"])
    gcs=bw.Gcs(c["bucket"],project=cc.ADC_PROJECT,workers=int(os.environ.get("UPLOAD_WORKERS","8"))); prefix=f"batch_in/v10/{c['run_id']}/{rec}"
    frames,meta=bw.unpack(c,src,idx["frames"],trails,bool(gaze),gcs,prefix,do_upload=True)
    done=cw.existing_clip_ids(api,c["out_repo"],rec,"captions",".json"); todo=[x for x in sorted(frames) if x not in done]
    if not todo: print("全部已完成"); return
    prompt=(Path(__file__).parent/"prompts"/("v9.txt" if gaze else "v9_nogaze.txt")).read_text().strip(); intervals=[]
    if idx.get("transcript"): intervals=cc.parse_transcript(src.read(idx["transcript"]["data_offset"],idx["transcript"]["size"]).decode("utf-8","replace"))
    jl=Path(f"/tmp/{rec}_v10.jsonl")
    with jl.open("w") as f:
      for cid in todo:
        names=sorted(frames[cid],key=cc.frame_index); sd=bw.build_sd(cid,names,meta,ocr,intervals); parts=[bw.part_text(prompt+"\n\n"+v9.build_context_text_v9(sd,True,True,None)+"\n\n"+"Read the interleaved medium-resolution gaze crops and return dense caption JSON with screen_text.")]
        for i,fn in enumerate(names):
          parts += [bw.part_text(f"[frame_index {i} — {cc.hkt_hms(fn)} HKT]"),bw.part_uri(meta[fn]["uri"],MEDIUM)]
          if i<4:
            u=gcs.put(f"{prefix}/c/__povclip{cid:04d}/{i:02d}.jpg",crop_bytes(src,meta,fn,gaze))
            parts += [bw.part_text(f"[ZOOM index {i} — {cc.hkt_hms(fn)} HKT]"),bw.part_uri(u,MEDIUM)]
        f.write(json.dumps({"request":{"contents":[{"role":"user","parts":parts}],"generationConfig":{"temperature":0.3,"maxOutputTokens":32768}}},ensure_ascii=False)+"\n")
    client=genai.Client(vertexai=True,project=cc.ADC_PROJECT,location=cc.ADC_LOCATION); sub=bw.batch_submit(client,MODEL,jl,gcs,prefix,"v10"); r=bw.batch_wait(client,sub,"v10",c["batch_timeout_min"],c["batch_poll_sec"])
    if not r["ok"]: raise RuntimeError(str(r))
    rows=list(bw.batch_rows(gcs,sub["dest"])); caps=Path(f"/tmp/{rec}_caps"); caps.mkdir(exist_ok=True)
    by={bw.row_clip_id(x):x for x in rows}
    for cid in todo:
      row=by.get(cid,{}); content=(row.get("response",{}).get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","") if row else ""); parsed=v9.parse_json(content)
      out={"clip_id":cid,"recording":rec,"clip_index":f"clip_{cid:04d}","pipeline":"v10-1pass-batch","mode":"v10_1pass_batch","media_resolution":"MEDIUM","model":MODEL,"n_frames":len(frames[cid]),"n_zooms":min(4,len(frames[cid])),"ok":bool(parsed),"parsed":parsed,"content_raw":content}
      (caps/f"clip_{cid:04d}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    cw.commit(api,c["out_repo"],[(p,f"captions/{rec}/{p.name}") for p in caps.glob("clip_*.json")],f"v10 batch clips {rec}"); print(f"DONE_OK {rec} {len(todo)}")
if __name__=="__main__": main()
