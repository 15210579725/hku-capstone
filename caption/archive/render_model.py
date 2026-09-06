import cv2,json,glob,os,textwrap
from pathlib import Path
root=Path('/Users/mac/Desktop/hku capstone/caption'); out=root/'caption_model_batch_20260905'/'qwen_rendered'; out.mkdir(exist_ok=True)
for jf in glob.glob(str(root/'caption_model_batch_20260905/qwen/*.json')):
 d=json.load(open(jf)); p=d.get('parsed');
 if not p: continue
 src=str(root/'caption_review_videos'/d['video']); cap=p.get('scene_summary','')+'\n\n'+p.get('activity_chain','')
 for s in p.get('segments',[])[:6]: cap+=f"\n{s.get('time_range','')} {s.get('action','')}"+(f" 屏幕: {'; '.join(s.get('screen_text',[]))}" if s.get('screen_text') else '')
 cap='\n'.join(textwrap.wrap(cap,55))
 cap_lines=cap.splitlines(); W,H=960,540; panel=420; capw=cv2.VideoCapture(src); fps=capw.get(cv2.CAP_PROP_FPS) or 30; outp=str(out/(Path(d['video']).stem+'_qwen.mp4')); wr=cv2.VideoWriter(outp,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W+panel,H))
 while True:
  ok,fr=capw.read()
  if not ok: break
  fr=cv2.resize(fr,(W,H)); card=__import__('numpy').zeros((H,panel,3),dtype='uint8'); y=28
  for line in cap_lines:
   cv2.putText(card,line[:52],(12,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA); y+=22
   if y>H-20: break
  wr.write(cv2.hconcat([fr,card]))
 capw.release(); wr.release(); print(outp)
