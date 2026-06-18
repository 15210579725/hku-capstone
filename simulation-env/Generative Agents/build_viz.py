# -*- coding: utf-8 -*-
"""把世界+决策步+评测结果内嵌生成自包含 2D 可视化 viz.html (双击即开,无需server)"""
import json
W=json.load(open('world/world_knowledge.json',encoding='utf-8'))
S=json.load(open('world/decision_steps.json',encoding='utf-8'))
try: TF=json.load(open('results_tf.json',encoding='utf-8'))
except: TF=[]
try: FR=json.load(open('results_fr.json',encoding='utf-8'))
except: FR=[]
DATA=json.dumps({"world":W,"steps":S,"tf":TF,"fr":FR},ensure_ascii=False)

HTML=r'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Lucia 仿真环境 · day1 上午</title><style>
body{margin:0;font-family:-apple-system,"PingFang SC",sans-serif;background:#1b1d23;color:#e6e6e6}
#wrap{display:flex;gap:14px;padding:16px}
#left{flex:0 0 640px}#right{flex:1;min-width:320px}
canvas{background:#23262e;border-radius:10px;box-shadow:0 2px 12px #0006}
h2{margin:4px 0 10px;font-size:18px}.sub{color:#9aa0aa;font-size:12px;margin-bottom:8px}
.ctrl{margin:12px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{background:#2f333c;color:#e6e6e6;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px}
button:hover{background:#3a3f4a}input[type=range]{flex:1;min-width:200px}
.card{background:#23262e;border-radius:10px;padding:14px;margin-bottom:10px}
.lbl{color:#7f8794;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.gt{color:#7fd17f}.pred{color:#7fb6ff}.sim{color:#ffc870}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.s1{background:#1f5d2f;color:#aef0ae}.s5{background:#6b5a1f;color:#ffe79a}.s0{background:#6b2626;color:#ffb0b0}
.metric{display:inline-block;background:#2a2e37;border-radius:8px;padding:8px 14px;margin:4px 6px 4px 0}
.metric b{font-size:20px;color:#7fb6ff}.metric span{display:block;font-size:11px;color:#9aa0aa}
</style></head><body><div id="wrap">
<div id="left"><h2>🏠 合宿之家 · Lucia 的上午 (2D 俯视)</h2>
<div class="sub">圆点=人物 · 红=Lucia · 头顶气泡=当前动作 · 高亮房间=Lucia所在</div>
<canvas id="cv" width="620" height="440"></canvas>
<div class="ctrl">
<button id="play">▶ 播放</button>
<select id="mode"><option value="gt">真实轨迹</option><option value="tf">TF预测</option><option value="fr">自由仿真</option></select>
<input type="range" id="slider" min="0" value="0"><span id="tstamp"></span></div>
<div id="metrics"></div></div>
<div id="right">
<div class="card"><div class="lbl">第 <span id="sid"></span> 步 · <span id="loc"></span></div>
<div id="ctx" style="margin-top:8px;line-height:1.5"></div></div>
<div class="card"><div class="lbl">真实 Lucia 的动作 (GT)</div><div class="gt" id="agt" style="margin-top:6px"></div>
<div class="lbl" style="margin-top:12px">AI 的动作 <span id="modename"></span></div><div id="aai" style="margin-top:6px"></div>
<div id="jscore" style="margin-top:8px"></div></div></div>
</div><script>
/*DATA*/
'''

HTML += r'''
function sideUpdate(i){
  const step=S[i], mode=document.getElementById('mode').value;
  document.getElementById('sid').textContent=i;
  document.getElementById('loc').textContent=step.location.join(' · ');
  document.getElementById('tstamp').textContent=step.time_start;
  document.getElementById('ctx').textContent=step.context;
  document.getElementById('agt').textContent=step.gt_action;
  const mn={gt:'(真实)',tf:'(TF预测)',fr:'(自由仿真)'}[mode];
  document.getElementById('modename').textContent=mn;
  let act,sc=null;
  if(mode==='tf'){act=(TF[i]||{}).pred_action;sc=(TF[i]||{}).score;}
  else if(mode==='fr'){act=(FR[i]||{}).sim_action;sc=(FR[i]||{}).score;}
  else{act=step.gt_action;}
  const ai=document.getElementById('aai'); ai.textContent=act||'—';
  ai.className= mode==='fr'?'sim':mode==='tf'?'pred':'gt';
  const js=document.getElementById('jscore');
  if(sc!==null){const cls=sc>=0.99?'s1':sc>=0.5?'s5':'s0';
    js.innerHTML='一致度 <span class="badge '+cls+'">'+sc+'</span>'+(sc<0.5?' &nbsp;⚠ 偏离真实':'');}
  else js.innerHTML='';
}
function metrics(){
  function summ(R,k){if(!R.length)return null;let a=0,s=0,l=0;R.forEach(x=>{a+=x.score;if(x.score>=0.99)s++;if(x.score>=0.5)l++;});const n=R.length;return{avg:(a/n).toFixed(3),strict:(100*s/n).toFixed(0),loose:(100*l/n).toFixed(0)};}
  const t=summ(TF),f=summ(FR);let h='';
  if(t)h+='<div class="metric"><b>'+t.loose+'%</b><span>TF宽松一致率</span></div><div class="metric"><b>'+t.strict+'%</b><span>TF严格一致率</span></div><div class="metric"><b>'+t.avg+'</b><span>TF平均分</span></div>';
  if(f)h+='<div class="metric"><b>'+f.loose+'%</b><span>自由仿真一致率</span></div>';
  document.getElementById('metrics').innerHTML=h;
}
function render(i){cur=i;slider.value=i;draw(i);sideUpdate(i);}
slider.oninput=e=>render(+e.target.value);
document.getElementById('mode').onchange=()=>render(cur);
document.getElementById('play').onclick=function(){
  playing=!playing;this.textContent=playing?'⏸ 暂停':'▶ 播放';
  if(playing)timer=setInterval(()=>{if(cur>=S.length-1){render(0);}else render(cur+1);},1400);
  else clearInterval(timer);};
if(!CanvasRenderingContext2D.prototype.roundRect){CanvasRenderingContext2D.prototype.roundRect=function(x,y,w,h,r){this.moveTo(x+r,y);this.arcTo(x+w,y,x+w,y+h,r);this.arcTo(x+w,y+h,x,y+h,r);this.arcTo(x,y+h,x,y,r);this.arcTo(x,y,x+w,y,r);return this;};}
metrics(); render(0);
</script></body></html>'''

html_out = HTML.replace('/*DATA*/', 'const DATA='+DATA+';')
open('viz.html','w',encoding='utf-8').write(html_out)
print('viz.html bytes=', len(html_out))
