# -*- coding: utf-8 -*-
"""第4步: 让 deepseek-v4-pro 扮演 Lucia, 看环境→出下一步动作.
  tf  teacher-forcing 逐点预测(并发): 每步用真实历史回填, 独立预测 -> 干净测一致率
  fr  free rollout(顺序): 用模型自己的输出滚动 -> 模拟Lucia的上午, 允许走分支
用法: python rollout.py tf   |   python rollout.py fr
"""
import os
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

KEY=os.environ.get("DEEPSEEK_API_KEY", "")
MODEL="deepseek-v4-pro"
client=openai.OpenAI(api_key=KEY, base_url="https://api.deepseek.com")

W=json.load(open("world/world_knowledge.json",encoding="utf-8"))
STEPS=json.load(open("world/steps_perc_day1.json",encoding="utf-8"))
PERSONA=W["persona_lucia"]

SYS=("你在扮演真人Lucia,在一个朋友共居实验的家里生活。"
     "你只能依据【此刻看到/听到的环境】+【你最近做过的事】+【你的性格】,"
     "做出你**下一步最可能做的一个动作**。"
     "重要: 真人逐刻的行为大多是不起眼的小动作、配合、观察、等待、搭把手,"
     "只有在自然的时候才会开口说话,不要每一步都长篇大论地发言。"
     "规则: 只输出一行中文,以\"我\"开头,像第一人称监控日志一样**简短具体**(通常6到20字);"
     "可以做数据里没有的事——做你这个真人真实会做的;不要解释,不要分点,不要加引号包整句。")

def env_text(s):
    obs=[]; seen=set()
    for x in s.get("observations",[])+s.get("real_perception",[]):
        if x not in seen: seen.add(x); obs.append(x)
    obs=obs[-14:]
    loc=" · ".join(s["location"])
    lines=[f"【我是谁】{PERSONA}",
           f"【此刻所在】{loc} —— 正在: {s['scene']}",
           f"【在场的人】{', '.join(s['present'])}",
           f"【高层情境】{s['l4_context']}",
           "【我刚刚看到/听到】" + ("\n  - "+"\n  - ".join(obs) if obs else "(暂无特别的)")]
    return "\n".join(lines)

def ask(s, history):
    hist = "\n".join(f"  {i+1}. {h}" for i,h in enumerate(history[-6:])) or "  (这是上午第一个动作)"
    user=f"{env_text(s)}\n【我最近做过】\n{hist}\n\n现在轮到我,我下一步会:"
    for attempt in range(3):
        try:
            r=client.chat.completions.create(model=MODEL,
                messages=[{"role":"system","content":SYS},{"role":"user","content":user}],
                temperature=0.8, max_tokens=1024, timeout=120)
            txt=(r.choices[0].message.content or "").strip().split("\n")[0].strip()
            txt=txt.strip("：:。\" ")
            if not txt.startswith("我"): txt="我"+txt.lstrip("我")
            return txt
        except Exception as e:
            if attempt==2: return f"[ERROR] {e}"
            time.sleep(2*(attempt+1))

def run_tf():
    """并发: 每步用真实GT历史(前几步真实动作)预测当前步."""
    gt=[s["gt_action"] for s in STEPS]
    def one(i):
        s=STEPS[i]; hist=gt[max(0,i-6):i]
        return i,{"step_id":s["step_id"],"time_start":s["time_start"],"location":s["location"],
                  "scene":s["scene"],"gt_action":s["gt_action"],"pred_action":ask(s,hist)}
    res=[None]*len(STEPS); t=time.time(); done=0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs=[ex.submit(one,i) for i in range(len(STEPS))]
        for f in as_completed(futs):
            i,r=f.result(); res[i]=r; done+=1
            if done%20==0: print(f"  TF {done}/{len(STEPS)}  {time.time()-t:.0f}s",flush=True)
    json.dump(res,open("results_tf.json","w"),ensure_ascii=False,indent=2)
    print(f"TF 完成 {len(res)} 步, {time.time()-t:.0f}s -> results_tf.json")

def run_fr():
    """顺序: 用模型自己的历史滚动, 模拟Lucia上午(允许分支)."""
    res=[]; hist=[]; t=time.time()
    for i,s in enumerate(STEPS):
        act=ask(s,hist); hist.append(act)
        res.append({"step_id":s["step_id"],"time_start":s["time_start"],"location":s["location"],
                    "scene":s["scene"],"gt_action":s["gt_action"],"sim_action":act})
        if (i+1)%20==0:
            print(f"  FR {i+1}/{len(STEPS)}  {time.time()-t:.0f}s",flush=True)
            json.dump(res,open("results_fr.json","w"),ensure_ascii=False,indent=2)
    json.dump(res,open("results_fr.json","w"),ensure_ascii=False,indent=2)
    print(f"FR 完成 {len(res)} 步, {time.time()-t:.0f}s -> results_fr.json")

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "tf"
    (run_tf if mode=="tf" else run_fr)()
