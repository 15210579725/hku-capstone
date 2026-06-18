# -*- coding: utf-8 -*-
"""第5步: 一致率评测. 对每个决策步, 判 AI 动作 vs Lucia真实动作(GT)的一致性.
4 类: 一致(1.0) / 部分一致(0.5) / 合理分支(0.3) / 不一致(0.0)
一致率 = (#一致 + 0.5*#部分一致)/N ;  合理分支率 = #合理分支/N ; 严格一致率 = #一致/N
用法: python judge.py results_tf.json   |   python judge.py results_fr.json
"""
import os
import json, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

client=openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
MODEL="deepseek-v4-pro"
PRED_KEY={"results_tf.json":"pred_action","results_fr.json":"sim_action"}

JSYS=("你是行为一致性评审。给你某情境下【真实动作】(GT)和【AI动作】,判断AI动作与真实动作的一致程度。"
      "只看动作的核心意图(做什么/对谁/对什么物)是否吻合,忽略措辞长短繁简。四选一:\n"
      "一致: 核心动作和对象基本相同(如都在擦桌子/都在指导组装/都在拼图)。\n"
      "部分一致: 同一情境同一对象但具体行为有出入,或意图相近方向一致。\n"
      "合理分支: 和GT不同,但在该情境下符合这个人性格、是合情合理的另一种选择。\n"
      "不一致: 跑题、不合情境、或这个人不会这么做。\n"
      "只输出JSON: {\"label\":\"一致|部分一致|合理分支|不一致\",\"reason\":\"<10字\"}")

SCORE={"一致":1.0,"部分一致":0.5,"合理分支":0.3,"不一致":0.0}

def judge_one(scene, observ, gt, pred):
    user=(f"情境: {scene}\n此前看到/听到: {observ}\n"
          f"【真实动作 GT】{gt}\n【AI动作】{pred}\n判断:")
    for a in range(3):
        try:
            r=client.chat.completions.create(model=MODEL,
                messages=[{"role":"system","content":JSYS},{"role":"user","content":user}],
                temperature=0.0, max_tokens=900, timeout=120)
            t=r.choices[0].message.content or ""
            m=re.search(r'\{.*\}', t, re.S)
            o=json.loads(m.group(0)) if m else {"label":"不一致","reason":"parse"}
            lab=o.get("label","不一致").strip()
            if lab not in SCORE: lab="不一致"
            return {"label":lab,"score":SCORE[lab],"reason":o.get("reason","")}
        except Exception as e:
            if a==2: return {"label":"不一致","score":0.0,"reason":f"err"}
            time.sleep(2)

def main(path):
    R=json.load(open(path,encoding="utf-8")); pk=PRED_KEY[path]
    STEPS={s["step_id"]:s for s in json.load(open("world/steps_perc_day1.json",encoding="utf-8"))}
    def one(i):
        x=R[i]; s=STEPS.get(x["step_id"],{})
        observ=" / ".join((s.get("observations",[])+s.get("real_perception",[]))[:6]) or "(无)"
        j=judge_one(x["scene"], observ, x["gt_action"], x[pk])
        return i,{**x,**j}
    out=[None]*len(R); t=time.time(); done=0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(one,i) for i in range(len(R))]):
            i,r=f.result(); out[i]=r; done+=1
            if done%30==0: print(f"  judge {done}/{len(R)} {time.time()-t:.0f}s",flush=True)
    outp=path.replace(".json","_judged.json")
    json.dump(out,open(outp,"w"),ensure_ascii=False,indent=2)
    # ── 汇总 ──
    n=len(out); from collections import Counter,defaultdict
    cnt=Counter(x["label"] for x in out); avg=sum(x["score"] for x in out)/n
    consist=sum(x["score"] for x in out if x["label"] in("一致","部分一致"))/n
    strict=cnt["一致"]/n; branch=cnt["合理分支"]/n
    print(f"\n===== {path} 一致率汇总 (N={n}) =====")
    print(f"  分布: " + " | ".join(f"{k}:{cnt[k]}" for k in ["一致","部分一致","合理分支","不一致"]))
    print(f"  一致率(一致+0.5部分) : {consist*100:.1f}%")
    print(f"  严格一致率(仅一致)   : {strict*100:.1f}%")
    print(f"  合理分支率           : {branch*100:.1f}%")
    print(f"  平均分               : {avg:.3f}")
    # 分场景
    sc=defaultdict(lambda:[0,0.0])
    for x in out: sc[x["scene"]][0]+=1; sc[x["scene"]][1]+=x["score"]
    print("  分场景平均分:")
    for k,(c,sm) in sc.items(): print(f"    {sm/c:.2f}  ({c:3d}步)  {k}")
    print(f"-> {outp}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "results_tf.json")
