# -*- coding: utf-8 -*-
"""第1步: 由粗到细搭建 Generative-Agents 式世界 + L3 决策步时间线.
- L5/L4 给场景骨架 + 位置;  L3 切决策步(每个 action=一步);  L1/L2 留给 perception.py 检索细节.
- 防泄露: gt_action 不进 context; 自述被错误 diarize 成"我听到Lucia说X"的还原为"我说X".
产出: world/world_knowledge.json (世界树+角色+人设) 和 world/decision_steps_day1.json
"""
import json, re, os

DATA = "/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA"
DAY = "day1"
T0, T1 = "11:00:00", "14:20:00"          # day1 上午(出门采购前)
os.makedirs("world", exist_ok=True)

EVENT_RE = re.compile(r"\[(\d{1,2}:\d{2}:\d{2})\s*->\s*(\d{1,2}:\d{2}:\d{2})\]\s*(.*)")
OBS_PREF = ("我看到","我看着","我看见","我看了","我听到","我听着","我听了","我听",
            "我观察","我注意到","我望着","我盯着","我环视","我打量","我无聊地看")
INT_PREF = ("我觉得","我感到","我感觉","我想着","我在想","我思考","我想到","我猜",
            "我有点","我有些","我不确定","我好奇","我期待","我构想","我犹豫","我纠结")

def sec(t):
    p = t.split(":"); return int(p[0])*3600+int(p[1])*60+int(p[2])

def parse(layer):
    out=[]
    for ln in open(f"{DATA}/{layer}/{DAY}/events.txt", encoding="utf-8"):
        m=EVENT_RE.match(ln.strip())
        if m: out.append({"s":m.group(1),"e":m.group(2),"t":m.group(3).strip()})
    return out

def fix_self(text):
    """把佩戴者自述被错误 diarize 的 '我听到Lucia说X' 还原为 '我说X'."""
    m=re.match(r"我听到Lucia说[\"“]?(.+?)[\"”]?$", text)
    return "我说\""+m.group(1)+"\"" if m else text

def classify(t):
    t=fix_self(t)
    if t.startswith("我说"): return "action"
    if not t.startswith("我"): return "observation"
    for p in INT_PREF:
        if t.startswith(p): return "internal"
    for p in OBS_PREF:
        if t.startswith(p): return "observation"
    return "action"

# ── 上午"场景→位置"时间线 (读 L4/L5 骨架人工对齐, sector·arena) ──
SCENES = [
    ("11:09:56","11:12:49","客厅","主桌","聚集围桌·戳手机录制·拆硬盘包装",["Jake","Shure","Katrina","Alice","Tasha"]),
    ("11:12:49","11:15:15","Jake工作间","设备区","参观Jake工作间听数据采集系统讲解",["Jake","Shure","Katrina","Alice","Tasha"]),
    ("11:15:15","11:23:33","客厅","主桌","组装硬盘(Lucia当mentor指导)",["Jake","Shure","Katrina","Alice"]),
    ("11:23:33","11:51:33","客厅","白板区","白板讨论最后一天嘉宾邀请+轮流分享DIY方案",["Jake","Shure","Katrina","Alice","Tasha"]),
    ("11:51:33","12:26:13","客厅","主桌","规划活动流程·时间线·采购方案",["Jake","Shure","Katrina","Alice","Tasha"]),
    ("12:26:13","13:23:28","客厅","拼图桌","和Alice、Tasha玩野生动物拼图",["Alice","Tasha","Jake"]),
    ("13:23:28","13:33:03","客厅","主桌","Tasha展示烘焙工具和原材料",["Tasha","Jake","Alice","Katrina"]),
    ("13:33:03","13:44:41","厨房","操作台","整理归类烘焙食材·巧克力放冰箱",["Tasha","Alice"]),
    ("13:44:41","13:59:40","客厅","主桌","点KFC外卖·讨论调酒方案",["Jake","Katrina","Tasha","Shure"]),
    ("13:59:40","14:10:15","客厅","主桌","讨论火锅采购清单",["Tasha","Jake","Katrina","Alice","Shure"]),
    ("14:10:15","14:20:00","庭院","餐桌区","准备户外用餐·擦桌椅",["Jake","Alice","Tasha","Katrina","Shure"]),
]
def locate(t):
    ts=sec(t)
    for s,e,sec_,arena,scene,present in SCENES:
        if sec(s)<=ts<sec(e): return [sec_,arena],scene,present
    return ["客厅","主桌"],"其他",["Jake","Shure","Katrina","Alice","Tasha"]

def l4_ctx(t, l4):
    ts=sec(t); best=""; bd=1e9
    for ev in l4:
        a,b=sec(ev["s"]),sec(ev["e"])
        if a<=ts<=b: return ev["t"]
        d=min(abs(ts-a),abs(ts-b))
        if d<bd: bd=d; best=ev["t"]
    return best

# ── 构建决策步 (L3, 每个 action 切一步; 之间的 obs/internal 作为观察) ──
def build_steps():
    l3=parse("L3"); l4=parse("L4")
    steps=[]; pend=[]; sid=0
    for ev in l3:
        if sec(ev["s"])<sec(T0): continue
        if sec(ev["s"])>=sec(T1): break
        cat=classify(ev["t"]); txt=fix_self(ev["t"])
        if cat in ("observation","internal"):
            pend.append(txt)
        else:
            sid+=1
            loc,scene,present=locate(ev["s"])
            steps.append({"step_id":sid,"time_start":ev["s"],"time_end":ev["e"],
                "location":loc,"scene":scene,"present":present,
                "l4_context":l4_ctx(ev["s"],l4),"observations":list(pend),
                "gt_action":txt,"gt_raw":f"[{ev['s']} -> {ev['e']}] {txt}"})
            pend=[]
    return steps

# ── 世界树 + 角色 + 人设 ──
WORLD = {
  "world":"合宿之家 (EgoLife 共居实验·最后第二天)",
  "day":"day1 上午 11:09–14:17 (录制前半天)",
  "tree":{
    "客厅":{"arenas":{
        "主桌":["大桌","6把椅子","手机","4个硬盘盒","螺丝刀","数据线","充电宝","眼镜"],
        "白板区":["白板","白板笔","嘉宾名单","花束快递","DIY道具"],
        "拼图桌":["野生动物拼图","拼图支架"],
        "沙发":["沙发","抱枕"]}},
    "Jake工作间":{"arenas":{"设备区":["数据采集系统","硬盘阵列","显示器"]}},
    "厨房":{"arenas":{"操作台":["冰箱","烤箱","烘焙模具","打蛋盆","巧克力","食材"]}},
    "庭院":{"arenas":{"餐桌区":["户外餐桌","凳子","抹布"]}},
    "走廊":{"arenas":{"过道":["纸箱","快递","洗衣机"]}},
    "二楼":{"arenas":{"楼上房间":["各自卧室","行李","衣物"]}}
  },
  "characters":{
    "Lucia":"我·本场佩戴者。环保DIY达人，本次活动的mentor角色；爱笑、乐于助人、偶尔会尴尬手足无措，爱刷小红书找灵感。",
    "Jake":"组织者/主持人，推进流程、分配任务、讲解设备。",
    "Shure":"爱主持白板、提议放BGM、点子多(讲故事换代币拍卖)。",
    "Katrina":"分享压花DIY，细心，关注招待细节。",
    "Alice":"带了野生动物拼图，负责联系司机，安静配合。",
    "Tasha":"带了全套烘焙物资(纸杯蛋糕)，下午是甜品主理人。"
  },
  "persona_lucia":(
    "我是Lucia，这次共居活动的mentor。我热衷环保手工(水母灯、纸箱小狗、环保短剧)，"
    "做过硬盘组装所以乐于指导别人。我性格随和爱笑，气氛尴尬时会主动打圆场或自嘲化解；"
    "看到别人需要帮忙会主动搭手(拆快递、擦桌子、打下手)。我喜欢刷小红书找灵感。"
    "说话偏口语、亲和，常用'哈哈''我觉得'。"),
  "feeling_tendencies":["爱笑/忍不住笑","偶尔尴尬手足无措","乐于助人主动搭手",
    "对环保和手工有热情","欣赏别人点子时会赞叹/拍手","无聊时会找事做"]
}

if __name__=="__main__":
    steps=build_steps()
    json.dump(WORLD, open("world/world_knowledge.json","w"), ensure_ascii=False, indent=2)
    json.dump(steps, open("world/decision_steps_day1.json","w"), ensure_ascii=False, indent=2)
    json.dump(steps, open("world/decision_steps.json","w"), ensure_ascii=False, indent=2)  # build_viz 兼容名
    print(f"L3 上午决策步: {len(steps)} 个 -> world/decision_steps_day1.json")
    from collections import Counter
    c=Counter(s["scene"] for s in steps)
    print("各场景决策步数:")
    for k,v in c.items(): print(f"  {v:3d}  {k}")
    print("\n样例 step 8:")
    s=steps[7]
    print(json.dumps({k:s[k] for k in["step_id","time_start","location","scene","present","observations","gt_action"]},ensure_ascii=False,indent=2))
