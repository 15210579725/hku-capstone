# -*- coding: utf-8 -*-
"""第3步: 把世界时间线 -> 官方 generative_agents 前端(Phaser/demo)吃的回放格式.
照搬官方 the_ville 地图与 sprite, 仅替换 movement 数据驱动 6 个 agent 回放 Lucia 的上午.
Lucia 的 description=真实动作(GT), chat气泡=🤖AI预测+一致标签 -> GT/预测一图并排.
产出: official_repo/.../compressed_storage/lucia_day1_morning/{master_movement,meta}.json
"""
import json, os

FE="official_repo/environment/frontend_server"
SIM="lucia_day1_morning"
OUT=f"{FE}/compressed_storage/{SIM}"
os.makedirs(OUT, exist_ok=True)

# 6 人 -> 官方现成 sprite 名 (空格→下划线即 png 名)
CAST={"Lucia":"Isabella Rodriguez","Jake":"Klaus Mueller","Shure":"Ryan Park",
      "Katrina":"Maria Lopez","Alice":"Hailey Johnson","Tasha":"Abigail Chen"}
NAMES=list(CAST.values())
RING=[(0,0),(2,0),(-2,0),(0,2),(2,-2),(-2,-2)]   # Lucia 居中, 其余拉开~2瓦片避免重叠

# arena -> the_ville 可视可行走区锚点瓦片(回放只做线性走位,不受碰撞影响)
ANCHOR={("客厅","主桌"):(76,20),("Jake工作间","设备区"):(86,18),
        ("客厅","白板区"):(72,23),("客厅","拼图桌"):(79,25),
        ("厨房","操作台"):(68,19),("庭院","餐桌区"):(80,31),
        ("走廊","过道"):(70,28),("二楼","楼上房间"):(88,26)}

def emoji(scene, act):
    s=scene+act
    for k,e in [("硬盘","🔧"),("组装","🔧"),("拼图","🧩"),("白板","🗒️"),("分享","🗣️"),
                ("讨论","💬"),("规划","💬"),("参观","👀"),("烘焙","🧁"),("蛋糕","🧁"),
                ("食材","🧁"),("擦","🧽"),("庭院","🧽"),("打扫","🧽"),("外卖","🍗"),
                ("KFC","🍗"),("手机","📱"),("戳","📱"),("花","💐"),("等待","🧍")]:
        if k in s: return e
    return "🙂"

R=json.load(open("results_tf_judged.json",encoding="utf-8"))
STEPS={s["step_id"]:s for s in json.load(open("world/steps_perc_day1.json",encoding="utf-8"))}

master={}
for i,x in enumerate(R):
    s=STEPS.get(x["step_id"],{})
    loc=tuple(x["location"]); ax,ay=ANCHOR.get(loc,(76,20))
    e=emoji(x["scene"], x["gt_action"])
    tag={"一致":"✅一致","部分一致":"≈部分","合理分支":"↗分支","不一致":"✗不一致"}.get(x.get("label"),"")
    persona={}
    for j,(role,spr) in enumerate(CAST.items()):
        dx,dy=RING[j]
        mv=[ax+dx, ay+dy]
        if role=="Lucia":
            desc=f"[{x['time_start']}] {x['gt_action']} @{'·'.join(x['location'])}"
            # 官方 chat 格式: [[说话人,内容],...] (传字符串会逐字符 garble)
            chat=[["真实GT", x['gt_action']], ["🤖AI预测", f"{x['pred_action']} 〔{tag}〕"]]
            persona[spr]={"movement":mv,"pronunciatio":e,"description":desc,"chat":chat}
        else:
            persona[spr]={"movement":mv,"pronunciatio":"💬" if role in s.get("present",[]) else "🚶",
                          "description":f"{role} 在 {'·'.join(x['location'])}","chat":None}
    master[str(i)]=persona

json.dump(master, open(f"{OUT}/master_movement.json","w"), ensure_ascii=False)
meta={"fork_sim_code":SIM,"start_date":"February 13, 2023",
      "curr_time":"February 13, 2023, 11:09:00","sec_per_step":10,
      "maze_name":"the_ville","persona_names":NAMES,"step":len(R)}
json.dump(meta, open(f"{OUT}/meta.json","w"), ensure_ascii=False, indent=2)
print(f"生成 {len(master)} 步 -> {OUT}")
print(f"角色映射: " + ", ".join(f"{r}={n}" for r,n in CAST.items()))
print(f"demo URL: http://localhost:8000/demo/{SIM}/0/2/")
