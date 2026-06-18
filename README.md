# 仓库结构
## emotion-probe方向包含3篇复现论文项目：antropic(gemma 模型）是171种情绪，线性探针，可以在这个基础上操作。RepE是奠基论文，有6种情绪。

潜在方向：
1. 根据egolife的caption数据集，仿照 anthropic 论文，生成相同场景不同情绪会发生的故事，喂进去，取中后层激活值，线性回归/pca/直接相减 拿到不同情绪的表征向量，然后逐 token 可视化那个人在一天中做不同事情的情绪（焦虑，兴奋，无聊），是否能对齐，操纵相同场景不同情绪，看他会做出什么。

2. 由于缺少情感的 ground truth，需要租 meta 眼镜，采集自己一天的数据，并实时（自言自语标注情绪）。并强迫自己这一天做些会引发强烈不同情绪的事情（看世界杯，兴奋；跑步 5km，疲倦；刷抖音 xhs；推进当前项目（无聊/焦躁/好奇））。

## simulation-env方向有两个主要项目：concordia：将llm作为世界模型，无可视化，只有 log；generative agents：可视化做的很好，world model 只是文本 json。

潜在方向：
1. 给 concordia 做前端，把 2d(generative agents)/3d(minecraft)根据 concordia 的 log 做出可视化场景重建。会方便人们 check 仿真时发生了什么。

2. 用 concordia/generative agents模拟egolife/自己的数据，在相同环境下会有怎样的走向，以及对于几个月的长期模拟，小人是否会走向模式重复。在金融危机/世界战争之前模拟，能否复现历史上重大转折

# 汇报文档

https://l0uyw787c0z.feishu.cn/wiki/KASFwMqcdiwE2ykNnwDcQDNinD5

## 仓库结构

```
hku capstone/
├── emotion-probe/                       # 核心:三条情绪探针复现线(按方法论文脉络命名)
│   ├── RepE-2023-Zou/                    # 【旗舰】Representation Engineering 读向量法,Qwen2.5-3B
│   │                                     #   中文日记 → 逐时刻 6 情绪曲线,438 条自陈金标准验证
│   ├── do_llm_feels_2025-wang/           # EmotionCircuits 复现,Llama-3.2-3B
│   │                                     #   六情绪电路级定位(MLP neurons / attn heads)+ steering 因果
│   └── anthropic-2026-gemma-version/     # Gemma 系探针,8 情绪(含 curiosity)
│                                         #   对比故事 → diff-of-means 方向 → 日记打分
├── simulation-env/                      # 「情绪 → 行为」证据链:多智能体行为仿真
│   ├── concordia-official/               # Google DeepMind Concordia + 自有 lucia_sim(EgoLife 行为预测)
│   └── Generative Agents/                # 斯坦福生成式智能体 + 自有 world/(Smallville 行为回放)
└── README.md / CLAUDE.md                # 本文 / 面向开发者的全局导览
```

## ⚠️ Clone 后必读

本仓库是多个研究脚本 + 三个打包的官方框架的集合,**不是一个一键即跑的单体应用**。Clone 后请注意:

1. 大文件未入库,需自行生成/下载
2. 部分脚本含硬编码绝对路径
3. API 密钥走环境变量
