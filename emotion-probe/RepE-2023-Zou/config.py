"""Agent C 全局配置：路径 / 滑窗参数 / 6 情绪顺序 / 层与契约假设。

所有路径绝对化；原始数据严格只读，一切产出写入 ARTIFACTS_DIR。
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 本项目根（emotion-probe/）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 只读原始数据根（Agent A4_LUCIA 事件粒度数据）
DATA_ROOT = "/Users/mac/Desktop/emotion-action predict/事件粒度迭代/data/A4_LUCIA"

# caption 文件模式：{DATA_ROOT}/{level}/day{d}/events.txt
def events_path(level: str, day: int) -> str:
    return os.path.join(DATA_ROOT, level, f"day{day}", "events.txt")

# 金标准自陈情绪 tsv
FEELINGS_TSV = os.path.join(DATA_ROOT, "feelings_all_days.tsv")

# ---------------------------------------------------------------------------
# 模型作用域：换模型时只改这一行（或设环境变量 EMOPROBE_MODEL_TAG）
# 每个模型的专属产物（方向向量/选层/真实分/验证/图）落到 runs/<MODEL_TAG>/
# 共享流水线代码（src、scripts、config）与模型无关中间产物（artifacts/）不变
# ---------------------------------------------------------------------------
MODEL_TAG = os.environ.get("EMOPROBE_MODEL_TAG", "qwen2.5-3b-instruct")

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
MODEL_DIR = os.path.join(RUNS_DIR, MODEL_TAG)
for _sub in ("directions", "select_layer", "scores", "validation", "figures"):
    os.makedirs(os.path.join(MODEL_DIR, _sub), exist_ok=True)

# 模型专属产物路径 helper：model_artifact("scores", "x.parquet")
def model_artifact(kind: str, name: str) -> str:
    return os.path.join(MODEL_DIR, kind, name)

# 方向 artifact 契约路径（GPU 阶段产出；本地 mock 时不存在）
AGENT_B_DIRECTIONS = model_artifact("directions", "emotion_vectors.pt")
SELECT_LAYER_DIR = os.path.join(MODEL_DIR, "select_layer")

# 模型无关的共享中间产物目录（caption 解析/滑窗/金标准时间线/mock 自测）
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def artifact(name: str) -> str:
    return os.path.join(ARTIFACTS_DIR, name)

# ---------------------------------------------------------------------------
# 数据维度常量
# ---------------------------------------------------------------------------
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
DAYS = list(range(1, 8))
DEFAULT_LEVEL = "L2"          # 以 L2 为基（事件粒度适中）

# 6 基本情绪固定顺序（必须与 Agent B 方向 artifact 一致）
EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
NEUTRAL = "neutral"
# 极性分组（用于极性相关验证）
POSITIVE_EMOTIONS = ["happiness", "surprise"]   # surprise 视作弱正/中性偏正
NEGATIVE_EMOTIONS = ["sadness", "anger", "fear", "disgust"]

# ---------------------------------------------------------------------------
# 滑窗参数（D1 已拍板）
# ---------------------------------------------------------------------------
WINDOW_MODE = "count"          # {"count", "time"}
# A) 事件计数窗（默认）
WIN_N = 8
WIN_STRIDE = 4                 # 重叠 50%
# B) 时间窗（备选，~3min 窗）
WIN_MINUTES = 3.0
STEP_MINUTES = 1.5

# ---------------------------------------------------------------------------
# Agent B 方向 artifact 契约假设（适配层在 direction_artifact.py）
# ---------------------------------------------------------------------------
MOCK_HIDDEN_DIM = 2048         # Qwen2.5-3B 量级；真实以 B artifact 为准
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
N_LAYERS_ASSUMED = 36          # Qwen2.5-3B 层数（best_layer 取中间层 ~ -18~-22）
REP_TOKEN = -1                 # 取最后 token

# ---------------------------------------------------------------------------
# 验证 / 打分参数
# ---------------------------------------------------------------------------
PRIMARY_VALIDATION_DAY = 1     # D4：day1 标注最密
HIT_THRESHOLD_SIGMA = 0.5      # 命中判据：z 分 > 当天均值 + 0.5σ（z 后即 > 0.5）
DIVERGENCE_TOPK = 15

# D2 争议细情绪映射开关：True=按争议词典映射(尴尬→fear 等)，False=一律 neutral
CONTROVERSIAL_AS_NEUTRAL = False
