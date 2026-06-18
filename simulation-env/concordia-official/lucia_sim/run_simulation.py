#!/usr/bin/env python3
"""Run the LUCIA ego-replay simulation.

Replays L1 caption data through Concordia, asking LUCIA (an LLM-driven agent)
to predict her next action at each decision point. After each prediction the
environment continues with the ground-truth action.

Usage:
    cd concordia-official
    source .venv/bin/activate
    PYTHONPATH=. python lucia_sim/run_simulation.py

    # with options
    PYTHONPATH=. python lucia_sim/run_simulation.py \
        --time-start 11:09 --time-end 11:15 --max-steps 5 --output-dir ./output
"""

import argparse
import json
import os
import sys

# Ensure the project root is on the path so lucia_sim imports work.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import sentence_transformers

from concordia.prefabs.entity import basic as basic_entity
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib

from lucia_sim.deepseek_model import DeepSeekLanguageModel
from lucia_sim.ego_replay_gm import EgoReplayGameMaster
from lucia_sim.data_parser import parse_l1_events, build_decision_points


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
  parser = argparse.ArgumentParser(
      description='LUCIA ego-replay simulation on Concordia.')
  parser.add_argument(
      '--time-start', type=str, default='11:09',
      help='Start time for L1 data window (HH:MM). Default: 11:09')
  parser.add_argument(
      '--time-end', type=str, default='11:15',
      help='End time for L1 data window (HH:MM). Default: 11:15')
  parser.add_argument(
      '--max-steps', type=int, default=None,
      help='Maximum simulation steps. Default: all decision points.')
  parser.add_argument(
      '--output-dir', type=str, default='./output',
      help='Directory for output files. Default: ./output')
  return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  args = parse_args()

  # ---- 1. Language model ------------------------------------------------
  print('[1/6] Initializing DeepSeek model ...')
  model = DeepSeekLanguageModel(
      model_name='deepseek-v4-pro',
      api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
  )

  # ---- 2. Embedder ------------------------------------------------------
  print('[2/6] Loading sentence-transformers embedder ...')
  st_model = sentence_transformers.SentenceTransformer(
      'sentence-transformers/all-MiniLM-L6-v2')
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  # ---- 3. Parse L1 data -------------------------------------------------
  print('[3/6] Parsing L1 caption data ...')
  events = parse_l1_events()
  decision_points = build_decision_points(
      events, time_start=args.time_start, time_end=args.time_end)
  print(f'       Found {len(decision_points)} decision points '
        f'between {args.time_start} and {args.time_end}')

  if not decision_points:
    print('No decision points found. Exiting.')
    return

  # ---- 4. Configure Concordia -------------------------------------------
  print('[4/6] Configuring Concordia simulation ...')

  max_steps = args.max_steps if args.max_steps else len(decision_points)

  prefabs = {
      'basic__Entity': basic_entity.Entity(),
      'ego_replay__GameMaster': EgoReplayGameMaster(),
  }

  instances = [
      # LUCIA entity
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Lucia',
              'goal': (
                  '自然地参与团队活动，做出符合当前情境的行动。'
                  '作为有经验的mentor，积极帮助团队成员。'
              ),
          },
      ),
      # Ego-replay GM
      prefab_lib.InstanceConfig(
          prefab='ego_replay__GameMaster',
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': 'ego replay rules',
              'decision_points': decision_points,
          },
      ),
  ]

  premise = (
      'EgoLife 是一个长期真人实验，多名参与者在同一空间共同生活和工作。'
      'Lucia 是其中一位经验丰富的 mentor，她正在和团队成员一起进行日常活动。'
      '以下是从 Lucia 的第一人称视角记录的事件。'
  )

  config = prefab_lib.Config(
      prefabs=prefabs,
      instances=instances,
      default_premise=premise,
      default_max_steps=max_steps,
  )

  # ---- 5. Run simulation ------------------------------------------------
  print(f'[5/6] Running simulation (max {max_steps} steps) ...')
  sim = simulation_lib.Simulation(config, model, embedder)
  simulation_log = sim.play()

  # ---- 6. Extract predictions and write output --------------------------
  print('[6/6] Writing output ...')
  os.makedirs(args.output_dir, exist_ok=True)

  # -- HTML log (official structured log) --
  html_path = os.path.join(args.output_dir, 'simulation_log.html')
  html_content = simulation_log.to_html(title='LUCIA Ego-Replay Simulation')
  with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
  print(f'  HTML log    -> {html_path}')

  # -- Extract predictions from the GM component --
  predictions = _extract_predictions(sim)

  # -- predictions.json --
  pred_path = os.path.join(args.output_dir, 'predictions.json')
  with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
  print(f'  Predictions -> {pred_path}')

  # -- interaction_log.txt --
  log_path = os.path.join(args.output_dir, 'interaction_log.txt')
  _write_interaction_log(log_path, predictions)
  print(f'  Text log    -> {log_path}')

  print(f'\nDone. {len(predictions)} predictions recorded.')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_predictions(sim: simulation_lib.Simulation) -> list[dict]:
  """Walk GM agents to find the EgoReplayScript component and get predictions."""
  from lucia_sim.ego_replay_gm import EgoReplayScript
  from concordia.components.game_master import event_resolution

  resolution_key = event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY

  for gm in sim.game_masters:
    try:
      component = gm.get_component(resolution_key)
      if isinstance(component, EgoReplayScript):
        return component.get_predictions()
    except (KeyError, TypeError):
      continue

  # Fallback: empty list if not found
  return []


def _write_interaction_log(
    path: str, predictions: list[dict]
) -> None:
  """Write a human-readable step-by-step log."""
  lines = []
  lines.append('=' * 60)
  lines.append('LUCIA Ego-Replay Interaction Log')
  lines.append('=' * 60)
  lines.append('')

  for i, pred in enumerate(predictions):
    lines.append(f'--- Step {pred.get("step", i)} '
                 f'[{pred.get("timestamp", "")}] ---')
    lines.append(f'  LUCIA prediction : {pred.get("prediction", "(none)")}')
    lines.append(f'  Ground truth     : {pred.get("gt_action", "(none)")}')
    lines.append('')

  lines.append('=' * 60)
  lines.append(f'Total steps: {len(predictions)}')
  lines.append('=' * 60)

  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))


if __name__ == '__main__':
  main()
