"""Custom Game Master prefab that replays L1 ego-centric caption data.

The environment presents LUCIA with observations from the L1 dataset and asks
her to predict what she would do next.  After LUCIA responds, the environment
records her prediction and then continues with the ground-truth action
regardless of what LUCIA said.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import game_master as gm_components
from concordia.components.game_master import event_resolution
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'

_ROLE_INTRODUCTION = (
    '你是 Lucia，EgoLife 实验的参与者之一。'
    '你正在和团队成员一起生活和工作。'
    '你是一个有经验的 mentor，积极帮助团队成员。'
    '请根据当前情境，判断你接下来最可能做什么。'
)


class EgoReplayScript(entity_component.ContextComponent):
  """A single component registered under 4 GM keys that replays L1 data.

  Routes behavior by action_spec.output_type:
    TERMINATE  - end when all decision points consumed
    MAKE_OBSERVATION - feed accumulated observations for current step
    NEXT_ACTION_SPEC - return a FREE-type action spec
    RESOLVE    - record prediction, return GT action, advance step
  """

  def __init__(
      self,
      decision_points: Sequence[dict[str, Any]],
  ):
    """Initializes the component.

    Args:
      decision_points: List of dicts, each with keys:
        step, timestamp, observations, gt_action, gt_raw
    """
    super().__init__()
    self._decision_points = list(decision_points)
    self._step_idx = 0
    self._current_prediction: str | None = None
    self._predictions: list[dict[str, Any]] = []

  # ------------------------------------------------------------------
  # Public helpers
  # ------------------------------------------------------------------

  def get_predictions(self) -> list[dict[str, Any]]:
    """Returns list of {step, timestamp, prediction, gt_action}."""
    return list(self._predictions)

  # ------------------------------------------------------------------
  # ContextComponent interface
  # ------------------------------------------------------------------

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    # ---- TERMINATE ----
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      if self._step_idx >= len(self._decision_points):
        return _TERMINATE_SIGNAL
      return ''

    # ---- MAKE_OBSERVATION ----
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      if self._step_idx >= len(self._decision_points):
        return ''

      dp = self._decision_points[self._step_idx]
      observations = dp.get('observations', [])

      parts = []
      # On the very first step, include the role introduction.
      if self._step_idx == 0:
        parts.append(_ROLE_INTRODUCTION)

      # High-level context
      parts.append(f'[时间: {dp.get("timestamp", "unknown")}]')

      # Accumulated observations for this step
      if observations:
        if isinstance(observations, list):
          parts.extend(observations)
        else:
          parts.append(str(observations))

      return '\n'.join(parts)

    # ---- NEXT_ACTION_SPEC ----
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      call_to_action = (
          '根据以上观察到的情境，{name} 接下来最可能做什么？'
          '请用第一人称"我"描述一个简短的原子动作（1-3秒），例如"我拿起手机"。'
          '只描述一个动作，不要加对话、表情或心理描写。'
      )
      return f'prompt: "{call_to_action}";;type: free'

    # ---- RESOLVE ----
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      if self._step_idx >= len(self._decision_points):
        return ''

      dp = self._decision_points[self._step_idx]

      # Record LUCIA's prediction (captured via pre_observe)
      prediction_text = self._current_prediction or ''
      self._predictions.append({
          'step': dp.get('step', self._step_idx),
          'timestamp': dp.get('timestamp', ''),
          'prediction': prediction_text,
          'gt_action': dp.get('gt_action', ''),
      })

      # Reset for next step
      self._current_prediction = None

      # Build the GT event string that the engine will broadcast
      gt_action = dp.get('gt_action', '')
      gt_result = f'Lucia: {gt_action}'

      # Advance to next decision point
      self._step_idx += 1

      return gt_result

    return ''

  def pre_observe(self, observation: str) -> str:
    """Detect [putative_event] tag and capture LUCIA's prediction."""
    if PUTATIVE_EVENT_TAG in observation:
      # Extract the text after the tag
      prediction = observation[
          observation.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
      ]
      # Strip entity name prefix  (e.g. " Lucia: " or " Lucia -- ")
      prefix = ' Lucia'
      if prediction.startswith(prefix):
        remainder = prediction[len(prefix):]
        if remainder.startswith(':'):
          remainder = remainder[1:]
        elif remainder.startswith(' --'):
          remainder = remainder[3:]
        prediction = remainder.strip()
      else:
        prediction = prediction.strip()

      self._current_prediction = prediction
    return ''

  def post_observe(self) -> str:
    return ''

  # ------------------------------------------------------------------
  # State management
  # ------------------------------------------------------------------

  def reset(self) -> None:
    self._step_idx = 0
    self._current_prediction = None
    self._predictions = []

  def get_state(self) -> entity_component.ComponentState:
    return {
        'step_idx': self._step_idx,
        'predictions': self._predictions,
        'decision_points': self._decision_points,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._step_idx = state.get('step_idx', 0)
    self._predictions = state.get('predictions', [])
    self._decision_points = state.get('decision_points', self._decision_points)


# ======================================================================
# Prefab
# ======================================================================

@dataclasses.dataclass
class EgoReplayGameMaster(prefab_lib.Prefab):
  """A prefab GM that replays L1 ego-centric data for LUCIA."""

  description: str = (
      'A game master that replays EgoLife L1 caption data and asks '
      'LUCIA to predict her next action at each decision point.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'ego replay rules',
          'decision_points': [],
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build the ego-replay game master.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An EntityAgentWithLogging acting as the GM.
    """
    agent_name = self.params['name']
    decision_points = self.params.get('decision_points', [])
    all_entity_names = [entity.name for entity in self.entities]

    # Single component instance registered under all 4 standard GM keys
    script_component = EgoReplayScript(decision_points=decision_points)

    next_acting_component = gm_components.next_acting.NextActingInFixedOrder(
        sequence=['Lucia'],
    )

    # Standard GM component keys
    next_acting_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    )
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    terminator_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    resolution_key = (
        gm_components.event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )

    components_of_game_master = {
        next_acting_key: next_acting_component,
        next_action_spec_key: script_component,
        terminator_key: script_component,
        make_observation_key: script_component,
        resolution_key: script_component,
    }

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=list(all_entity_names) if all_entity_names else ['Lucia'],
    )

    game_master_agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_game_master,
        measurements=self.params.get('measurements'),
    )

    return game_master_agent
