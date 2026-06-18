"""DeepSeek language model for Concordia.

Concordia's built-in GptLanguageModel sends reasoning_effort and verbosity
parameters designed for GPT-5, which DeepSeek rejects. This wrapper implements
the LanguageModel interface directly with standard OpenAI SDK.
"""

from collections.abc import Collection, Sequence
from typing import override

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import openai


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class DeepSeekLanguageModel(language_model.LanguageModel):

    def __init__(
        self,
        model_name: str = "deepseek-v4-pro",
        *,
        api_key: str | None = None,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
    ):
        if api_key is None:
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set")
        self._model_name = model_name
        self._measurements = measurements
        self._channel = channel
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = 0.7,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        del terminators, top_k

        messages = [
            {
                "role": "system",
                "content": (
                    "You always continue input provided by the user and you "
                    "never repeat what the user already said. 请用中文回答。"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
        )

        result = response.choices[0].message.content
        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel, {"raw_text_length": len(result)}
            )
        return result

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        augmented = (
            prompt
            + "\nRespond EXACTLY with one of the following strings:\n"
            + "\n".join(responses)
            + "."
        )

        for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            answer = self.sample_text(augmented, temperature=0.1, seed=seed)
            answer = answer.strip()
            for idx, resp in enumerate(responses):
                if resp in answer or answer in resp:
                    if self._measurements is not None:
                        self._measurements.publish_datum(
                            self._channel, {"choices_calls": attempts}
                        )
                    return idx, responses[idx], {}
            try:
                idx = responses.index(answer)
                return idx, responses[idx], {}
            except ValueError:
                continue

        return 0, responses[0], {}
