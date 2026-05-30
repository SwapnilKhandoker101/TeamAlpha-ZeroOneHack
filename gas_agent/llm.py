"""A thin wrapper over the Featherless (OpenAI-compatible) inference endpoint.

Both LLM roles in this agent — the tag-picker that prepares the Sybilion call and
the explainer that narrates the deterministic decision — talk to Featherless
through this one module. It exists to keep two concerns in a single place:

* graceful degradation — if no Featherless key is configured the helpers raise a
  clear :class:`LLMUnavailable`, so each agent can fall back to a deterministic
  path and the demo always runs off the cache;
* robust JSON parsing — small instruct models like to wrap JSON in ```json
  fences and add a sentence of preamble; :func:`chat_json` strips that and
  returns parsed data instead of making every caller re-implement it.

Neither helper ever makes the hedging decision — that is the job of
:mod:`gas_agent.hedge_policy`. These only prepare inputs and phrase outputs.
"""

from __future__ import annotations

import json

from gas_agent import config


class LLMUnavailable(RuntimeError):
    """Raised when no Featherless key is configured, or a call fails. Callers are
    expected to catch this and fall back to their deterministic path."""


def featherless_available() -> bool:
    return config.have_featherless_key()


def _client():
    if not config.have_featherless_key():
        raise LLMUnavailable("no FEATHERLESS_API_KEY configured")
    # Imported lazily so the dashboard can run off cache without the dependency.
    from openai import OpenAI

    # Featherless serves models serverless, so a cold model can take tens of
    # seconds on the first hit; a generous timeout + retries absorbs that rather
    # than dropping to the fallback narrative mid-demo.
    return OpenAI(
        api_key=config.FEATHERLESS_API_KEY,
        base_url=config.FEATHERLESS_BASE_URL,
        timeout=90.0,
        max_retries=2,
    )


def chat_text(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    """One-shot chat completion returning the assistant's text."""
    client = _client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as error:  # network / auth / model-not-found
        raise LLMUnavailable(str(error)) from error
    return (response.choices[0].message.content or "").strip()


def strip_json_fence(text: str) -> str:
    """Pull the JSON body out of a model reply that may be wrapped in a
    ```json ... ``` fence or padded with prose. Returns the original text when
    there is no fence so a bare JSON object still parses."""
    fence = text.find("```")
    if fence == -1:
        return text.strip()
    start = text.find("\n", fence)
    if start == -1:
        return text.strip()
    end = text.find("```", start)
    body = text[start + 1 : end] if end != -1 else text[start + 1 :]
    return body.strip()


def chat_json(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> dict | list:
    """Like :func:`chat_text` but parses the reply as JSON (fences tolerated)."""
    raw = chat_text(model, system, user, temperature=temperature, max_tokens=max_tokens)
    try:
        return json.loads(strip_json_fence(raw))
    except json.JSONDecodeError as error:
        raise LLMUnavailable(f"model did not return valid JSON: {raw[:200]!r}") from error
