"""OpenAI-compatible /v1/chat/completions proxy over Tinker SamplingClients.

Serves several checkpoints at once, routed by the request `model` field
(qwen3-8b-base, qwen3-8b-ihrlvr). The eval harness hits this via proxy://
candidates (config.resolve_candidate_transport). Only build_app() is unit-
tested (with fake samplers); the Tinker-backed sampler + main() are exercised
by the parity smoke test (plan Task 10).
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse


class Sampler(Protocol):
    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float, reasoning: bool) -> str: ...


def build_app(samplers: dict[str, Sampler], api_key: str) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request, authorization: str = Header(default="")):
        if authorization.removeprefix("Bearer ").strip() != api_key:
            raise HTTPException(status_code=401, detail="bad proxy key")
        body = await req.json()
        model = body.get("model")
        sampler = samplers.get(model)
        if sampler is None:
            raise HTTPException(status_code=404, detail=f"unknown model {model!r}")
        text = sampler.generate(
            messages=body["messages"],
            max_tokens=body.get("max_tokens", 8000),
            temperature=body.get("temperature", 0.0),
            # The OpenAI SDK merges extra_body into the top-level request JSON
            # (_merge_mappings(json_data, options.extra_json)) before sending,
            # so "reasoning" arrives as a top-level key; there is never an
            # "extra_body" key on the wire.
            reasoning=bool((body.get("reasoning") or {}).get("enabled", False)),
        )

        if not body.get("stream"):
            return {
                "id": "chatcmpl-proxy", "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
            }

        # Streaming: the eval harness (pipeline/generate.py) calls candidates
        # with stream=True and consumes SSE deltas via the OpenAI SDK's
        # SSEDecoder, which yields zero events from a plain JSON body. Sampling
        # already happened above (sync); this generator only frames the
        # already-computed text as a single-delta chunk followed by a
        # finish_reason="stop" chunk and the terminal [DONE]; legal SSE, and
        # all frames arrive immediately so the harness's per-chunk deadline
        # logic is never starved.
        created = int(time.time())

        def _chunk(delta: dict, finish_reason: str | None) -> dict:
            return {
                "id": "chatcmpl-proxy", "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        def _stream():
            yield f"data: {json.dumps(_chunk({'role': 'assistant', 'content': text}, None))}\n\n"
            yield f"data: {json.dumps(_chunk({}, 'stop'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


class TinkerSampler:
    """Wraps a Tinker SamplingClient + a pair of renderers (thinking-on and
    thinking-off) for one checkpoint. Base and ft samplers share the SAME
    renderer pair so only weights differ (fair baseline).

    NOTE: tinker_cookbook bakes "thinking" into which renderer you build
    (e.g. get_renderer("qwen3", tok) vs get_renderer("qwen3_disable_thinking",
    tok)) rather than into a build_generation_prompt(..., thinking=...) kwarg,
    confirmed against the installed tinker_cookbook==0.4.3 renderer
    registry offline (no live server access). Hence two renderers per
    sampler, selected by the `reasoning` flag at call time.
    """

    def __init__(self, sampling_client, renderer, renderer_no_thinking):
        self.client = sampling_client
        self.renderer = renderer
        self.renderer_no_thinking = renderer_no_thinking

    def generate(self, messages, max_tokens, temperature, reasoning) -> str:
        import tinker
        from tinker_cookbook import renderers

        renderer = self.renderer if reasoning else self.renderer_no_thinking
        prompt = renderer.build_generation_prompt(messages)

        # LIVE-VERIFY (Task 10 parity smoke): SamplingParams field names and
        # the sample()->future->.result() blocking shape were confirmed via
        # `inspect.signature` against installed tinker==0.22.7, but never
        # exercised against a live Tinker server. Re-check num_samples=1 is
        # right (vs wanting sample_async in an async context) and that
        # sequences[0] is the correct "first sample" index.
        future = self.client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature),
        )
        result = future.result()
        tokens = result.sequences[0].tokens

        # LIVE-VERIFY (Task 10 parity smoke): parse_response's return shape
        # (Message, ParseTermination) and get_text_content's stripping of
        # <think> blocks were confirmed via source inspection of
        # tinker_cookbook==0.4.3, not against real sampled tokens.
        message, _parse_termination = renderer.parse_response(tokens)
        return renderers.get_text_content(message)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # TINKER_API_KEY et al. live in .env (repo convention, cf. config.py)
    import tinker
    import uvicorn
    from tinker_cookbook import renderers

    api_key = os.environ.get("PROXY_API_KEY", "local-dev")
    service = tinker.ServiceClient()  # reads TINKER_API_KEY
    base_client = service.create_sampling_client(model_path=os.environ["BASE_CHECKPOINT"])
    ft_client = service.create_sampling_client(model_path=os.environ["FT_CHECKPOINT"])

    # LIVE-VERIFY (Task 10 parity smoke): confirm "qwen3" / "qwen3_disable_thinking"
    # are the right renderer names for the Qwen3-8B checkpoints actually in use, and
    # that reusing base_client's tokenizer for the ft renderer is valid (same base
    # model/tokenizer, only LoRA/weights differ).
    tokenizer = base_client.get_tokenizer()
    renderer = renderers.get_renderer("qwen3", tokenizer)
    renderer_no_thinking = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    samplers = {
        "qwen3-8b-base": TinkerSampler(base_client, renderer, renderer_no_thinking),
        "qwen3-8b-ihrlvr": TinkerSampler(ft_client, renderer, renderer_no_thinking),
    }
    uvicorn.run(build_app(samplers, api_key), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
