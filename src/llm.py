"""LM Studio 客户端。

为什么用 /v1/completions 而不是 /v1/chat/completions：
Qwen3.5 是推理型模型，经 chat 接口调用时全部 token 进入 reasoning 段，
content 为空（实测 1199 token / 45 秒仍未产出答案）。
LM Studio 不透传 chat_template_kwargs，无法关闭思考模式。
故手工拼接对话模板，并预填空的 <think></think> 块跳过推理段。
实测：同一问题 4 秒产出答案，约 25 tok/s。
"""
from __future__ import annotations
import json, time, urllib.request

BASE = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3.5-9b"


def _post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def build_prompt(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def generate(system: str, user: str, max_tokens: int = 600,
             temperature: float = 0.0) -> dict:
    t0 = time.time()
    d = _post("/completions", {
        "model": MODEL, "prompt": build_prompt(system, user),
        "max_tokens": max_tokens, "temperature": temperature,
        "stop": ["<|im_end|>"],
    })
    text = d["choices"][0]["text"].strip()
    usage = d.get("usage", {})
    return {
        "text": text,
        "latency_s": round(time.time() - t0, 2),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": d["choices"][0].get("finish_reason"),
    }


def embed(texts: list[str]) -> list[list[float]]:
    d = _post("/embeddings", {"model": "text-embedding-nomic-embed-text-v1.5",
                              "input": texts})
    return [x["embedding"] for x in d["data"]]
