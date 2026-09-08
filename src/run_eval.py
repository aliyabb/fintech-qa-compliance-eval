"""全量评测。产出 results/raw_runs.jsonl（原始应答）与 results/metrics.json（指标）。

原始应答全部落盘并提交到仓库，任何人都可以在不配置任何 API 的情况下
用 report.py 重新生成报告，核对每一个数字的来源。
"""
from __future__ import annotations
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pipelines import pipeline_a, pipeline_b, pipeline_c
from evaluate import grade
from retriever import load_corpus

RESULTS = pathlib.Path("results"); RESULTS.mkdir(exist_ok=True)
RAW = RESULTS / "raw_runs.jsonl"
PIPES = {"A_纯LLM": pipeline_a, "B_RAG": pipeline_b, "C_RAG+护栏+工具": pipeline_c}


def main():
    gold = [json.loads(l) for l in open("goldenset/goldenset.jsonl", encoding="utf-8") if l.strip()]
    corpus_text = "\n".join(r["正文"] for r in load_corpus())

    done = set()
    if RAW.exists():
        for l in open(RAW, encoding="utf-8"):
            try:
                d = json.loads(l); done.add((d["qid"], d["pipeline"]))
            except Exception: pass

    total = len(gold) * len(PIPES); n = len(done); t0 = time.time()
    with open(RAW, "a", encoding="utf-8") as f:
        for item in gold:
            for name, fn in PIPES.items():
                if (item["qid"], name) in done: continue
                try:
                    res = fn(item["问题"])
                except Exception as e:
                    res = {"answer": f"__ERROR__ {e}", "contexts": [], "latency_s": 0,
                           "tokens": 0, "route": "-"}
                g = grade(item, res, corpus_text)
                rec = {"qid": item["qid"], "类型": item["类型"], "问题": item["问题"],
                       "pipeline": name, "依据片段": item["依据片段"], "锚点": item["锚点"],
                       **{k: res.get(k) for k in
                          ("answer", "contexts", "context_text", "latency_s",
                           "tokens", "route", "guardrail", "tool_used")},
                       "grade": g}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                n += 1
                el = time.time() - t0
                print(f"[{n}/{total}] {item['qid']} {name} "
                      f"{'OK ' if g['correct'] else 'NG '} "
                      f"eta={(el/max(n-len(done),1))*(total-n)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
