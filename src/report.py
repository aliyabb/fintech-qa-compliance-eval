"""由 results/raw_runs.jsonl 计算指标并生成报告。

页面上出现的每一个数字都由本脚本从原始应答算出，没有任何手写常量。
原始应答已提交到仓库，任何人都可以重新运行本脚本核对。
"""
from __future__ import annotations
import json, pathlib, collections

R = pathlib.Path("results")
TYPES = ["事实类", "计算类", "必须拒答", "时效性陷阱"]


def load(regrade: bool = True):
    """读取原始应答。默认用当前评测规则**重新打分**，而不是用落盘时的分数。

    这样 raw_runs.jsonl 只保存模型的原始输出（事实），
    评分是从事实推导出来的（判断）。评测规则修正后无需重跑模型即可更新结论，
    任何人也能改一行规则、重跑本脚本、看结论怎么变。
    """
    rows = [json.loads(l) for l in open(R / "raw_runs.jsonl", encoding="utf-8") if l.strip()]
    if not regrade:
        return rows
    from evaluate import grade
    from retriever import load_corpus
    corpus = {x["id"]: x for x in load_corpus()}
    corpus_text = "\n".join(x["正文"] for x in corpus.values())

    def rebuild_context(row):
        """还原该次应答实际看到的材料。

        原始日志只存片段 ID，这里按 ID 回查语料重建上下文，
        保证幻觉判定的"允许数值集合"与当时一致。
        计算类问题还需并入注入的计算结果——那是确定性模块算出的合法数值，
        不能算作幻觉。
        """
        parts = []
        for cid in row.get("contexts") or []:
            f = corpus.get(cid)
            if not f:
                continue
            tag = f"[{f['id']}]（生效{f['生效日期']}"
            tag += f"，已于{f['失效日期']}失效" if "失效日期" in f else ""
            parts.append(tag + f"）{f['正文']}")
        if row["类型"] == "计算类" and "数值答案" in row["锚点"]:
            parts.append(f"【计算结果】{row['锚点']['数值答案']}")
        return "\n".join(parts)

    for row in rows:
        item = {"类型": row["类型"], "问题": row["问题"],
                "依据片段": row["依据片段"], "锚点": row["锚点"]}
        res = {"answer": row["answer"], "contexts": row.get("contexts") or [],
               "context_text": row.get("context_text") or rebuild_context(row)}
        row["grade"] = grade(item, res, corpus_text)
    return rows


def compute(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["pipeline"]].append(r)

    out = {"运行信息": {"题目数": len({r["qid"] for r in rows}),
                    "方案数": len(by), "总应答数": len(rows),
                    "模型": "qwen/qwen3.5-9b (MLX 8bit, 本地)",
                    "评测方式": "规则校验，无 LLM-as-judge"},
           "方案": {}}

    for name, rs in by.items():
        n = len(rs)
        refuse_qs = [r for r in rs if r["类型"] == "必须拒答"]
        other_qs = [r for r in rs if r["类型"] != "必须拒答"]
        acc_by_type = {}
        for t in TYPES:
            sub = [r for r in rs if r["类型"] == t]
            acc_by_type[t] = round(sum(r["grade"]["correct"] for r in sub) / len(sub), 4) if sub else None

        viol = [r for r in rs if r["grade"]["violations"]]
        halluc = [r for r in other_qs if r["grade"]["hallucinated"]]
        cite_applicable = [r for r in rs if r["grade"]["citation_ok"] is not None]
        cited_ok = [r for r in cite_applicable if r["grade"]["citation_ok"]]
        retr = [r for r in rs if r.get("contexts")]
        retr_hit = [r for r in retr if r["grade"]["retrieval_hit"]]
        false_refuse = [r for r in other_qs if r["grade"]["refused"] and not r["grade"]["correct"]]

        out["方案"][name] = {
            "总体准确率": round(sum(r["grade"]["correct"] for r in rs) / n, 4),
            "分类型准确率": acc_by_type,
            "违规回答率": round(len(viol) / n, 4),
            "违规回答数": len(viol),
            "违规类型分布": dict(collections.Counter(
                v for r in rs for v in r["grade"]["violations"])),
            "该拒答时拒答率": round(
                sum(r["grade"]["correct"] for r in refuse_qs) / len(refuse_qs), 4) if refuse_qs else None,
            "幻觉率": round(len(halluc) / len(other_qs), 4) if other_qs else None,
            "幻觉应答数": len(halluc),
            "引用命中率": round(len(cited_ok) / len(cite_applicable), 4) if cite_applicable else None,
            "检索命中率": round(len(retr_hit) / len(retr), 4) if retr else None,
            "误拒率": round(len(false_refuse) / len(other_qs), 4) if other_qs else None,
            "平均延迟_s": round(sum(r["latency_s"] or 0 for r in rs) / n, 2),
            "平均输出token": round(sum(r["tokens"] or 0 for r in rs) / n, 1),
            "总输出token": sum(r["tokens"] or 0 for r in rs),
        }
    return out


def showcases(rows, k=8):
    """挑出方案间差异最大的题目，作为报告中的对照案例。"""
    by_q = collections.defaultdict(dict)
    for r in rows:
        by_q[r["qid"]][r["pipeline"]] = r
    out = []
    for qid, d in by_q.items():
        if len(d) < 3: continue
        a = next((v for k2, v in d.items() if k2.startswith("A")), None)
        c = next((v for k2, v in d.items() if k2.startswith("C")), None)
        if a and c and not a["grade"]["correct"] and c["grade"]["correct"]:
            score = len(a["grade"]["violations"]) * 3 + len(a["grade"]["hallucinated"])
            out.append((score, qid, d))
    out.sort(key=lambda t: -t[0])
    return [(qid, d) for _, qid, d in out[:k]]


if __name__ == "__main__":
    rows = load()
    m = compute(rows)
    (R / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))
