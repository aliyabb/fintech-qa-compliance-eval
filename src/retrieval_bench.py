"""检索质量单独评测。不调用生成模型，因此几秒钟就能跑完。

动机：首轮全量评测中，方案 C 的多数失败并非模型出错——
模型按指令正确地回答了"资料中未找到相关信息"，
是**正确的片段没有进入 top-k**。生成端和检索端的失败必须分开归因，
否则会把检索问题误判成模型能力问题。
"""
from __future__ import annotations
import json, math, re, collections
from retriever import build_index, _cos
from llm import embed

def tokenize(s: str) -> list[str]:
    """中文按字 bigram + 数字/英文按词，无需分词库。"""
    s = re.sub(r"\s+", "", s)
    toks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", s)
    han = re.findall(r"[一-鿿]", s)
    toks += han + ["".join(p) for p in zip(han, han[1:])]
    return toks


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = collections.Counter()
        for d in self.docs:
            for t in set(d): self.df[t] += 1
        self.tf = [collections.Counter(d) for d in self.docs]

    def scores(self, q: str) -> list[float]:
        qt = tokenize(q); out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in qt:
                f = self.tf[i].get(t, 0)
                if not f: continue
                idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def _norm(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def evaluate(k_values=(3, 5, 8), alphas=(0.0, 0.3, 0.5, 0.7, 1.0)):
    corpus, vecs = build_index()
    texts = [f"{r['来源']}｜{r['分组']}｜{r['正文']}" for r in corpus]
    bm = BM25(texts)
    gold = [json.loads(l) for l in open("goldenset/goldenset.jsonl", encoding="utf-8") if l.strip()]
    gold = [g for g in gold if g["类型"] != "必须拒答"]      # 拒答题不依赖检索

    qs = [g["问题"] for g in gold]
    qvecs = embed(qs)

    out = {}
    for alpha in alphas:                                     # alpha=1 纯向量，0 纯BM25
        for k in k_values:
            hit = 0
            for g, qv in zip(gold, qvecs):
                v = _norm([_cos(qv, x) for x in vecs])
                b = _norm(bm.scores(g["问题"]))
                mix = [alpha * a + (1 - alpha) * c for a, c in zip(v, b)]
                top = sorted(range(len(corpus)), key=lambda i: -mix[i])[:k]
                ids = {corpus[i]["id"] for i in top}
                hit += bool(ids & set(g["依据片段"]))
            out[f"alpha={alpha} k={k}"] = round(hit / len(gold), 4)
    return out, len(gold)


if __name__ == "__main__":
    res, n = evaluate()
    print(f"检索命中率（{n} 道非拒答题，命中=依据片段进入 top-k）\n")
    print(f"{'':>12} " + " ".join(f"k={k:<6}" for k in (3, 5, 8)))
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        tag = {0.0: "纯BM25", 1.0: "纯向量"}.get(alpha, f"混合{alpha}")
        row = " ".join(f"{res[f'alpha={alpha} k={k}']*100:>6.1f}%" for k in (3, 5, 8))
        print(f"{tag:>12} {row}")
    import pathlib
    pathlib.Path("results/retrieval_bench.json").write_text(
        json.dumps({"题数": n, "命中率": res}, ensure_ascii=False, indent=2), encoding="utf-8")
