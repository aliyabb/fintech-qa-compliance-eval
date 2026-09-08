"""混合检索：BM25 + 向量，全程离线。

配置依据 —— 实测（src/retrieval_bench.py，75 道非拒答题）：

           k=3     k=5     k=8
  纯BM25   100.0%  100.0%  100.0%
  混合0.3   98.7%  100.0%  100.0%
  纯向量    57.3%   72.0%   88.0%

纯向量检索只有 57.3% 命中率，是首轮评测中方案 C 失败的主因——
模型按指令正确回答了"资料中未找到相关信息"，错在片段没被检索到。
原因：embedding 模型以英文为主，而金融条款短、密集堆叠数字与专有名词，
这类文本上词面匹配显著强于语义匹配。

最终采用 alpha=0.3、k=5（命中率 100%）：以 BM25 为主，
保留三成向量权重以应对换一种说法的提问。
"""
from __future__ import annotations
import collections, json, math, pathlib, re
from llm import embed

ALPHA = 0.3   # 向量权重；1-ALPHA 为 BM25 权重
TOP_K = 5

CORPUS = pathlib.Path("corpus/corpus.jsonl")
CACHE = pathlib.Path("corpus/.embeddings.json")


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _text_for_embedding(r: dict) -> str:
    return f"{r['来源']}｜{r['分组']}｜生效{r['生效日期']}｜{r['正文']}"


def build_index(force: bool = False) -> tuple[list[dict], list[list[float]]]:
    corpus = load_corpus()
    if CACHE.exists() and not force:
        vecs = json.loads(CACHE.read_text())
        if len(vecs) == len(corpus):
            return corpus, vecs
    vecs = embed([_text_for_embedding(r) for r in corpus])
    CACHE.write_text(json.dumps(vecs))
    return corpus, vecs


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def tokenize(s: str) -> list[str]:
    """中文按字与字 bigram，数字/英文按词。避免引入分词依赖。"""
    s = re.sub(r"\s+", "", s)
    toks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", s)
    han = re.findall(r"[一-鿿]", s)
    return toks + han + ["".join(p) for p in zip(han, han[1:])]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = collections.Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.tf = [collections.Counter(d) for d in self.docs]

    def scores(self, q: str) -> list[float]:
        qt, out = tokenize(q), []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in qt:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


_BM25 = None


def _bm25() -> BM25:
    global _BM25
    if _BM25 is None:
        _BM25 = BM25([_text_for_embedding(r) for r in load_corpus()])
    return _BM25


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def search(query: str, k: int = TOP_K, alpha: float = ALPHA) -> list[dict]:
    corpus, vecs = build_index()
    qv = embed([query])[0]
    v = _minmax([_cos(qv, x) for x in vecs])
    b = _minmax(_bm25().scores(query))
    mix = [alpha * a + (1 - alpha) * c for a, c in zip(v, b)]
    order = sorted(range(len(corpus)), key=lambda i: -mix[i])[:k]
    return [{**corpus[i], "score": round(mix[i], 4)} for i in order]
