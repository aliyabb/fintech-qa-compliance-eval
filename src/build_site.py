"""生成 docs/index.html。所有数字从 results/metrics.json 读取。"""
from __future__ import annotations
import json, pathlib, html
from report import load, compute, showcases

R = pathlib.Path("results"); D = pathlib.Path("docs"); D.mkdir(exist_ok=True)
PIPE_ORDER = ["A_纯LLM", "B_RAG", "C_RAG+护栏+工具"]
LABEL = {"A_纯LLM": "A · 纯 LLM", "B_RAG": "B · RAG",
         "C_RAG+护栏+工具": "C · RAG + 合规护栏 + 计算工具"}

CSS = (pathlib.Path(__file__).parent / "style.css").read_text(encoding="utf-8")


def pct(v):
    return "—" if v is None else f"{v*100:.1f}%"


def build():
    rows = load(); m = compute(rows)
    P = m["方案"]; info = m["运行信息"]
    a, c = P["A_纯LLM"], P["C_RAG+护栏+工具"]

    def kpi(n, label, desc, cls=""):
        return (f'<div class="kpi"><div class="n {cls}">{n}</div>'
                f'<div class="l">{label}</div><div class="d">{desc}</div></div>')

    kpis = "".join([
        kpi(pct(a["违规回答率"]), "违规回答率 · 方案A（纯LLM）",
            "给出推荐、收益预测或保本表述的应答占比", "bad"),
        kpi(pct(c["违规回答率"]), "违规回答率 · 方案C", "加入合规护栏后", "good"),
        kpi(pct(a["幻觉率"]), "幻觉率 · 方案A",
            "答案含依据材料中不存在的数值", "bad"),
        kpi(pct(c["幻觉率"]), "幻觉率 · 方案C", "加入检索与引用约束后", "good"),
    ])

    # 指标总表
    metrics = [("总体准确率","总体准确率",False),("事实类","分类型准确率.事实类",False),
               ("计算类","分类型准确率.计算类",False),("时效性陷阱","分类型准确率.时效性陷阱",False),
               ("该拒答时拒答率","该拒答时拒答率",False),("违规回答率","违规回答率",True),
               ("幻觉率","幻觉率",True),("误拒率","误拒率",True),
               ("引用命中率","引用命中率",False),("检索命中率","检索命中率",False)]
    head = "".join(f"<th>{LABEL[p]}</th>" for p in PIPE_ORDER)
    body = ""
    for label, path, lower_better in metrics:
        cells = ""
        vals = []
        for p in PIPE_ORDER:
            d = P[p]
            for k in path.split("."): d = d.get(k) if isinstance(d, dict) else None
            vals.append(d)
        best = None
        nn = [v for v in vals if v is not None]
        if nn: best = min(nn) if lower_better else max(nn)
        for v in vals:
            cls = "good" if (v is not None and v == best) else ""
            cells += f'<td class="num {cls}">{pct(v)}</td>'
        body += f"<tr><td>{label}{' ↓越低越好' if lower_better else ''}</td>{cells}</tr>"

    for label, key in [("平均延迟","平均延迟_s"),("平均输出token","平均输出token")]:
        cells = "".join(f'<td class="num">{P[p][key]}</td>' for p in PIPE_ORDER)
        body += f"<tr><td>{label}</td>{cells}</tr>"

    # 违规类型分布
    vt = ""
    for p in PIPE_ORDER:
        dist = P[p]["违规类型分布"]
        s = "、".join(f"{k} {v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])) or "无"
        vt += f"<tr><td>{LABEL[p]}</td><td>{P[p]['违规回答数']}</td><td>{s}</td></tr>"

    # 对照案例
    cases = ""
    for qid, d in showcases(rows, 6):
        ra = next(v for k, v in d.items() if k.startswith("A"))
        rc = next(v for k, v in d.items() if k.startswith("C"))
        gv = ra["grade"]
        flags = []
        if gv["violations"]: flags.append(f'<span class="bad">违规：{"、".join(gv["violations"])}</span>')
        if gv["hallucinated"]: flags.append(f'<span class="bad">幻觉数值：{", ".join(str(x) for x in gv["hallucinated"][:6])}</span>')
        cases += f"""<div class="case"><div class="q">{html.escape(ra['问题'])}</div>
<div class="meta">{ra['类型']} · {qid} · 依据 {'/'.join(ra['依据片段'])} · {' · '.join(flags)}</div>
<div class="ans a"><span class="tag">方案 A（纯 LLM）</span>{html.escape(ra['answer'][:300])}…</div>
<div class="ans c"><span class="tag">方案 C（RAG + 护栏 + 计算工具）</span>{html.escape(rc['answer'][:300])}</div></div>"""

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>民信金融问答助手 · 幻觉与合规评测台</title><style>{CSS}</style></head><body>
<div class="wrap">
<header>
<h1>金融问答助手：幻觉率与违规回答率评测台</h1>
<p class="sub">同一批 {info['题目数']} 道题、同一个本地模型，对比三种产品方案。
衡量的不是"模型聪不聪明"，而是<strong>产品化处理能把一个不可靠的模型修到什么程度</strong>。</p>
<div class="tags"><span>消费金融</span><span>RAG</span><span>合规护栏</span>
<span>确定性计算工具</span><span>规则化评测</span><span>本地私有化部署</span></div>
<p class="lede" style="margin-top:18px"><a href="demo.html"><strong>→ 先看产品演示</strong></a>
　用户实际会看到的界面，以及"没有产品化处理会怎样"的逐条对照。本页是它背后的证据。</p>
</header>

<h2>核心结论</h2>
<p class="lede">下面四个数字来自一次完整运行，原始应答已提交至仓库 <code>results/raw_runs.jsonl</code>。</p>
<div class="kpis">{kpis}</div>

<h2>完整指标</h2>
<div class="scroll"><table><thead><tr><th>指标</th>{head}</tr></thead><tbody>{body}</tbody></table></div>

<h3>违规回答的构成</h3>
<p class="lede">违规回答率按<strong>内容</strong>判定，不按是否说了免责声明。
实测中模型会先声明"我无法提供投资建议"，随后照样给出推荐与收益数字——
形式上拒绝、实质上违规。这个判定口径的修正是本项目最重要的一处产品判断。</p>
<div class="scroll"><table><thead><tr><th>方案</th><th>违规应答数</th><th>违规类型分布</th></tr></thead>
<tbody>{vt}</tbody></table></div>

<h2>对照案例</h2>
<p class="lede">从方案 A 失败、方案 C 通过的题目中，按严重度自动挑选。未经人工筛选。</p>
{cases}

<h2>评测方法与局限</h2>
<div class="note">
<strong>为什么不用 LLM-as-judge。</strong>本项目全程本地运行，可用的最强模型就是被测模型本身。
用被测模型给自己打分存在自我偏好偏差，9B 模型识别幻觉的能力也不可靠。
因此改为在<strong>出题时</strong>就为每道题定义可机器校验的锚点：
事实类校验具体数值、计算类比对 <code>calc.py</code> 的精确结果、
时效性陷阱要求命中新值且不含旧值、必须拒答类做内容级违规检测。整个评测是确定性的。
</div>
<div class="note">
<strong>语料是合成的，这是有意的。</strong>全部条款来自虚构的「民信银行 / 民信理财 / 民信支付」，
不对应任何真实机构。合成语料保证模型不可能在预训练中见过答案，
因此方案 A 的失败是真实的、RAG 的增益可以干净归因；
也使"旧值 / 新值 + 生效日期"的时效性陷阱可以被有意构造而非碰运气。
</div>
<div class="note">
<strong>已知局限。</strong>（1）规则检测存在假阴性，措辞迂回的违规可能漏检；
（2）单模型、单次运行，未做多次重复与置信区间；
（3）合成语料的分布与真实产品文档不同，绝对数值不可外推，
方案间的<strong>相对差异</strong>才是本报告的结论。
</div>

<footer>
模型 {info['模型']} · {info['方案数']} 个方案 × {info['题目数']} 道题 = {info['总应答数']} 次应答 ·
评测方式：{info['评测方式']}<br>
本页所有数字由 <code>src/build_site.py</code> 从 <code>results/metrics.json</code> 生成，无手写常量。
</footer>
</div></body></html>"""
    (D / "index.html").write_text(page, encoding="utf-8")
    (R / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"docs/index.html 已生成（{len(page)} 字节）")


def inject_readme(m):
    """把关键指标写进 README 的标记区间。README 里同样没有手写数字。"""
    P, info = m["方案"], m["运行信息"]
    a, c = P["A_纯LLM"], P["C_RAG+护栏+工具"]
    hdr = "".join(f"| {LABEL[p]} " for p in PIPE_ORDER)
    sep = "|---" * (len(PIPE_ORDER) + 1) + "|"

    def line(label, path, lower=True):
        cells = ""
        for p in PIPE_ORDER:
            d = P[p]
            for k in path.split("."):
                d = d.get(k) if isinstance(d, dict) else None
            cells += f"| {pct(d)} "
        arrow = " ↓" if lower else ""
        return f"| **{label}**{arrow} {cells}|\n"

    block = f"""## 核心结果

同一批 {info['题目数']} 道题、同一个本地模型（{info['模型']}），{info['总应答数']} 次应答。

| 指标 {hdr}|
{sep}
{line('违规回答率', '违规回答率')}{line('幻觉率', '幻觉率')}\
{line('该拒答时拒答率', '该拒答时拒答率', lower=False)}\
{line('计算类准确率', '分类型准确率.计算类', lower=False)}\
{line('时效性陷阱通过率', '分类型准确率.时效性陷阱', lower=False)}\
{line('总体准确率', '总体准确率', lower=False)}

从 **A → C**：违规回答率 {pct(a['违规回答率'])} → **{pct(c['违规回答率'])}**，
幻觉率 {pct(a['幻觉率'])} → **{pct(c['幻觉率'])}**，
计算类准确率 {pct(a['分类型准确率']['计算类'])} → **{pct(c['分类型准确率']['计算类'])}**。

平均延迟 A {a['平均延迟_s']}s vs C {c['平均延迟_s']}s —— 加了检索反而**更快**：
无依据时模型会大段铺陈，有依据的回答短而准。**幻觉是要花钱的。**

> 表中每个数字由 `src/build_site.py` 从 `results/raw_runs.jsonl` 重新计算得到。"""

    rp = pathlib.Path("README.md")
    t = rp.read_text(encoding="utf-8")
    a0, b0 = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
    t = t[:t.index(a0) + len(a0)] + "\n" + block + "\n" + t[t.index(b0):]
    rp.write_text(t, encoding="utf-8")
    print("README 指标区已更新")


if __name__ == "__main__":
    build()
    inject_readme(json.loads((R / "metrics.json").read_text(encoding="utf-8")))
