"""生成 docs/demo.html —— 产品演示页（民信智能助手）。

与 index.html（评测报告）的分工：
  index.html  给出证据：三种方案在 100 道题上的表现
  demo.html   给出产品：用户实际会看到的界面，以及"没有产品化处理会怎样"的对照

所有问答都是评测运行中的**真实记录**（results/raw_runs.jsonl），不是手写脚本。
"""
from __future__ import annotations
import json, pathlib, html, collections
from report import load
from evaluate import hallucinated_numbers, numbers
from retriever import load_corpus

D = pathlib.Path("docs"); D.mkdir(exist_ok=True)
ORDER = ["事实类", "计算类", "时效性陷阱", "必须拒答"]
TYPE_LABEL = {"事实类": "条款咨询", "计算类": "费用测算",
              "时效性陷阱": "条款时效", "必须拒答": "越界请求"}


def mark_hallucinations(text: str, bad: list[float]) -> str:
    """把幻觉数值在原文里标出来。让用户直接看见问题，而不是读一个比率。"""
    out = html.escape(text)
    for v in sorted(bad, key=lambda x: -abs(x)):
        for form in {f"{v:g}", f"{v:.2f}".rstrip("0").rstrip("."), f"{int(v)}" if v == int(v) else ""}:
            if form and form in out:
                out = out.replace(form, f'<mark>{form}</mark>', 1)
                break
    return out.replace("\n", "<br>")


def build():
    rows = load()
    corpus = {c["id"]: c for c in load_corpus()}
    by_q = collections.defaultdict(dict)
    for r in rows:
        by_q[r["qid"]][r["pipeline"][0]] = r

    items = []
    for qid, d in sorted(by_q.items()):
        if "C" not in d or "A" not in d:
            continue
        c, a = d["C"], d["A"]
        ctx = c.get("context_text") or ""
        bad = sorted(hallucinated_numbers(a["answer"], a["问题"], "\n".join(
            x["正文"] for x in corpus.values())))
        items.append({
            "qid": qid, "type": c["类型"], "q": c["问题"],
            "c": c["answer"], "a": a["answer"],
            "route": c.get("route"), "guard": bool(c.get("guardrail")),
            "tool": bool(c.get("tool_used")),
            "cites": sorted(set(__import__("re").findall(r"F\d{3}", c["answer"]))),
            "ctx": [{"id": i, "text": corpus[i]["正文"], "date": corpus[i]["生效日期"]}
                    for i in (c.get("contexts") or []) if i in corpus],
            "a_bad": bad,
            "a_marked": mark_hallucinations(a["answer"], bad),
            "lat_c": c["latency_s"], "lat_a": a["latency_s"],
        })

    picked, seen = [], collections.Counter()
    for t in ORDER:
        for it in items:
            if it["type"] == t and seen[t] < 6:
                picked.append(it); seen[t] += 1
    (D / "demo_data.json").write_text(json.dumps(picked, ensure_ascii=False), encoding="utf-8")

    css = (pathlib.Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    extra = """
.app{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--card);margin:18px 0}
.appbar{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--line);
 background:color-mix(in srgb,var(--accent) 8%,transparent)}
.logo{width:26px;height:26px;border-radius:7px;background:var(--accent);color:#fff;display:grid;
 place-items:center;font-weight:700;font-size:13px}
.appname{font-weight:640;font-size:14px}
.appsub{font-size:11.5px;color:var(--muted);margin-left:auto}
.panes{display:grid;grid-template-columns:250px 1fr;min-height:430px}
@media(max-width:760px){.panes{grid-template-columns:1fr}}
.qlist{border-right:1px solid var(--line);overflow-y:auto;max-height:530px}
.qgroup{font-size:11px;color:var(--muted);padding:11px 14px 5px;letter-spacing:.5px}
.qitem{padding:9px 14px;font-size:12.5px;cursor:pointer;border-left:2px solid transparent;line-height:1.5}
.qitem:hover{background:var(--code)}
.qitem[aria-selected=true]{background:var(--code);border-left-color:var(--accent);font-weight:600}
.chat{padding:18px 20px}
.bubble-u{background:var(--accent);color:#fff;border-radius:14px 14px 4px 14px;padding:9px 14px;
 display:inline-block;max-width:82%;font-size:13.5px;float:right;clear:both;margin-bottom:14px}
.bubble-b{background:var(--code);border-radius:14px 14px 14px 4px;padding:12px 15px;
 display:inline-block;max-width:88%;font-size:13.5px;clear:both;line-height:1.75}
.clear{clear:both}
.chips{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:11px;border:1px solid var(--line);background:var(--bg);border-radius:12px;
 padding:2px 9px;cursor:pointer;color:var(--muted)}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.src{margin-top:9px;font-size:12px;color:var(--muted);border-left:2px solid var(--line);
 padding-left:11px;display:none}
.src.on{display:block}
.badge{display:inline-block;font-size:10.5px;border-radius:5px;padding:1.5px 7px;margin-right:5px}
.b-guard{background:var(--good);color:#fff}.b-tool{background:var(--accent);color:#fff}
.b-route{background:var(--code);color:var(--muted)}
.toggle{margin:16px 0 0;padding-top:14px;border-top:1px dashed var(--line)}
.toggle button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:8px;
 padding:7px 13px;font-size:12.5px;cursor:pointer;font-family:inherit}
.toggle button:hover{border-color:var(--bad);color:var(--bad)}
.before{display:none;margin-top:13px}
.before.on{display:block}
.before .bubble-b{background:color-mix(in srgb,var(--bad) 9%,transparent);
 border:1px solid color-mix(in srgb,var(--bad) 28%,transparent)}
mark{background:color-mix(in srgb,var(--bad) 32%,transparent);color:inherit;
 padding:0 3px;border-radius:3px;font-weight:600}
.warnline{font-size:12px;color:var(--bad);margin-top:9px}
"""
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>民信智能助手 · 产品演示</title><style>{css}{extra}</style></head><body><div class="wrap">
<header>
<h1>民信智能助手</h1>
<p class="sub">消费金融场景的条款问答助手。左边选一个问题，右边是用户实际会看到的回答——
带依据引用、会拒绝越界请求、金额由计算模块给出。</p>
<div class="tags"><span>产品演示</span><span>真实应答回放</span>
<span><a href="index.html">→ 背后的评测报告</a></span></div>
</header>

<div class="app">
  <div class="appbar"><div class="logo">民</div>
    <div><div class="appname">民信智能助手</div></div>
    <div class="appsub">虚构机构 · 演示用</div></div>
  <div class="panes">
    <div class="qlist" id="qlist"></div>
    <div class="chat" id="chat"></div>
  </div>
</div>

<h2>这个演示想说明什么</h2>
<p class="lede">每条回答下面都有「看看没有产品化处理会怎样」的开关。
点开后是<strong>同一个模型、同一个问题</strong>在没有检索、没有护栏、没有计算工具时的回答，
其中语料里根本不存在的数值会被标红。</p>
<div class="note">这不是为了证明模型不行。恰恰相反——
它说明 <strong>AI 产品的价值大部分不在模型里，而在模型外面那层设计里</strong>：
检索什么、拒绝什么、哪些事交给代码。
这层设计好不好，只能靠测量来判断，所以才有了<a href="index.html">评测报告</a>。</div>

<footer>问答均为评测运行中的真实记录（<code>results/raw_runs.jsonl</code>），非手写脚本。
本页由 <code>src/build_demo.py</code> 生成。</footer>
</div>
<script>
fetch("demo_data.json").then(r=>r.json()).then(D=>{{
  const ql=document.getElementById("qlist"), chat=document.getElementById("chat");
  const LBL={json.dumps(TYPE_LABEL, ensure_ascii=False)};
  let cur=null, html_="";
  {json.dumps(ORDER, ensure_ascii=False)}.forEach(t=>{{
    const g=D.filter(x=>x.type===t); if(!g.length) return;
    html_+='<div class="qgroup">'+LBL[t]+'</div>';
    g.forEach(x=>{{ html_+='<div class="qitem" data-q="'+x.qid+'">'+x.q+'</div>'; }});
  }});
  ql.innerHTML=html_;

  function show(qid){{
    const x=D.find(y=>y.qid===qid); if(!x) return; cur=qid;
    [...ql.children].forEach(e=>e.setAttribute("aria-selected", e.dataset.q===qid));
    const badges=(x.guard?'<span class="badge b-guard">合规护栏拦截</span>':'')
      +(x.tool?'<span class="badge b-tool">调用计算模块</span>':'')
      +(x.route&&x.route!=='-'?'<span class="badge b-route">意图 '+x.route+'</span>':'');
    const chips=x.ctx.map(c=>'<span class="chip" data-src="'+c.id+'">'+c.id+' · 生效'+c.date+'</span>').join("");
    const srcs=x.ctx.map(c=>'<div class="src" id="src-'+c.id+'">['+c.id+'] '+c.text+'</div>').join("");
    chat.innerHTML='<div class="bubble-u">'+x.q+'</div><div class="clear"></div>'
      +'<div class="bubble-b">'+badges+'<br>'+x.c.replace(/\\n/g,"<br>")+'</div><div class="clear"></div>'
      +'<div class="chips">'+chips+'</div>'+srcs
      +'<div class="toggle"><button id="tg">看看没有产品化处理会怎样 ↓</button>'
      +'<div class="before" id="bf"><div class="bubble-b">'+x.a_marked+'</div><div class="clear"></div>'
      +(x.a_bad.length?'<div class="warnline">标红处为语料中不存在的数值 —— 模型自行编造。'
        +'本条共 '+x.a_bad.length+' 处。</div>':'')
      +'</div></div>';
    document.getElementById("tg").onclick=e=>{{
      const b=document.getElementById("bf"); b.classList.toggle("on");
      e.target.textContent=b.classList.contains("on")?"收起 ↑":"看看没有产品化处理会怎样 ↓";
    }};
    chat.querySelectorAll(".chip").forEach(c=>c.onclick=()=>
      document.getElementById("src-"+c.dataset.src).classList.toggle("on"));
  }}
  ql.addEventListener("click",e=>{{const i=e.target.closest(".qitem"); if(i) show(i.dataset.q);}});
  show(D[0].qid);
}});
</script></body></html>"""
    (D / "demo.html").write_text(page, encoding="utf-8")
    print(f"docs/demo.html 已生成（{len(picked)} 个演示问答）")


if __name__ == "__main__":
    build()
