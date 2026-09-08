"""三条产品方案。

A 纯LLM   —— 基线：直接问模型，无任何产品化处理
B RAG     —— 加入检索与引用要求
C RAG+护栏+工具 —— 加入合规意图路由与确定性计算

C 的设计要点见 docs/AI能力边界决策表.md：
金额与利率计算不交给模型，由 calc.py 完成后注入答案。
"""
from __future__ import annotations
import json, re
from llm import generate
from retriever import search, TOP_K
from calc import installment_apr, early_settlement_cost, redemption_fee

SYS_A = "你是民信银行的智能客服助手，回答客户关于产品条款、费率和规则的问题。"

SYS_B = ("你是民信银行的智能客服助手。严格依据下方【参考资料】回答，"
         "并在每个结论句末用[片段ID]标注依据。"
         "参考资料中没有的信息，必须回答「资料中未找到相关信息」，不得推测或补充。"
         "回答简洁，不超过三句话。")

SYS_C = SYS_B + "\n若【计算结果】已给出，直接使用该数值，不要自行计算。"

REFUSAL = (
    "抱歉，我不能为您推荐具体产品、预测收益或判断买卖时点。"
    "民信集团合规要求：未取得投资顾问业务资格不得提供具体投资建议[F029]；"
    "任何自动化工具不得承诺保本或对未来收益作出数值预测[F027]。\n"
    "我可以为您做的是：说明产品的风险等级、费率结构、赎回规则等客观信息，"
    "或协助您理解风险揭示书的内容。您也可以联系持牌投资顾问获取个性化建议。"
)

ROUTER_SYS = (
 "你是一个意图分类器。判断客户问题属于哪一类，只输出JSON，不要解释。\n"
 "类别：\n"
 "  info  —— 咨询客观信息（费率、期限、限额、规则、到账时间）\n"
 "  calc  —— 需要计算具体金额或年化利率\n"
 "  advice—— 请求推荐产品、比较后选一个、判断买卖时点、请求个性化配置\n"
 "  yield —— 请求预测收益、承诺保本、询问能赚多少、要求达到某收益率\n"
 "输出格式：{\"intent\":\"info|calc|advice|yield\"}")


def _ctx(hits: list[dict]) -> str:
    out = []
    for h in hits:
        tag = f"[{h['id']}]（生效{h['生效日期']}"
        tag += f"，已于{h['失效日期']}失效" if "失效日期" in h else ""
        out.append(tag + f"）{h['正文']}")
    return "\n".join(out)


def route(question: str) -> str:
    r = generate(ROUTER_SYS, question, max_tokens=40)
    m = re.search(r'"intent"\s*:\s*"(\w+)"', r["text"])
    return m.group(1) if m else "info"


PARAM_SYS = ("从问题中抽取计算参数，只输出JSON。字段：\n"
 '{"kind":"installment|early|redeem|withdraw","principal":数字,"periods":数字,'
 '"paid_periods":数字,"days":数字,"product":"添利债券A|智选混合C"}\n'
 "缺失字段填 null。installment=分期还款或年化，early=提前结清，"
 "redeem=理财赎回费，withdraw=提现手续费。")

RATES = {3: 0.0055, 6: 0.0058, 12: 0.0060, 24: 0.0062}
TIERS = {"添利债券A": [(7,0.015),(30,0.005),(None,0.0)],
         "智选混合C": [(7,0.015),(None,0.0)]}


def compute(question: str) -> dict | None:
    """确定性计算。参数抽取交给模型，算术一律由 calc.py 完成。"""
    r = generate(PARAM_SYS, question, max_tokens=120)
    try:
        p = json.loads(re.search(r"\{.*\}", r["text"], re.S).group(0))
    except Exception:
        return None
    k = p.get("kind")
    try:
        if k == "installment" and p.get("principal") and p.get("periods"):
            n = int(p["periods"])
            if n not in RATES: return None
            return {"计算结果": installment_apr(float(p["principal"]), n, RATES[n])}
        if k == "early" and p.get("principal") and p.get("periods"):
            n = int(p["periods"])
            return {"计算结果": early_settlement_cost(
                float(p["principal"]), n, RATES.get(n, 0.006),
                int(p.get("paid_periods") or 0))}
        if k == "redeem" and p.get("principal") and p.get("days") is not None:
            prod = p.get("product") or "添利债券A"
            return {"计算结果": redemption_fee(float(p["principal"]), int(p["days"]),
                                            TIERS.get(prod, TIERS["添利债券A"]))}
        if k == "withdraw" and p.get("principal"):
            amt = float(p["principal"])
            return {"计算结果": {"提现金额": amt, "费率": 0.001,
                              "手续费": round(max(amt*0.001, 0.10), 2)}}
    except Exception:
        return None
    return None


def pipeline_a(q: str) -> dict:
    r = generate(SYS_A, q, max_tokens=400)
    return {"answer": r["text"], "contexts": [], "latency_s": r["latency_s"],
            "tokens": r["completion_tokens"], "route": "-"}


def pipeline_b(q: str, k: int = TOP_K) -> dict:
    hits = search(q, k=k)
    r = generate(SYS_B, f"【参考资料】\n{_ctx(hits)}\n\n【客户问题】{q}", max_tokens=400)
    return {"answer": r["text"], "contexts": [h["id"] for h in hits],
            "context_text": _ctx(hits), "latency_s": r["latency_s"],
            "tokens": r["completion_tokens"], "route": "-"}


def pipeline_c(q: str, k: int = TOP_K) -> dict:
    intent = route(q)
    if intent in ("advice", "yield"):
        return {"answer": REFUSAL, "contexts": ["F027", "F029"],
                "context_text": "", "latency_s": 0.0, "tokens": 0,
                "route": intent, "guardrail": True}
    hits = search(q, k=k)
    calc_block = ""
    calc = compute(q) if intent == "calc" else None
    if calc:
        calc_block = "\n【计算结果】（由确定性计算模块给出，直接引用）\n" + \
                     json.dumps(calc["计算结果"], ensure_ascii=False)
    r = generate(SYS_C, f"【参考资料】\n{_ctx(hits)}{calc_block}\n\n【客户问题】{q}",
                 max_tokens=400)
    return {"answer": r["text"], "contexts": [h["id"] for h in hits],
            "context_text": _ctx(hits) + calc_block, "latency_s": r["latency_s"],
            "tokens": r["completion_tokens"], "route": intent,
            "tool_used": bool(calc)}
