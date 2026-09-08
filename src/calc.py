"""确定性金融计算模块。

设计原则：所有涉及金额与利率的计算一律由本模块完成，不交给大模型。
理由见 docs/AI能力边界决策表.md —— 大模型在多步数值计算上不可靠，
而分期成本的错误直接构成对客户的误导。
"""
from __future__ import annotations


def installment_schedule(principal: float, periods: int, fee_rate: float) -> dict:
    """信用卡账单分期。

    民信银行规则（F003）：手续费按分期本金**全额**计收，每期收取，
    不随剩余本金递减。因此每期还款额固定为 本金/期数 + 本金*费率。
    """
    principal_part = principal / periods
    fee_part = principal * fee_rate
    payment = principal_part + fee_part
    return {
        "本金": round(principal, 2),
        "期数": periods,
        "每期费率": fee_rate,
        "每期还款": round(payment, 2),
        "每期手续费": round(fee_part, 2),
        "手续费合计": round(fee_part * periods, 2),
        "还款总额": round(payment * periods, 2),
        "名义费率之和": round(fee_rate * periods, 6),
    }


def irr(cashflows: list[float], lo: float = -0.9999, hi: float = 10.0,
        tol: float = 1e-12, max_iter: int = 500) -> float:
    """二分法求内部收益率（每期）。cashflows[0] 为负（放款），其余为正（还款）。"""
    def npv(r: float) -> float:
        return sum(cf / (1.0 + r) ** i for i, cf in enumerate(cashflows))

    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        raise ValueError("IRR 不在搜索区间内")
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def installment_apr(principal: float, periods: int, fee_rate: float) -> dict:
    """分期业务的 IRR 口径实际年化利率（F009 / F030 要求的披露口径）。"""
    s = installment_schedule(principal, periods, fee_rate)
    flows = [-principal] + [s["每期还款"]] * periods
    r_period = irr(flows)
    apr_simple = r_period * 12          # 单利年化（监管常用 IRR*12 口径）
    apr_effective = (1 + r_period) ** 12 - 1   # 复利年化
    return {
        **s,
        "每期IRR": round(r_period, 8),
        "实际年化利率_IRR口径": round(apr_simple, 6),
        "实际年化利率_复利": round(apr_effective, 6),
        "倍数_相对名义费率": round(apr_simple / s["名义费率之和"], 4),
    }


def early_settlement_cost(principal: float, periods: int, fee_rate: float,
                          paid_periods: int) -> dict:
    """提前结清总成本。

    民信银行规则（F004）：剩余期数手续费一次性收取、不予减免，
    因此提前结清**不降低**手续费总额。
    """
    total_fee = principal * fee_rate * periods
    return {
        "已还期数": paid_periods,
        "剩余期数": periods - paid_periods,
        "已付手续费": round(principal * fee_rate * paid_periods, 2),
        "结清时须补付手续费": round(principal * fee_rate * (periods - paid_periods), 2),
        "手续费总额": round(total_fee, 2),
        "是否因提前结清而减少": False,
    }


def redemption_fee(amount: float, holding_days: int, tiers: list[tuple]) -> dict:
    """理财赎回费。tiers: [(不足天数上限, 费率), ...]，最后一档用 None 表示无上限。"""
    for upper, rate in tiers:
        if upper is None or holding_days < upper:
            return {"持有天数": holding_days, "适用费率": rate,
                    "赎回费": round(amount * rate, 2),
                    "实际到账": round(amount * (1 - rate), 2)}
    return {}
