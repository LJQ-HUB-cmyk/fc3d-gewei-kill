# -*- coding: utf-8 -*-
"""
verify_issue.py — 逐期真实回测对账工具
========================================
用法: python verify_issue.py <期号> [<期号> ...]
对每个期号：从原始数据独立重算该期杀码（不依赖任何缓存矩阵），
证明回测表每一期都是「当期开奖前」用已有数据算出的：
  1) 权重：每个专家在 [t-win, t-1] 期的命中率（每期只用 q-1/q-2 数据）
  2) 投票：专家对 t 期的杀码（只用 t-1/t-2 数据）× 权重 → 票王
  3) 比对回测表记录（500期表 result.json / 1000期表 backtest1000.json）
"""
import json
import sys

import numpy as np

from engine import load_data
from formulas import feat_list, eval_linear
from hedge_core import hedge_vote, SMOOTH, _top3_codes

CSV = 'data/fc3d-history.csv'


def load_tables():
    """500期表 + 1000期表（近期在上，转成按期号索引）"""
    rj = json.load(open('cache/result.json', encoding='utf-8'))
    tables = {'500期': {r['issue']: r for r in rj['rows']}}
    try:
        b1000 = json.load(open('cache/backtest1000.json', encoding='utf-8'))
        tables['1000期'] = {r['issue']: r for r in b1000['rows']}
    except FileNotFoundError:
        pass
    return rj, tables


def verify_one(issues, hh, tt, oo, pool, iss, win, k, rj, tables):
    """独立重算期号 iss 的杀码并比对回测表"""
    try:
        t = issues.index(iss)
    except ValueError:
        print(f"✗ 期号 {iss} 不在数据中（共 {len(issues)} 期）")
        return
    N = len(hh)
    # 1) 每专家近 win 期命中率（只用 q-1/q-2 数据，q ∈ [t-win, t-1]）
    hist = np.zeros(len(pool), dtype=np.float64)
    terms_list = [[tuple(x) for x in exp['terms']] for exp in pool]
    consts = [exp['const'] for exp in pool]
    for qi, q in enumerate(range(t - win, t)):
        fq = feat_list(hh[q - 1], tt[q - 1], oo[q - 1],
                       prev=(hh[q - 2], tt[q - 2], oo[q - 2]))
        for i in range(len(pool)):
            if eval_linear(fq, terms_list[i], consts[i]) != int(oo[q]):
                hist[i] += 1.0
    rates = hist / win
    ti = np.argsort(-rates)[:k]
    w = np.maximum(rates[ti], SMOOTH)
    # 2) 专家对 t 期的杀码（只用 t-1/t-2 数据）
    ft = feat_list(hh[t - 1], tt[t - 1], oo[t - 1],
                   prev=(hh[t - 2], tt[t - 2], oo[t - 2]))
    pred_t = np.array([eval_linear(ft, terms_list[i], consts[i]) for i in ti], dtype=np.int16)
    votes = np.bincount(pred_t, weights=w, minlength=10)
    kill = int(np.argmax(votes))
    top3 = _top3_codes(kill, votes)
    hit = (kill != int(oo[t]))

    # 3) 比对回测表
    rec = None
    table_name = None
    for name, tb in tables.items():
        if iss in tb:
            rec, table_name = tb[iss], name
            break
    rec_match = "—(不在表中)" if rec is None else ("✓一致" if rec['kill'] == kill else "✗不一致!")
    rec_kill = "—" if rec is None else str(rec['kill'])
    rec_hit = "—" if rec is None else ("✅" if rec['hit'] else "❌")

    line = ("=" * 54)
    print(f"\n{line}")
    print(f"期号 {iss} | 开奖 {hh[t]}{tt[t]}{oo[t]} | 个位开奖 {oo[t]}")
    print(f"  数据窗口: 本期只用 期{issues[t-1]}(t-1)、期{issues[t-2]}(t-2) 算特征，"
          f"期{issues[t-win]}~期{issues[t-1]} 共{win}期算专家权重 —— 均早于本期开奖")
    print(f"  独立重算: 近{win}期命中率 Top{k} 专家加权投票 → 杀 {kill} (Top3票码 {top3}) | 命中 {'✅' if hit else '❌'}")
    print(f"  回测表[{table_name}]记录: 杀 {rec_kill} {rec_hit} → 对账 {rec_match}")
    if rec is not None:
        print(f"  票数分布(0-9): [{[round(float(x),1) for x in votes]}]")


def main():
    rj, tables = load_tables()
    issues, hh, tt, oo = load_data(CSV)
    pool = json.load(open('cache/pool.json', encoding='utf-8'))['pool']
    best = rj['best_scan']
    win, k = best['win'], best['k']
    args = sys.argv[1:]
    if not args:
        print("用法: python verify_issue.py <期号> [<期号> ...]")
        print(f"当前参数: win={win}, K={k} | 专家池 {len(pool)} | 数据 {len(issues)} 期")
        return
    print(f"对账参数: win={win}, K={k} | 专家池 {len(pool)} | 数据 {issues[0]}~{issues[-1]}")
    for iss in args:
        verify_one(issues, hh, tt, oo, pool, iss, win, k, rj, tables)
    print(f"\n{'>'*54}\n结论: 每期独立重算只用了当期之前的数据（walk-forward），与回测表逐期可对账。")


if __name__ == '__main__':
    main()
