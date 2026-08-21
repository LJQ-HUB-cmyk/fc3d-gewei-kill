# -*- coding: utf-8 -*-
"""
回测1000期验证：当前专家池 + 当前最优参数(win=100,k=200)，walk-forward 逐期真实预测。
1000期 = 前500期(7730~8229, 样本外: 池子未在该段选出) + 后500期(8230~8729, 段内: 池子选出段)。
输出统计 + cache/backtest1000.json（rows 近期在上）。
"""
import json
import sys
import numpy as np
from engine import load_data
from formulas import feat_list
from hedge_core import hedge_vote, _top3_codes, SMOOTH

CSV = 'data/fc3d-history.csv'
issues, hh, tt, oo = load_data(CSV)
pool = json.load(open('cache/pool.json', encoding='utf-8'))['pool']
rj = json.load(open('cache/result.json', encoding='utf-8'))
best = rj['best_scan']
win, k = best['win'], best['k']
N = len(hh)                      # 8730
SEG = 1000
start = N - SEG                  # 7730
L0 = start - win                 # 7630（win 期历史）
print(f"1000期回测: 段 {issues[start]}~{issues[N-1]} | 参数 win={win}, K={k} | 专家池 {len(pool)}")

# 矩阵
F_ext = np.array([
    feat_list(hh[t - 1], tt[t - 1], oo[t - 1],
              prev=(hh[t - 2], tt[t - 2], oo[t - 2]))
    for t in range(L0, N + 1)
], dtype=np.int16)
at_ext = np.concatenate([np.asarray(oo[L0:N], dtype=np.int16), [0]])
Kp = len(pool)
pred = np.zeros((Kp, N - L0 + 1), dtype=np.int16)
for i, exp in enumerate(pool):
    cols = np.array([idx for _, idx in exp['terms']], dtype=np.intp)
    coeffs = np.array([c for c, _ in exp['terms']], dtype=np.int16)
    if len(cols) == 1:
        pred[i, :] = (F_ext[:, cols[0]] * coeffs[0] + exp['const']) % 10
    else:
        pred[i, :] = ((F_ext[:, cols] * coeffs[None, :]).sum(axis=1) + exp['const']) % 10
hit = (pred != at_ext[None, :])

rows = []
for t in range(start, N):
    j = t - L0
    kill, ti, w, votes, top_rate = hedge_vote(win, k, SMOOTH, j, hit, pred)
    rows.append({
        'issue': str(issues[t]), 'num': f"{hh[t]}{tt[t]}{oo[t]}",
        'kill': int(kill), 'hit': bool(kill != int(oo[t])),
        'top3': _top3_codes(int(kill), votes),
        'fname': pool[int(ti[0])]['name'], 'fam': pool[int(ti[0])]['family'],
    })

hits = [r['hit'] for r in rows]


def seg_stats(hs, label):
    n = len(hs)
    hitn = sum(hs)
    max_lose = cl = 0
    for h in hs:
        cl = cl + 1 if not h else 0
        max_lose = max(max_lose, cl)
    cur_w = cur_l = 0
    for h in reversed(hs):
        if h: cur_w += 1
        else: break
    for h in reversed(hs):
        if not h: cur_l += 1
        else: break
    print(f"  {label:>10}: {hitn}/{n} = {hitn/n*100:.2f}% | 最大连错 {max_lose} | 当前连中 {cur_w} / 连错 {cur_l}")
    return {'label': label, 'hit': hitn, 'total': n, 'rate': round(hitn / n, 4),
            'max_lose': max_lose, 'cur_win': cur_w, 'cur_lose': cur_l}


print("\n=== 分段统计 ===")
out_stats = seg_stats(hits[:500], '前500(样本外)')
in_stats = seg_stats(hits[500:], '后500(段内)')
all_stats = seg_stats(hits, '1000期合计')
last100 = seg_stats(hits[-100:], '最近100期')

# 逐100期滚动
print("\n=== 每100期命中率 ===")
for i in range(0, SEG, 100):
    seg = hits[i:i + 100]
    segn = sum(seg)
    segno = issues[start + i]
    segn_o = issues[start + i + 99]
    print(f"  {segno}~{segn_o}: {segn}/100 = {segn/100*100:.2f}%")

out = {
    'params': {'win': win, 'k': k, 'smooth': SMOOTH, 'baseline': 0.9},
    'data_info': {'n_issues': N, 'first': issues[start], 'last': issues[-1]},
    'stats': {'all': all_stats, 'out': out_stats, 'in': in_stats, 'last100': last100},
    'rows': list(reversed(rows)),   # 近期在上
}
with open('cache/backtest1000.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)
print("\n已写 cache/backtest1000.json（rows 近期在上）")
