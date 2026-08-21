# -*- coding: utf-8 -*-
"""双段稳健性分析：270组合(win,k) 在前段500期(7730~8229)的样本外命中率 r_prev。
目的：在「段内高命中」候选中挑「样本外不崩」的参数（防选择偏差峰值）。
输出：段内 top15 + 各自 r_prev；段内>=99% 中 r_prev 最高者。
"""
import json
import numpy as np
from engine import load_data
from formulas import feat_list
from hedge_core import hedge_vote, SMOOTH, WIN_GRID, K_GRID

CSV = 'data/fc3d-history.csv'
issues, hh, tt, oo = load_data(CSV)
pool = json.load(open('cache/pool.json', encoding='utf-8'))['pool']
N = len(hh)  # 8730
SEG_IN = 8230      # 段内（池子选出段）
SEG_OUT = 7730     # 前段（样本外）
WIN_MAX = 300      # 前段矩阵需要 win<=300 的历史


def build_segment(seg_start, win_max):
    L0 = seg_start - win_max
    F_ext = np.array([
        feat_list(hh[t - 1], tt[t - 1], oo[t - 1],
                  prev=(hh[t - 2], tt[t - 2], oo[t - 2]))
        for t in range(L0, seg_start + 500)
    ], dtype=np.int16)
    at_ext = np.asarray(oo[L0:seg_start + 500], dtype=np.int16)
    Kp = len(pool)
    pred = np.zeros((Kp, 500 + win_max), dtype=np.int16)
    for i, exp in enumerate(pool):
        cols = np.array([idx for _, idx in exp['terms']], dtype=np.intp)
        coeffs = np.array([c for c, _ in exp['terms']], dtype=np.int16)
        if len(cols) == 1:
            pred[i, :] = (F_ext[:, cols[0]] * coeffs[0] + exp['const']) % 10
        else:
            pred[i, :] = ((F_ext[:, cols] * coeffs[None, :]).sum(axis=1) + exp['const']) % 10
    hit = (pred != at_ext[None, :])
    return hit, pred, L0


def scan_segment(seg_start, hit, pred, L0):
    """返回 {(win,k): hits/500}"""
    res = {}
    for win in WIN_GRID:
        for k in K_GRID:
            h = 0
            for t in range(seg_start, seg_start + 500):
                j = t - L0
                kill, *_ = hedge_vote(win, k, SMOOTH, j, hit, pred)
                if kill != int(oo[t]):
                    h += 1
            res[(win, k)] = h
    return res


hit_in, pred_in, L0_in = build_segment(SEG_IN, WIN_MAX)
hit_out, pred_out, L0_out = build_segment(SEG_OUT, WIN_MAX)
r_in = scan_segment(SEG_IN, hit_in, pred_in, L0_in)
r_out = scan_segment(SEG_OUT, hit_out, pred_out, L0_out)

rows = []
for (win, k), h_in in r_in.items():
    h_out = r_out[(win, k)]
    rows.append({'win': win, 'k': k, 'in': h_in, 'out': h_out,
                 'in_rate': h_in / 5, 'out_rate': h_out / 5})

# 1) 段内 top15
top_in = sorted(rows, key=lambda r: (-r['in'], -r['k'], -r['win']))[:15]
print("=== 段内 top15（含样本外表现） ===")
print(f"{'win':>4} {'k':>4} {'段内':>6} {'样本外':>6}  差")
for r in top_in:
    print(f"{r['win']:>4} {r['k']:>4} {r['in_rate']:6.2f}% {r['out_rate']:6.2f}%  {r['in']-r['out']:+d}")

# 2) 段内 >= 99% 的候选中样本外最高
cand = [r for r in rows if r['in'] >= 495]
cand.sort(key=lambda r: (-r['out'], -r['in'], -r['k'], -r['win']))
print(f"\n=== 段内>=99.0%({len(cand)}个候选) 中 样本外最高 top10 ===")
for r in cand[:10]:
    print(f"win={r['win']:>3} k={r['k']:>3} | 段内 {r['in_rate']:.2f}% | 样本外 {r['out_rate']:.2f}% | 差 {r['in']-r['out']:+d}")

# 3) 段内>=98.5% 候选的样本外分布
cand2 = [r for r in rows if r['in'] >= 492]
cand2.sort(key=lambda r: (-r['out'], -r['in'], -r['k'], -r['win']))
print(f"\n=== 段内>=98.4%({len(cand2)}个候选) 中 样本外最高 top10 ===")
for r in cand2[:10]:
    print(f"win={r['win']:>3} k={r['k']:>3} | 段内 {r['in_rate']:.2f}% | 样本外 {r['out_rate']:.2f}% | 差 {r['in']-r['out']:+d}")

# 全表存 json 备用
with open('cache/oos_scan.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False)
print("\n已写 cache/oos_scan.json")
