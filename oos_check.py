# -*- coding: utf-8 -*-
"""样本外验证：同一专家池(500期段8231~8730选出) + 指定参数，在更早500期段独立回测。
比较 原最优(win=90,k=32) vs 新最优(win=300,k=256) 在 段内(8231~8730) 与 段外(7731~8230) 的表现。
"""
import json
import numpy as np
from engine import load_data
from formulas import feat_list
from hedge_core import hedge_vote, SMOOTH

CSV = 'data/fc3d-history.csv'
issues, hh, tt, oo = load_data(CSV)
pool = json.load(open('cache/pool.json', encoding='utf-8'))['pool']
N = len(hh)  # 8730


def test_segment(seg_start, win, k):
    """段 [seg_start, seg_start+500)，返回命中数/500"""
    seg = seg_start
    L0 = seg - win  # 特征矩阵起点，保证回测首期 j-win=0
    F_ext = np.array([
        feat_list(hh[t - 1], tt[t - 1], oo[t - 1],
                  prev=(hh[t - 2], tt[t - 2], oo[t - 2]))
        for t in range(L0, seg + 500)
    ], dtype=np.int16)
    at_ext = np.asarray(oo[L0:seg + 500], dtype=np.int16)
    Kp = len(pool)
    pred = np.zeros((Kp, 500 + win), dtype=np.int16)
    for i, exp in enumerate(pool):
        cols = np.array([idx for _, idx in exp['terms']], dtype=np.intp)
        coeffs = np.array([c for c, _ in exp['terms']], dtype=np.int16)
        if len(cols) == 1:
            pred[i, :] = (F_ext[:, cols[0]] * coeffs[0] + exp['const']) % 10
        else:
            pred[i, :] = ((F_ext[:, cols] * coeffs[None, :]).sum(axis=1) + exp['const']) % 10
    hit = (pred != at_ext[None, :])
    hits = 0
    for t in range(seg, seg + 500):
        j = t - L0  # ∈ [win, win+499]
        kill, *_ = hedge_vote(win, k, SMOOTH, j, hit, pred)
        if kill != int(oo[t]):
            hits += 1
    return hits


for (win, k, label) in [(90, 32, '原最优'), (300, 256, '新最优'), (300, 399, '全员k=399'), (500, 256, 'win=500')]:
    in_hits = test_segment(8230, win, k)   # 段内（池子选出段 8230~8729）
    out_hits = test_segment(7730, win, k)  # 段外（样本外 7730~8229）
    print(f"{label}: win={win:3d} k={k:3d} | 段内 {in_hits}/500 = {in_hits/5:.2f}% | "
          f"段外(样本外) {out_hits}/500 = {out_hits/5:.2f}% | 差 {in_hits-out_hits:+d}")
