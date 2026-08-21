# -*- coding: utf-8 -*-
"""生成「个位杀一码_1000期回测.html」：读 cache/backtest1000.json，1000期完整回测表 + 分段统计。"""
import json
import os
from gen_site import CSS_TEXT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(BASE_DIR, '个位杀一码_1000期回测.html')

EXTRA_CSS = """
.roll{display:flex;align-items:flex-end;gap:6px;height:64px;padding:4px 2px}
.roll-i{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.roll-bar{width:100%;max-width:34px;border-radius:4px 4px 0 0;background:linear-gradient(180deg,#60a5fa,#2563eb)}
.roll-bar.g{background:linear-gradient(180deg,#34d399,#059669)}
.roll-bar.b{background:linear-gradient(180deg,#f87171,#dc2626)}
.roll-i span{font-size:10px;color:#6b7280;margin-top:3px}
.seg-line{display:flex;justify-content:space-between;font-size:11px;color:#6b7280;margin-top:6px}
"""

def build_html(data):
    s = data['stats']
    rows = data['rows']          # 近期在上
    p = data['params']
    di = data['data_info']
    # 每100期统计（从近期往远算：最近100 / 前100...）
    n = len(rows)
    per100 = []
    for i in range(0, n, 100):
        seg = rows[i:i + 100]
        h = sum(1 for r in seg if r['hit'])
        per100.append({'label': f"{seg[-1]['issue'][:7]}~{seg[0]['issue'][:7]}", 'rate': h / len(seg)})
    def f(x): return f"{x*100:.1f}%"
    def card(t, v, c=''):
        return f'<div class="stat{c}"><div class="v" style="color:{c}">{v}</div><div class="k">{t}</div></div>'
    roll = '<div class="roll">'
    for seg in per100:
        r = seg['rate']
        cls = 'g' if r >= 0.95 else ('b' if r < 0.93 else '')
        hpx = max(6, int(r * 60))
        roll += f'<div class="roll-i"><div class="roll-bar {cls}" style="height:{hpx}px"></div><span>{seg["label"]}</span></div>'
    roll += '</div>'
    segline = ('<div class="seg-line"><span>← 远期 2023年</span><span>每100期命中率 → 近期 2026年</span></div>')
    trs = []
    for r in rows:
        t3 = '·'.join(str(c) if i > 0 else f'<b>{c}</b>' for i, c in enumerate(r['top3']))
        trs.append(
            f'<tr class="{"miss-row" if not r["hit"] else ""}">'
            f'<td class="iss">{r["issue"]}</td><td class="num">{r["num"]}</td>'
            f'<td class="t3">{t3}</td>'
            f'<td class="kill {"hit" if r["hit"] else "miss"}">{r["kill"]}</td>'
            f'<td class="res">{"✅" if r["hit"] else "❌"}</td>'
            f'<td class="fname">{r["fname"]}</td></tr>')
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>福彩3D · 个位杀一码 1000期回测</title>
<style>{CSS_TEXT}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="topbar"><div>
    <div class="t">福彩3D · <b>个位杀一码</b> · 1000期回测</div>
    <div class="sub">段 {di['first']}~{di['last']} · 参数 win={p['win']}/K={p['k']} · 固定快照</div>
  </div></div>

  <div class="card">
    <h3>📊 1000期回测汇总 <span style="color:#9ca3af;font-weight:400">(Hedge win={p['win']}/K={p['k']} walk-forward)</span></h3>
    <div class="stat-grid">
      {card('1000期合计', f(s['all']['rate']), '#2563eb')}
      {card('命中/总数', f"{s['all']['hit']}/{s['all']['total']}")}
      {card('最大连错', s['all']['max_lose'])}
      {card('前500·样本外', f(s['out']['rate']))}
      {card('后500·段内', f(s['in']['rate']))}
      {card('最近100期', f(s['last100']['rate']), '#059669')}
    </div>
    <div class="warn">⚠ 后500期(段内)为选择偏差红利（专家池+参数均在该段选出）；<b>前500期(样本外) {f(s['out']['rate'])} 才是真实水平</b>，1000期合计 {f(s['all']['rate'])} 含偏差上浮。随机基线 90%。</div>
  </div>

  <div class="card">
    <h3>📈 每100期命中率</h3>
    {roll}
    {segline}
  </div>

  <div class="card">
    <h3>📋 1000期逐期真实预测记录 <span style="color:#9ca3af;font-weight:400">(近期在上)</span></h3>
    <div class="tbl-scroll">
      <table>
        <thead><tr><th>期号</th><th>开奖</th><th>票码Top3</th><th>杀码</th><th>结果</th><th>首席专家</th></tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table>
    </div>
  </div>

  <div class="footer"><b>说明</b><br>
    ① 1000期 = 2023277~2026222，参数沿用当前最优 win={p['win']}/K={p['k']}，逐期只用 t-1/t-2 数据（walk-forward 不偷看未来）。<br>
    ② 前500期(2023277~2025226)为<b>样本外</b>：专家池未在该段选出；后500期(2025227~2026222)为<b>段内</b>：专家池与参数均在该段优化，含选择偏差。<br>
    ③ 真实预测水平更接近样本外 <b>{f(s['out']['rate'])}</b>；1000期合计 {f(s['all']['rate'])} 为乐观上界。<b>不构成任何购彩建议</b>。
  </div>
</div>
</body>
</html>
"""

def main():
    with open(os.path.join(BASE_DIR, 'cache', 'backtest1000.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    html = build_html(data)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    s = data['stats']
    print(f"已生成: {OUT_HTML}")
    print(f"1000期合计 {s['all']['rate']*100:.2f}% | 前500样本外 {s['out']['rate']*100:.2f}% | 后500段内 {s['in']['rate']*100:.2f}%")

if __name__ == '__main__':
    main()
