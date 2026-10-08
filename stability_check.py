"""ACIC 2023: метод «мешочков + Монте-Карло» против известных подходов.
Положите training_sample.csv и testing_sample.csv рядом со скриптом (или поправьте пути)."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
import sys; rng = np.random.default_rng(int(sys.argv[1]))
W = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]; WS = [f'{w:.1f}' for w in W]
H = 5                                      # горизонт прогноза (шаги 0..4 в test = недели 95..99)

tr = pd.read_csv('training_sample.csv').sort_values(['unitID', 'weekID']).reset_index(drop=True)
te = pd.read_csv('testing_sample.csv').sort_values(['unitID', 'step']).reset_index(drop=True)
for c in ['C1', 'C2', 'C3']: tr[c + '_code'] = tr[c].astype('category').cat.codes
units = tr.unitID.unique(); T = tr.weekID.max()

# ---------- 1. общий прогноз уровня (GBM, прямой многошаговый) ----------
g = tr.groupby('unitID')
for l in range(4): tr[f'y_lag{l}'] = g.outcome.shift(l); tr[f'w_lag{l}'] = g.treatment.shift(l)
rows = []
for k in range(1, H + 1):
    d = tr.copy(); d['k'] = k
    d['target'] = g.outcome.shift(-k); d['w_future'] = g.treatment.shift(-k)
    rows.append(d)
D = pd.concat(rows).dropna(subset=['y_lag3'])
F = [f'y_lag{l}' for l in range(4)] + [f'w_lag{l}' for l in range(4)] + \
    ['k', 'w_future', 'X1', 'X2', 'X3', 'C1_code', 'C3_code', 'weekID']
fit = D.dropna(subset=['target'])
gbm = HGBR(max_iter=400, learning_rate=0.05).fit(fit[F], fit.target)
last = tr[tr.weekID == T].set_index('unitID').loc[units]
def gbm_pred(w):
    out = []
    for k in range(1, H + 1):
        x = last.copy(); x['k'] = k; x['w_future'] = w
        out.append(gbm.predict(x[F]))
    return np.stack(out, 1)                 # units × H
P_gbm = {w: gbm_pred(w) for w in W}

# ---------- 2. ваш метод: мешочки + Монте-Карло ----------
# мешочек юнита = его исходы в недели без скидки (treatment = 0)
bags = tr[tr.treatment == 0].groupby('unitID').outcome.apply(np.array).loc[units]
lens = bags.apply(len).values; offs = np.r_[0, np.cumsum(lens)[:-1]]; flat = np.concatenate(bags.values)
uidx = pd.Series(np.arange(len(units)), index=units)
tr['u'] = tr.unitID.map(uidx).values

# сегменты (заранее): C1, C3, X1 квартиль, X3 (доля недель с X3=1), уровень продаж квартиль
ust = tr.groupby('u').agg(C1=('C1', 'first'), C3=('C3', 'first'), X1=('X1', 'first'),
                          X3=('X3', 'mean'), lvl=('outcome', 'mean'))
seg = pd.DataFrame({'C1': ust.C1, 'C3': ust.C3,
                    'X1q': pd.qcut(ust.X1, 4, labels=False), 'X3': (ust.X3 > 0).astype(int),
                    'lvlq': pd.qcut(ust.lvl, 4, labels=False)})
seg_codes = {c: seg[c].astype('category').cat.codes.values for c in seg}

def mc_effects(n_iter=200):
    """Для каждого сегмента и уровня скидки w: (Σ после / Σ вытащенное)_w ÷ (то же)_контроль.
    Вытаскивание: обратное преобразование — U~[0,1) → элемент мешочка с индексом floor(U·len)."""
    u, y, w = tr.u.values, tr.outcome.values.astype(float), tr.treatment.values
    res = {c: {ww: [] for ww in W[1:]} for c in seg}
    dif = {c: {ww: [] for ww in W[1:]} for c in seg}
    for _ in range(n_iter):
        b = rng.integers(0, len(u), len(u))                       # бутстрэп недель-юнитов
        ub, yb, wb = u[b], y[b], w[b]
        drawn = flat[offs[ub] + (rng.random(len(ub)) * lens[ub]).astype(int)]
        for c in seg:
            s = seg_codes[c][ub]; ns = s.max() + 1
            def sums(mask):
                return (np.bincount(s[mask], yb[mask], ns), np.bincount(s[mask], drawn[mask], ns),
                        np.bincount(s[mask], minlength=ns))
            yc, dc, nc = sums(wb == 0)
            for ww in W[1:]:
                ym, dm, nm = sums(np.isclose(wb, ww))
                with np.errstate(all='ignore'):
                    res[c][ww].append((ym / dm) / (yc / dc))
                    dif[c][ww].append((ym - dm) / nm - (yc - dc) / nc)
    return ({c: {ww: np.nanmean(np.log(np.array(v)), 0) for ww, v in d.items()} for c, d in res.items()},
            {c: {ww: np.nanmean(np.array(v), 0) for ww, v in d.items()} for c, d in dif.items()})
log_ratio, diff = mc_effects()

def unit_effect(table, ww):          # среднее эффектов сегментов юнита по всем фичам
    return np.nanmean([np.nan_to_num(table[c][ww][seg_codes[c]], nan=0.0) for c in seg], 0)

# ---------- 3. TWFE: эффект скидки с фикс. эффектами юнита и недели (классика) ----------
def twfe():
    d = pd.get_dummies(tr.treatment.round(1)).drop(columns=0.0).astype(float); d.columns = W[1:]
    y = tr.outcome.astype(float).copy(); X = d.copy()
    for _ in range(20):                                         # альтернирующее демин-ние
        for key in ['unitID', 'weekID']:
            y -= y.groupby(tr[key]).transform('mean')
            X -= X.groupby(tr[key]).transform('mean')
    return dict(zip(W[1:], np.linalg.lstsq(X.values, y.values, rcond=None)[0]))
tw = twfe()

# ---------- 4. сборка прогнозов ----------
base0 = P_gbm[0.0]
preds = {
    'Наивный (последнее значение)': {w: np.repeat(last.outcome.values[:, None], H, 1) for w in W},
    'GBM S-learner (прогнозная модель)': P_gbm,
    'TWFE (средний эффект) + GBM уровень': {w: base0 + (tw[w] if w else 0) for w in W},
    'Ваш метод: отношение + GBM уровень': {w: base0 * (np.exp(unit_effect(log_ratio, w))[:, None] if w else 1) for w in W},
    'Вариант: разница + GBM уровень': {w: base0 + (unit_effect(diff, w)[:, None] if w else 0) for w in W},
}
# полностью ваш метод: уровень = среднее последних 4 недель, очищенное от скидки
r_all = {w: (np.exp(unit_effect(log_ratio, w)) if w else np.ones(len(units))) for w in W}
recent = tr[tr.weekID > T - 4]
defl = recent.outcome.values / np.choose(np.round(recent.treatment.values * 10).astype(int),
                                         [r_all[w][recent.u.values] for w in W])
own_base = pd.Series(defl).groupby(recent.u.values).mean().values
preds['Ваш метод целиком (свой уровень)'] = {w: np.repeat((own_base * r_all[w])[:, None], H, 1) for w in W}

# тот же метод, но эффект — разница: уровень = последние 4 недели минус эффект их скидки, прогноз = уровень + эффект
d_all = {w: (unit_effect(diff, w) if w else np.zeros(len(units))) for w in W}
defl_d = recent.outcome.values - np.choose(np.round(recent.treatment.values * 10).astype(int),
                                           [d_all[w][recent.u.values] for w in W])
own_base_d = pd.Series(defl_d).groupby(recent.u.values).mean().values
preds['Ваш метод целиком, разница'] = {w: np.repeat((own_base_d + d_all[w])[:, None], H, 1) for w in W}

# ---------- 5. оценка ----------
truth = {w: te.pivot(index='unitID', columns='step', values=s).loc[units].values for w, s in zip(W, WS)}
true_eff = (truth[0.5] - truth[0.0]).mean(1)
out = []
for name, p in preds.items():
    rmse = np.sqrt(np.mean([(p[w] - truth[w]) ** 2 for w in W]))
    eff_rmse = np.sqrt(np.mean([((p[w] - p[0.0]) - (truth[w] - truth[0.0])) ** 2 for w in W[1:]]))
    pe = (p[0.5] - p[0.0]).mean(1)
    rho = spearmanr(pe, true_eff).correlation if np.std(pe) > 1e-6 else np.nan
    top = np.argsort(-pe)[: len(pe) * 3 // 10] if np.std(pe) > 1e-6 else None
    capture = true_eff[top].sum() / np.sort(true_eff)[::-1][: len(top)].sum() if top is not None else np.nan
    out.append(dict(метод=name, RMSE_итог=rmse, RMSE_эффекта=eff_rmse, спирмен_эффекта=rho, топ30_доля_от_идеала=capture))
pd.set_option('display.width', 200)
print(pd.DataFrame(out).round(3).to_string(index=False))
cost = 0.4
TRUE = np.stack([truth[w] for w in W], 0)
P = np.stack([(TRUE[i] * ((1 - W[i]) - cost)).sum(1) for i in range(6)], 0)        # 6 × товары
pick = lambda pred: np.argmax(np.stack([(pred[w] * ((1 - w) - cost)).sum(1) for w in W], 0), 0)
idx = np.arange(len(units))
mine = P[pick(preds['Ваш метод целиком (свой уровень)']), idx]
ml = P[pick(preds['GBM S-learner (прогнозная модель)']), idx]
cur = P[np.round(tr[tr.weekID == T].set_index('unitID').loc[units].treatment.values * 10).astype(int), idx]
none = P[0]
np.save(f'stab_{sys.argv[1]}.npy', np.stack([mine, ml, cur, none]))
print(sys.argv[1], 'mine vs cur %+.1f%%' % ((mine.sum()/cur.sum()-1)*100), 'ml vs cur %+.1f%%' % ((ml.sum()/cur.sum()-1)*100))
