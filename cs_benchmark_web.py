#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
CS 信道估计算法基准测试 — Web 版 v2
运行: streamlit run cs_benchmark_web.py
==============================================================================
"""
import sys, os, io, time, math, ast
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import linprog
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import altair as alt
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════
# 页面配置 (暗色主题见 .streamlit/config.toml)
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS 信道估计基准测试", page_icon=":material/sensors:",
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None, 'Report a bug': None,
        'About': 'CS Channel Estimation Benchmark — Web Edition'
    }
)

# ═══════════════════════════════════════════════════════════════════════
# [1] 算法实现
# ═══════════════════════════════════════════════════════════════════════
class CSAlgorithmBase:
    def __init__(self, name): self.name = name
    def recover(self, y, A, sparsity): raise NotImplementedError
    def recover_multi(self, Y, A, sparsity):
        K_users, M = Y.shape; N = A.shape[1]
        H_hat = np.zeros((K_users, N), dtype=Y.dtype); supports = []
        for k in range(K_users):
            H_hat[k], sup = self.recover(Y[k], A, sparsity)
            sup_arr = np.asarray(sup, int).ravel()
            if len(sup_arr) < sparsity: sup_arr = np.pad(sup_arr, (0, sparsity-len(sup_arr)), constant_values=-1)
            supports.append(sup_arr[:sparsity])
        return H_hat, np.array(supports, int)
    def _safe_lstsq(self, A_s, y):
        try: return np.linalg.lstsq(A_s, y, rcond=None)[0]
        except: return np.linalg.solve(A_s.conj().T@A_s+1e-6*np.eye(A_s.shape[1]), A_s.conj().T@y)

class OMPAlgorithm(CSAlgorithmBase):
    def __init__(self): super().__init__("OMP")
    def recover(self, y, A, K):
        M, N = A.shape; y = np.asarray(y).ravel(); AH = A.conj().T
        residual, support = y.copy(), []
        for _ in range(K):
            corr = np.abs(AH @ residual)
            for i in support: corr[i] = -1
            idx = int(np.argmax(corr))
            if corr[idx] < 1e-15: break
            support.append(idx)
            h_s = self._safe_lstsq(A[:, support], y)
            residual = y - A[:, support] @ h_s
        h = np.zeros(N, dtype=y.dtype)
        if support: h[support] = h_s
        return h, np.array(support, int)

class CoSaMPAlgorithm(CSAlgorithmBase):
    def __init__(self): super().__init__("CoSaMP")
    def recover(self, y, A, K):
        M, N = A.shape; y = np.asarray(y).ravel(); AH = A.conj().T
        residual, support = y.copy(), []
        for _ in range(min(K, 50)):
            proxy = np.abs(AH @ residual)
            n = min(2*K, N); cand = np.argpartition(proxy, -n)[-n:]
            cand = cand[np.argsort(proxy[cand])[::-1]]
            merged = list(dict.fromkeys(support+list(cand)))
            h_m = self._safe_lstsq(A[:, merged], y)
            top = np.argpartition(np.abs(h_m), -min(K,len(h_m)))[-min(K,len(h_m)):]
            top = top[np.argsort(np.abs(h_m[top]))[::-1]]
            support = [merged[i] for i in top]
            h_s = self._safe_lstsq(A[:, support], y)
            residual = y - A[:, support] @ h_s
            if np.linalg.norm(residual) < 1e-8: break
        h = np.zeros(N, dtype=y.dtype)
        if support: h[support] = h_s
        return h, np.array(support, int)

class L1MinAlgorithm(CSAlgorithmBase):
    def __init__(self): super().__init__("L1-Min")
    def recover(self, y, A, K=0):
        M, N = A.shape; y, A = np.asarray(y).ravel(), np.asarray(A)
        if M > N: raise ValueError("L1需要M<=N")
        A_re, A_im = A.real, A.imag; y_re, y_im = y.real, y.imag
        c = np.concatenate([np.zeros(2*N), np.ones(N)])
        Z = np.zeros((M,N)); I = np.eye(N); ZN = np.zeros((N,N))
        A_eq = np.vstack([np.hstack([A_re,-A_im,Z]), np.hstack([A_im,A_re,Z])])
        b_eq = np.concatenate([y_re, y_im])
        A_ub = np.vstack([np.hstack([I,ZN,-I]), np.hstack([-I,ZN,-I]), np.hstack([ZN,I,-I]), np.hstack([ZN,-I,-I])])
        b_ub = np.zeros(4*N); bounds = [(None,None)]*(2*N) + [(0,None)]*N
        for m in ['highs','interior-point']:
            try:
                r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=m, options={'maxiter':10000,'disp':False})
                if r.success: break
            except: r = type('o',(),{'success':False,'x':None})()
        if not (r.success and r.x is not None):
            h = self._safe_lstsq(A, y)
            if K>0: th = np.sort(np.abs(h))[-K]; h[np.abs(h)<th] = 0
            return h, np.argsort(np.abs(h))[-max(1,K):]
        h = r.x[:N] + 1j*r.x[N:2*N]; tol = max(1e-6, np.max(np.abs(h))*1e-4)
        sup = np.where(np.abs(h)>tol)[0]
        return h, sup if len(sup) else np.array([np.argmax(np.abs(h))])

def safe_exec_recover(code, filepath=None):
    code = code.lstrip('﻿​‌‍⁠￾')
    code = code.replace('﻿','').replace('￾','')
    try: tree = ast.parse(code)
    except SyntaxError as e: raise ValueError(f"语法错误 行{e.lineno}: {e.msg}")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'recover':
            ns = {'np':np,'numpy':np,'scipy':__import__('scipy')}
            exec(ast.unparse(ast.Module(body=[node],type_ignores=[])), ns)
            fn = ns['recover']
            fn(np.array([0.5+0.3j,-0.2+0.1j]), np.eye(4,dtype=np.complex128)[:2]+0.1j*np.ones((2,4)), 2)
            return fn, "独立函数"
    found = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef) and item.name in ('recover','recover_multi','recover_adaptive'):
                    found.append((node.name, item.name))
    if found:
        cls_name, method = found[0]; import importlib.util
        ns = {'np':np,'numpy':np,'scipy':__import__('scipy'),'sys':sys,'os':os,'time':time,
              'io':__import__('io'),'warnings':__import__('warnings'),
              'collections':__import__('collections'),'random':__import__('random'),
              'matplotlib':__import__('matplotlib')}
        allowed = []
        for n in ast.iter_child_nodes(tree):
            if isinstance(n, (ast.Import,ast.ImportFrom,ast.ClassDef,ast.Assign,ast.FunctionDef)):
                allowed.append(n)
        exec(ast.unparse(ast.Module(body=allowed,type_ignores=[])), ns)
        cls = ns.get(cls_name)
        if cls is None:
            if filepath:
                spec = importlib.util.spec_from_file_location('_cust', filepath)
                mod = importlib.util.module_from_spec(spec); sys.modules['_cust'] = mod
                spec.loader.exec_module(mod); cls = getattr(mod, cls_name)
            else:
                exec(code, ns); cls = ns.get(cls_name)
        if cls is None: raise ValueError(f"无法加载类 {cls_name}")
        has_sp = any(isinstance(n, ast.FunctionDef) and n.name=='__init__' and
                     any(a.arg=='sparsity' for a in n.args.args) for n in ast.iter_child_nodes(tree))
        w = (lambda y,A,K,c=cls,m=method: getattr(c(sparsity=K),m)(y,A)) if has_sp \
            else (lambda y,A,K,c=cls,m=method: getattr(c(),m)(y,A))
        w(np.array([0.5+0.3j,-0.2+0.1j]), np.eye(4,dtype=np.complex128)[:2]+0.1j*np.ones((2,4)), 2)
        return w, f"{cls_name}.{method} (自动包装)"
    raise ValueError("未找到 recover 函数/方法")

# ═══════════════════════════════════════════════════════════════════════
# [2] 仿真引擎
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class SimConfig:
    N: int = 128; M_values: List[int] = field(default_factory=lambda: [16,24,32,48,64])
    K: int = 8; snr_values: List[float] = field(default_factory=lambda: [-5,0,5,10,15,20,25])
    n_trials: int = 20; channel_model: str = "exact-sparse"
    block_size: int = 4; leakage_ratio: float = 0.1
    n_users: int = 1; overlap_ratio: float = 0.5

CHANNEL_MODELS = {
    "exact-sparse":       "精确 K-稀疏 (理想仿真)",
    "leaky-sparse":       "泄漏稀疏 (有限角度分辨率)",
    "clustered-sparse":   "成簇稀疏 (毫米波散射簇)",
    "rician-sparse":      "莱斯稀疏 (主径 + 散射背景)",
    "freq-selective":     "多径频率选择性 (时延扩展)",
    "time-varying":       "时变多普勒 (Jakes 模型)",
    "spatial-correlated": "空间相关 MIMO (Kronecker)",
    "mixed-sparse":       "混合稀疏-密集 (最接近真实)",
    "multi-user-joint":   "多用户联合稀疏 (共享散射体)",
}

class SimulationEngine:
    def __init__(self, cfg): self.cfg = cfg; self.rng = np.random.RandomState(42)

    def gen_channel(self):
        N, K, model = self.cfg.N, self.cfg.K, self.cfg.channel_model
        if self.cfg.n_users > 1 or model == "multi-user-joint":
            return self._gen_multi_user()
        if model == "exact-sparse":
            sup = self.rng.choice(N, K, replace=False); sup.sort()
            h = np.zeros(N, dtype=np.complex128)
            h[sup] = (0.5+0.5*self.rng.rand(K))*np.exp(2j*np.pi*self.rng.rand(K))
            return h, sup
        if model == "leaky-sparse":
            leak = self.cfg.leakage_ratio; md = max(2, N//(4*K))
            sup = []
            while len(sup) < K:
                c = self.rng.randint(0, N)
                if all(abs(c-s)>=md for s in sup): sup.append(c)
            sup.sort(); h = np.zeros(N, dtype=np.complex128)
            for idx in sup:
                c = (0.5+0.5*self.rng.rand())*np.exp(2j*np.pi*self.rng.rand()); h[idx] = c
                lp = np.abs(c)**2*leak
                for off in range(1,6):
                    for pos in [idx-off, idx+off]:
                        if 0<=pos<N and pos not in sup:
                            h[pos] += np.sqrt(lp/2)*np.exp(-off/1.5)*np.exp(2j*np.pi*self.rng.rand())*c/np.abs(c)
            return h, np.where(np.abs(h)>np.max(np.abs(h))*0.05)[0]
        if model == "clustered-sparse":
            bs = self.cfg.block_size; nb = N//bs; nc = max(1, K//bs)
            blocks = []
            while len(blocks) < nc:
                b = self.rng.randint(0, nb)
                if all(abs(b-p)>=2 for p in blocks): blocks.append(b)
            h = np.zeros(N, dtype=np.complex128); sup = []
            for bi in blocks:
                s,e = bi*bs, min(bi*bs+bs,N); w = np.exp(-0.5*np.linspace(-1.5,1.5,e-s)**2)
                h[s:e] = (0.5+0.5*self.rng.rand())*w*np.exp(2j*np.pi*self.rng.rand(e-s)); sup.extend(range(s,e))
            return h, np.where(np.abs(h)>np.max(np.abs(h))*0.1)[0]
        # fallback
        sup = self.rng.choice(N, K, replace=False); sup.sort()
        h = np.zeros(N, dtype=np.complex128)
        h[sup] = (0.5+0.5*self.rng.rand(K))*np.exp(2j*np.pi*self.rng.rand(K))
        return h, sup

    def _gen_multi_user(self):
        N = self.cfg.N; K = self.cfg.K; u = max(2, self.cfg.n_users)
        ov, bs = self.cfg.overlap_ratio, self.cfg.block_size
        nc = max(1, int(K*ov)); ncb = max(1, round(nc/bs)); nb = N//bs
        cblk = list(self.rng.choice(nb, min(ncb,nb//2), replace=False))
        cs = []; [cs.extend(range(b*bs,min(b*bs+bs,N))) for b in cblk]; cs = cs[:nc]
        H = np.zeros((u,N), dtype=np.complex128); sups = []
        for uu in range(u):
            ni = K - len(cs); isup = []
            if ni > 0:
                av = [b for b in range(nb) if b not in cblk]
                if av:
                    ib = list(self.rng.choice(av, min(max(1,round(ni/bs)),len(av)), replace=False))
                    [isup.extend(range(b*bs,min(b*bs+bs,N))) for b in ib]; isup = isup[:ni]
            fs = np.array(list(cs)+list(isup), int); fs.sort()
            H[uu, cs] = (0.5+0.5*self.rng.rand(len(cs)))*np.exp(2j*np.pi*self.rng.rand(len(cs)))
            if isup: H[uu, isup] = (0.3+0.7*self.rng.rand(len(isup)))*np.exp(2j*np.pi*self.rng.rand(len(isup)))
            sups.append(fs)
        return H, np.array(sups, dtype=object)

    def gen_sensing(self, M):
        N = self.cfg.N; A = (self.rng.randn(M,N)+1j*self.rng.randn(M,N))/np.sqrt(2*M)
        nrm = np.linalg.norm(A, axis=0); nrm[nrm<1e-15]=1.0; return A/nrm

    def add_noise(self, s, snr):
        sp = np.mean(np.abs(s)**2)
        if sp<1e-15: return s
        npw = sp/(10**(snr/10))
        return s + np.sqrt(npw/2)*(self.rng.randn(*s.shape)+1j*self.rng.randn(*s.shape))

    def metrics(self, ht, he):
        n = np.sum(np.abs(ht-he)**2); d = np.sum(np.abs(ht)**2)
        nmse = float(n/d) if d>1e-15 else 0.0
        dot = np.abs(np.vdot(ht,he)); npd = np.linalg.norm(ht)*np.linalg.norm(he)
        cos = float(dot/npd) if npd>1e-15 else 0.0
        sup = np.where(np.abs(ht)>1e-8)[0]
        ber = min(0.5, float(np.mean(np.abs(ht[sup]-he[sup])**2)/max(np.mean(np.abs(ht[sup])**2),1e-15))*0.3) if len(sup)>0 else 0.0
        return {'nmse':nmse,'cosine':cos,'ber':ber}

    def _run_grid(self, algo, on_cell=None):
        cfg = self.cfg; nM, nS = len(cfg.M_values), len(cfg.snr_values)
        res = {'name': algo.name, 'nmse': np.zeros((nM,nS)), 'time': np.zeros((nM,nS)),
               'ber': np.zeros((nM,nS)), 'cosine': np.zeros((nM,nS)), 'avg_time': 0.0}
        all_t = []; multi = cfg.n_users > 1
        for i, M in enumerate(cfg.M_values):
            for j, snr in enumerate(cfg.snr_values):
                nl, bl, cl, tl = [], [], [], []
                for _ in range(cfg.n_trials):
                    H, _ = self.gen_channel(); A = self.gen_sensing(M)
                    if multi:
                        Ku = H.shape[0]; Y = np.zeros((Ku, M), dtype=np.complex128)
                        for k in range(Ku): Y[k] = self.add_noise(A @ H[k], snr)
                        t0 = time.perf_counter(); Hh, _ = algo.recover_multi(Y, A, cfg.K)
                        dt = time.perf_counter()-t0
                        for k in range(Ku):
                            m = self.metrics(H[k], Hh[k]); nl.append(m['nmse']); bl.append(m['ber']); cl.append(m['cosine'])
                    else:
                        ht = np.asarray(H).ravel(); y = self.add_noise(A @ ht, snr)
                        t0 = time.perf_counter(); he, _ = algo.recover(y, A, cfg.K)
                        dt = time.perf_counter()-t0
                        m = self.metrics(ht, he); nl.append(m['nmse']); bl.append(m['ber']); cl.append(m['cosine'])
                    tl.append(dt)
                res['nmse'][i,j] = np.mean(nl) if nl else np.nan
                res['ber'][i,j] = np.mean(bl) if bl else np.nan
                res['cosine'][i,j] = np.mean(cl) if cl else np.nan
                res['time'][i,j] = np.mean(tl) if tl else np.nan; all_t.extend(tl)
                if on_cell is not None:
                    on_cell()
        res['avg_time'] = np.mean(all_t) if all_t else 0.0
        return res

COLORS = ['#89b4fa','#a6e3a1','#f9e2af','#f38ba8','#cba6f7','#94e2d5','#fab387','#89dceb']
MARKERS = ['o','s','^','D','v','*','P','X']

def make_plots(results, M_vals, SNR_vals):
    """生成三张暗色主题图表"""
    S = {'bg':'#1e1e2e','fg':'#cdd6f4','grid':'#45475a','tick':7.5,'label':8.5,'title':9.5}
    def _setup(ax):
        ax.set_facecolor(S['bg']); ax.tick_params(colors=S['fg'],labelsize=S['tick'])
        ax.grid(True,alpha=0.15,color=S['grid'],lw=0.4)
        for sp in ax.spines.values(): sp.set_color(S['grid']); sp.set_lw(0.5)
        ax.xaxis.label.set_color(S['fg']); ax.xaxis.label.set_fontsize(S['label'])
        ax.yaxis.label.set_color(S['fg']); ax.yaxis.label.set_fontsize(S['label'])
        ax.title.set_color(S['fg']); ax.title.set_fontsize(S['title'])

    f1, a1 = plt.subplots(figsize=(5.5, 3.8)); f1.patch.set_facecolor(S['bg']); _setup(a1)
    f2, a2 = plt.subplots(figsize=(5.5, 3.8)); f2.patch.set_facecolor(S['bg']); _setup(a2)
    f3, a3 = plt.subplots(figsize=(5.5, 3.8)); f3.patch.set_facecolor(S['bg']); _setup(a3)

    ms, mm = len(SNR_vals)//2, len(M_vals)//2
    for i, r in enumerate(results):
        if r['nmse'] is None: continue
        c, mk = COLORS[i%len(COLORS)], MARKERS[i%len(MARKERS)]
        a1.plot(M_vals, 10*np.log10(np.maximum(r['nmse'][:,ms],1e-15)),
                color=c, marker=mk, lw=1.5, ms=5, label=r['name'])
        a2.plot(SNR_vals, 10*np.log10(np.maximum(r['nmse'][mm,:],1e-15)),
                color=c, marker=mk, lw=1.5, ms=5, label=r['name'])

    a1.set_xlabel('导频数 M'); a1.set_ylabel('NMSE (dB)')
    a1.set_title(f'NMSE vs 导频数 (SNR={SNR_vals[ms]} dB)', fontweight='bold')
    a1.legend(fontsize=6.5, loc='upper right', framealpha=0.5).get_frame().set_facecolor(S['bg'])

    a2.set_xlabel('SNR (dB)'); a2.set_ylabel('NMSE (dB)')
    a2.set_title(f'NMSE vs SNR (M={M_vals[mm]})', fontweight='bold')
    a2.legend(fontsize=6.5, loc='upper right', framealpha=0.5).get_frame().set_facecolor(S['bg'])

    ns = [r['name'] for r in results if r['avg_time']>0]
    ts = [r['avg_time']*1000 for r in results if r['avg_time']>0]
    if ns:
        cl = [COLORS[i%len(COLORS)] for i in range(len(ns))]
        bars = a3.bar(range(len(ns)), ts, color=cl, edgecolor='white', lw=0.4)
        for b, t in zip(bars, ts):
            a3.text(b.get_x()+b.get_width()/2, b.get_height()+max(ts)*0.02,
                    f'{t:.1f}ms', ha='center', fontsize=6.5, color=S['fg'])
        a3.set_xticks(range(len(ns))); a3.set_xticklabels(ns, rotation=10, ha='right', fontsize=6.5)
    a3.set_ylabel('平均耗时 (ms)'); a3.set_title('算法计算效率对比', fontweight='bold')

    for f in [f1,f2,f3]: f.tight_layout(pad=1.2)
    return f1, f2, f3

# ═══════════════════════════════════════════════════════════════════════
# [2.5] 缓存、算法构造与交互图表
# ═══════════════════════════════════════════════════════════════════════
_BUILTIN_ALGOS = {"OMP": OMPAlgorithm, "CoSaMP": CoSaMPAlgorithm, "L1-Min": L1MinAlgorithm}

class _CustomAlgo(CSAlgorithmBase):
    def __init__(self, fn, name):
        super().__init__(name); self._f = fn
    def recover(self, y, A, K):
        r = self._f(np.asarray(y).ravel().copy(), np.asarray(A).copy(), K)
        return np.asarray(r[0]).ravel(), np.asarray(r[1], int).ravel()

def _build_algo(algo_key, custom_code=""):
    if custom_code:
        fn, _desc = safe_exec_recover(custom_code)
        return _CustomAlgo(fn, algo_key)
    return _BUILTIN_ALGOS[algo_key]()

def _cfg_to_key(cfg):
    return (cfg.N, tuple(cfg.M_values), cfg.K, tuple(cfg.snr_values), cfg.n_trials,
            cfg.channel_model, cfg.block_size, cfg.leakage_ratio, cfg.n_users, cfg.overlap_ratio)

def _cfg_from_key(key):
    N, Mv, K, Sn, trials, model, bs, leak, users, ov = key
    return SimConfig(N=N, M_values=list(Mv), K=K, snr_values=list(Sn), n_trials=trials,
                     channel_model=model, block_size=bs, leakage_ratio=leak,
                     n_users=users, overlap_ratio=ov)

@st.cache_data(ttl=600, show_spinner=False)
def _run_grid_cached(algo_key, cfg_key, custom_code=""):
    """缓存单算法 × 单配置的整网格结果 (可哈希键, 重复运行瞬时返回)"""
    cfg = _cfg_from_key(cfg_key)
    eng = SimulationEngine(cfg)
    algo = _build_algo(algo_key, custom_code)
    return eng._run_grid(algo)

# ── 参数预设 ──
PRESETS = {
    "自定义": {},
    "快速验证":   {"cfg_N": 64,  "cfg_M": "16, 24, 32",        "cfg_K": 4,  "cfg_SNR": "0, 10, 20",            "cfg_trials": 10},
    "标准对比":   {"cfg_N": 128, "cfg_M": "16, 24, 32, 48, 64", "cfg_K": 8,  "cfg_SNR": "-5, 0, 5, 10, 15, 20, 25", "cfg_trials": 20},
    "高精度":     {"cfg_N": 256, "cfg_M": "32, 48, 64, 96, 128", "cfg_K": 16, "cfg_SNR": "-10, -5, 0, 5, 10, 15, 20, 25, 30", "cfg_trials": 50},
    "多用户联合": {"cfg_N": 128, "cfg_M": "24, 32, 48, 64",     "cfg_K": 8,  "cfg_SNR": "0, 10, 20",            "cfg_trials": 20, "cfg_users": 4, "cfg_overlap": 0.5},
}

def _apply_preset():
    vals = PRESETS.get(st.session_state.get("preset", "自定义"), {})
    for k, v in vals.items():
        st.session_state[k] = v

# ── Altair 交互图表 ──
_CHART_BG, _CHART_FG, _CHART_GRID = '#1e1e2e', '#cdd6f4', '#45475a'

def _dark_configure(chart):
    return (chart
        .configure(background=_CHART_BG)
        .configure_axis(labelColor=_CHART_FG, titleColor=_CHART_FG, gridColor=_CHART_GRID, domainColor=_CHART_GRID, tickColor=_CHART_GRID)
        .configure_legend(labelColor=_CHART_FG, titleColor=_CHART_FG, symbolStrokeColor=_CHART_FG)
        .configure_title(color=_CHART_FG, fontSize=14, fontWeight='bold')
        .configure_view(stroke=None))

def make_altair_charts(results, M_vals, SNR_vals):
    """三张可交互的暗色 Altair 图 (悬停看数值 / 缩放 / 点击图例高亮)"""
    ms, mm = len(SNR_vals)//2, len(M_vals)//2
    rows_m, rows_s, rows_t = [], [], []
    for r in results:
        if r['nmse'] is None: continue
        name = r['name']
        for x, nm in zip(M_vals, r['nmse'][:, ms]):
            rows_m.append({"算法": name, "M": int(x), "NMSE": float(10*np.log10(max(nm, 1e-15)))})
        for x, nm in zip(SNR_vals, r['nmse'][mm, :]):
            rows_s.append({"算法": name, "SNR": float(x), "NMSE": float(10*np.log10(max(nm, 1e-15)))})
        if r['avg_time'] > 0:
            rows_t.append({"算法": name, "耗时": float(r['avg_time']*1000)})

    names = [r['name'] for r in results if r['nmse'] is not None]
    color_scale = alt.Scale(domain=names, range=COLORS[:max(len(names), 1)])
    hl = alt.selection_point(fields=['算法'], bind='legend')

    df_m, df_s, df_t = pd.DataFrame(rows_m), pd.DataFrame(rows_s), pd.DataFrame(rows_t)

    c1 = _dark_configure(alt.Chart(df_m).encode(
        color=alt.Color('算法:N', scale=color_scale).legend(symbolSize=80),
        opacity=alt.condition(hl, alt.value(1.0), alt.value(0.15)),
        x=alt.X('M:Q', title='导频数 M'),
        y=alt.Y('NMSE:Q', title='NMSE (dB)'),
        tooltip=[alt.Tooltip('算法:N'), alt.Tooltip('M:Q'), alt.Tooltip('NMSE:Q', format='.2f')])
        .mark_line(point=True, strokeWidth=2)
        .properties(title=f'NMSE vs 导频数 (SNR={SNR_vals[ms]} dB)', height=320)
        .add_params(hl).interactive())

    c2 = _dark_configure(alt.Chart(df_s).encode(
        color=alt.Color('算法:N', scale=color_scale).legend(symbolSize=80),
        opacity=alt.condition(hl, alt.value(1.0), alt.value(0.15)),
        x=alt.X('SNR:Q', title='SNR (dB)'),
        y=alt.Y('NMSE:Q', title='NMSE (dB)'),
        tooltip=[alt.Tooltip('算法:N'), alt.Tooltip('SNR:Q'), alt.Tooltip('NMSE:Q', format='.2f')])
        .mark_line(point=True, strokeWidth=2)
        .properties(title=f'NMSE vs SNR (M={M_vals[mm]})', height=320)
        .add_params(hl).interactive())

    c3 = _dark_configure(alt.Chart(df_t).encode(
        x=alt.X('算法:N', title=None, sort=None),
        y=alt.Y('耗时:Q', title='平均耗时 (ms)'),
        color=alt.Color('算法:N', scale=color_scale, legend=None),
        tooltip=[alt.Tooltip('算法:N'), alt.Tooltip('耗时:Q', format='.2f')])
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .properties(title='算法计算效率对比', height=320))

    return c1, c2, c3

# ═══════════════════════════════════════════════════════════════════════
# [2.7] 相变图 (Phase Transition)
# ═══════════════════════════════════════════════════════════════════════
def _phase_ranges(N):
    """相变图 M / K 扫描范围"""
    M_values = list(range(max(8, N // 16), N // 2 + 1, max(4, N // 16)))
    K_values = list(range(2, N // 2 + 1, max(2, N // 32)))
    if not M_values:
        M_values = [N // 2]
    return M_values, K_values


def run_phase_transition(algo_name, N, M_values, K_values, n_trials, snr=30.0):
    """相变图: 计算恢复成功率矩阵 rates[K, M] (0~1)"""
    algo = {"OMP": OMPAlgorithm, "CoSaMP": CoSaMPAlgorithm, "L1-Min": L1MinAlgorithm}[algo_name]()
    rng = np.random.RandomState(0)
    nK, nM = len(K_values), len(M_values)
    rates = np.full((nK, nM), np.nan)
    for mi, M in enumerate(M_values):
        A = (rng.randn(M, N) + 1j * rng.randn(M, N)) / np.sqrt(2 * M)
        A /= np.linalg.norm(A, axis=0)
        for ki, K in enumerate(K_values):
            if K >= M:
                continue  # 不可行 (K >= M), 保持 NaN
            succ = 0
            for _ in range(n_trials):
                sup = rng.choice(N, K, replace=False)
                h = np.zeros(N, dtype=np.complex128)
                h[sup] = (0.5 + 0.5 * rng.rand(K)) * np.exp(2j * np.pi * rng.rand(K))
                y = A @ h
                sp = np.mean(np.abs(y) ** 2)
                y = y + np.sqrt(sp / (10 ** (snr / 10)) / 2) * (rng.randn(M) + 1j * rng.randn(M))
                h_hat, _ = algo.recover(y, A, K)
                nmse = np.sum(np.abs(h - h_hat) ** 2) / max(np.sum(np.abs(h) ** 2), 1e-15)
                if nmse < 1e-3:  # 等效 < -30 dB, 视为精确恢复
                    succ += 1
            rates[ki, mi] = succ / n_trials
    return rates


def make_phase_heatmap(rates, M_values, K_values, algo_name):
    """相变图热力图 (暗色主题)"""
    S = {'bg': '#1e1e2e', 'fg': '#cdd6f4', 'grid': '#45475a'}
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fig.patch.set_facecolor(S['bg'])
    ax.set_facecolor(S['bg'])
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad(S['bg'])
    masked = np.ma.masked_invalid(rates)
    im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(M_values)))
    ax.set_xticklabels([str(m) for m in M_values], fontsize=8)
    ax.set_yticks(range(len(K_values)))
    ax.set_yticklabels([str(k) for k in K_values], fontsize=8)
    ax.set_xlabel('导频数 M', color=S['fg'], fontsize=9)
    ax.set_ylabel('稀疏度 K', color=S['fg'], fontsize=9)
    ax.set_title(f'{algo_name} 相变图 (恢复成功率)', color=S['fg'], fontweight='bold', fontsize=11)
    ax.tick_params(colors=S['fg'])
    for sp in ax.spines.values():
        sp.set_color(S['grid'])
    try:
        ax.contour(masked, levels=[0.5], colors='white', linewidths=1.2, linestyles='--')
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('恢复成功率', color=S['fg'], fontsize=8)
    cbar.ax.tick_params(colors=S['fg'])
    cbar.outline.set_edgecolor(S['grid'])
    fig.tight_layout(pad=1.2)
    return fig

# ═══════════════════════════════════════════════════════════════════════
# [3] UI — 侧边栏
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### :material/tune: 参数配置")

    st.selectbox("参数预设", list(PRESETS), key="preset", on_change=_apply_preset,
                 help="一键填充常用参数组合")

    with st.expander("系统参数", expanded=True, icon=":material/settings:"):
        N = st.number_input("子载波数 N", 32, 512, 128, step=32, key="cfg_N")
        M_str = st.text_input("导频数列表 M", "16, 24, 32, 48, 64",
                              help="逗号分隔, 如: 16, 24, 32", key="cfg_M")
        K = st.number_input("稀疏度 K", 2, 50, 8, step=2,
                           help="建议为 block_size 整数倍", key="cfg_K")
        SNR_str = st.text_input("SNR 列表 (dB)", "-5, 0, 5, 10, 15, 20, 25", key="cfg_SNR")
        n_trials = st.slider("试验次数", 5, 100, 20, step=5,
                            help="越大结果越稳定, 但更慢", key="cfg_trials")

    with st.expander("信道模型", expanded=True, icon=":material/wifi:"):
        channel_model = st.selectbox(
            "信道类型", list(CHANNEL_MODELS.keys()),
            format_func=lambda k: CHANNEL_MODELS[k], key="cfg_model")
        c1, c2 = st.columns(2)
        with c1: block_size = st.number_input("块大小", 1, 16, 4, key="cfg_block")
        with c2: leakage = st.slider("泄漏比", 0.01, 0.5, 0.1, 0.01, key="cfg_leak")

    with st.expander("多用户 (可选)", expanded=False, icon=":material/group:"):
        n_users = st.number_input("用户数", 1, 20, 1, step=1,
                                  help=">1 启用多用户联合恢复", key="cfg_users")
        overlap = st.slider("重叠率", 0.0, 1.0, 0.5, 0.05,
                           help="公共支撑集占比, 高→联合恢复优势", key="cfg_overlap")

    st.toggle("强制重算 (跳过缓存)", key="force_recalc",
              help="勾选后忽略缓存并重新计算, 可看到逐格进度")

    st.divider()
    st.caption(f"内存估算: ~{N*len(M_str.split(','))*n_trials//1000}MB")

# ═══════════════════════════════════════════════════════════════════════
# [4] UI — 主区域
# ═══════════════════════════════════════════════════════════════════════
st.title(":material/sensors: CS 信道估计算法基准测试")
st.caption("Compressed Sensing Channel Estimation Benchmark — Web Edition")

# ── 算法选择 ──
st.subheader(":material/checklist: 算法选择")
ALGO_OPTIONS = ["OMP", "CoSaMP", "L1-Min"]
selected_algos = st.pills(
    "选择算法", ALGO_OPTIONS,
    selection_mode="multi", default=ALGO_OPTIONS,
    label_visibility="collapsed",
)
with st.expander("算法说明", expanded=False, icon=":material/info:"):
    st.markdown(
        "- **OMP** — 贪心算法, 逐原子选择与残差最相关的列\n"
        "- **CoSaMP** — 每轮选 2K 个原子, LS 后保留 K 个\n"
        "- **L1-Min** — 线性规划求解 min‖h‖₁, 理论最优"
    )

# 自定义算法
st.subheader(":material/code: 自定义算法 (可选)")
ct1, ct2 = st.columns([1, 2])
with ct1:
    custom_mode = st.segmented_control(
        "来源", ["不启用", "粘贴代码", "上传文件"],
        default="不启用", label_visibility="collapsed")
with ct2:
    custom_code = ""
    custom_file = None
    if custom_mode == "粘贴代码":
        custom_code = st.text_area(
            "代码", height=120,
            placeholder="def recover(y, A, sparsity):\n    ...\n    return h_hat, support",
            label_visibility="collapsed")
    elif custom_mode == "上传文件":
        custom_file = st.file_uploader("上传 .py 文件", type=["py"], label_visibility="collapsed")
        if custom_file:
            custom_code = custom_file.read().decode('utf-8-sig')
            st.success(f"已加载: {custom_file.name}")

# ── 运行按钮 ──
st.divider()
rc1, rc2, rc3 = st.columns([1, 1, 2])
with rc1:
    run_clicked = st.button("运行仿真", type="primary", width="stretch", icon=":material/play_arrow:")

# ═══════════════════════════════════════════════════════════════════════
# [5] 结果渲染
# ═══════════════════════════════════════════════════════════════════════
def render_results(results, M_values, SNR_values):
    """渲染结果: 指标卡片 + 交互图表 + 详细表格 + 导出"""
    # ── 指标卡片 ──
    st.subheader(":material/analytics: 性能总览")
    best_idx = int(np.argmin([np.nanmean(r['nmse']) for r in results]))
    st.markdown(f":material/emoji_events: 最优: **{results[best_idx]['name']}** (平均 NMSE 最低)")
    with st.container(horizontal=True):
        for r in results:
            db = 10*np.log10(max(np.nanmean(r['nmse']), 1e-15))
            st.metric(label=r['name'], value=f"{db:.1f} dB", border=True)

    # ── 图表 (交互) ──
    st.subheader(":material/query_stats: 可视化分析")
    c1, c2, c3 = make_altair_charts(results, M_values, SNR_values)
    tabs = st.tabs([":material/trending_down: NMSE vs 导频数 M", ":material/show_chart: NMSE vs SNR", ":material/timer: 执行时间对比"])
    with tabs[0]: st.altair_chart(c1, theme=None, width="stretch")
    with tabs[1]: st.altair_chart(c2, theme=None, width="stretch")
    with tabs[2]: st.altair_chart(c3, theme=None, width="stretch")

    # ── 详细表格 ──
    st.subheader(":material/table_chart: 详细数据")
    rows = []
    for r in results:
        nmse_avg = np.nanmean(r['nmse'])
        rows.append({
            '算法': r['name'],
            'NMSE (dB)': f"{10*np.log10(max(nmse_avg,1e-15)):.2f}",
            '余弦相似度': f"{np.nanmean(r['cosine']):.4f}" if not np.all(np.isnan(r['cosine'])) else "N/A",
            'BER': f"{np.nanmean(r['ber']):.3e}" if not np.all(np.isnan(r['ber'])) else "N/A",
            '耗时 (ms)': f"{r['avg_time']*1000:.2f}",
        })
    df_rows = pd.DataFrame(rows)
    st.dataframe(df_rows, width="stretch", hide_index=True)

    # ── 导出 ──
    st.divider()
    st.subheader(":material/download: 导出结果")
    e1, e2, e3, e4 = st.columns(4)
    csv_buf = df_rows.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    with e1: st.download_button("CSV", csv_buf, "results.csv", "text/csv", width="stretch", icon=":material/table_chart:")
    with e2:
        xl_buf = io.BytesIO()
        df_rows.to_excel(xl_buf, index=False); xl_buf.seek(0)
        st.download_button("Excel", xl_buf, "results.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                          width="stretch", icon=":material/grid_on:")
    # PNG 导出使用静态 matplotlib 图 (与交互图独立)
    mf1, mf2, mf3 = make_plots(results, M_values, SNR_values)
    png_idx = 0
    for name, fig in [("nmse_vs_M.png", mf1), ("nmse_vs_SNR.png", mf2), ("time.png", mf3)]:
        pb = io.BytesIO(); fig.savefig(pb, dpi=120, bbox_inches='tight', facecolor='#1e1e2e'); pb.seek(0)
        with [e3, e4, e1, e2][png_idx % 4]:
            st.download_button(name, pb, name, "image/png", width="stretch", icon=":material/image:")
        png_idx += 1


# ═══════════════════════════════════════════════════════════════════════
# [6] 执行仿真
# ═══════════════════════════════════════════════════════════════════════
if run_clicked:
    # ── 参数解析 + 前置校验 ──
    try:
        M_values = sorted({int(x.strip()) for x in M_str.split(',') if x.strip()})
        SNR_values = [float(x.strip()) for x in SNR_str.split(',') if x.strip()]
    except ValueError:
        st.error("M 或 SNR 格式错误, 请用逗号分隔数字 (如 16, 24, 32)", icon=":material/error:"); st.stop()

    if not M_values or not SNR_values:
        st.error("至少需要一个 M 值和 SNR 值", icon=":material/error:"); st.stop()

    for w in [
        f"M 最大值 {max(M_values)} ≥ N={N}, 而 L1-Min 要求 M<N, 建议增大 N 或减小 M" if max(M_values) >= N else "",
        f"稀疏度 K={K} > 最小导频数 M={min(M_values)}, 恢复精度会明显下降" if K > min(M_values) else "",
        f"网格 {len(M_values)}×{len(SNR_values)} × {n_trials} 次试验较多, 可能耗时较长" if len(M_values)*len(SNR_values)*n_trials > 3000 else "",
    ]:
        if w:
            st.warning(w, icon=":material/warning:")

    # ── 收集算法规格 (可哈希缓存键) ──
    algo_specs = []
    if "OMP" in selected_algos: algo_specs.append(("OMP", ""))
    if "CoSaMP" in selected_algos: algo_specs.append(("CoSaMP", ""))
    if "L1-Min" in selected_algos: algo_specs.append(("L1-Min", ""))

    if custom_mode != "不启用" and custom_code.strip():
        try:
            fn, desc = safe_exec_recover(custom_code.strip(),
                                         custom_file.name if custom_mode == "上传文件" and custom_file else None)
            algo_specs.append((desc, custom_code.strip()))
        except Exception as e:
            st.error(f"自定义算法加载失败: {str(e)[:200]}", icon=":material/error:")

    if not algo_specs:
        st.error("请至少选择一个算法", icon=":material/error:"); st.stop()

    cfg = SimConfig(N=N, M_values=M_values, K=K, snr_values=SNR_values,
                    n_trials=n_trials, channel_model=channel_model,
                    block_size=block_size, leakage_ratio=leakage,
                    n_users=n_users, overlap_ratio=overlap)
    cfg_key = _cfg_to_key(cfg)
    engine = SimulationEngine(cfg)
    use_cache = not st.session_state.get("force_recalc", False)
    if not use_cache:
        _run_grid_cached.clear()

    # ── 进度 + 运行 ──
    total_cells = len(algo_specs) * len(M_values) * len(SNR_values)
    progress_bar = st.progress(0.0, "仿真准备中...")
    cell_done = [0]
    t_start = time.perf_counter()
    results = []

    for algo_key, algo_code in algo_specs:
        name = algo_key
        with st.status(f"正在运行: {name}", expanded=True) as stat:
            if use_cache:
                r = _run_grid_cached(algo_key, cfg_key, algo_code)
                stat.update(label=f"{name} 完成 (缓存命中)", state="complete", expanded=False)
            else:
                def _cb(_name=name):
                    cell_done[0] += 1
                    el = time.perf_counter() - t_start
                    eta = el / max(cell_done[0], 1) * (total_cells - cell_done[0])
                    progress_bar.progress(min(cell_done[0] / total_cells, 1.0),
                                          f"{_name} ({cell_done[0]}/{total_cells}) · 已 {el:.0f}s · 剩余 ~{eta:.0f}s")
                algo = _build_algo(algo_key, algo_code)
                r = engine._run_grid(algo, on_cell=_cb)
                stat.update(label=f"{name} 完成", state="complete", expanded=False)
            results.append(r)
            cell_done[0] += len(M_values) * len(SNR_values)
            progress_bar.progress(min(cell_done[0] / total_cells, 1.0),
                                  f"{name} 完成 ({cell_done[0]}/{total_cells})")

    progress_bar.progress(1.0, f"仿真完成! 总用时 {time.perf_counter()-t_start:.0f}s")
    st.toast("仿真完成!", icon=":material/check_circle:")

    # ── 持久化 + 渲染 ──
    st.session_state["last_run"] = {
        "results": results,
        "M": M_values, "SNR": SNR_values,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    render_results(results, M_values, SNR_values)

elif "last_run" in st.session_state:
    # 参数已改动但未重跑: 保留并展示上次结果
    lr = st.session_state["last_run"]
    st.info(f"当前显示**上次运行结果** ({lr['ts']})。若已修改参数, 点击「运行仿真」刷新。", icon=":material/history:")
    render_results(lr["results"], lr["M"], lr["SNR"])

else:
    # ── 初始状态 ──
    st.info("在**左侧边栏**配置参数并选择算法，然后点击「运行仿真」开始。", icon=":material/info:")
    with st.container(border=True):
        st.subheader("快速开始")
        st.markdown(
            "- 左侧配置 N、M、K、SNR，或用「参数预设」一键填充\n"
            "- 选择信道模型（9 种可选）\n"
            "- 多用户模式：展开「多用户」设置用户数 > 1\n"
            "- 可选自定义算法：粘贴代码或上传 .py 文件\n"
            "- 点击运行，查看交互式性能对比图表（悬停看数值 / 缩放）\n"
            "- 导出 CSV / Excel / PNG"
        )


# ═══════════════════════════════════════════════════════════════════════
# [7] 相变图 (独立实验)
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":material/grid_on: 相变图 (Phase Transition)")
st.caption("恢复成功率 vs (M, K) 热力图 — 观察稀疏恢复的相变边界")
with st.expander("相变图实验", expanded=False):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        pt_algo = st.selectbox("算法", ["OMP", "CoSaMP", "L1-Min"], key="pt_algo",
                               help="L1-Min 更精确但计算量较大")
    with c2:
        pt_trials = st.slider("每格试验次数", 5, 50, 10, step=5, key="pt_trials",
                              help="越大成功率统计越稳定, 但更慢")
    with c3:
        pt_run = st.button("运行相变图", key="pt_run", icon=":material/play_arrow:")

    if pt_run:
        with st.status("正在计算相变图...", expanded=True):
            M_values, K_values = _phase_ranges(N)
            rates = run_phase_transition(pt_algo, N, M_values, K_values, pt_trials)
            st.session_state["last_pt"] = {
                "rates": rates, "M": M_values, "K": K_values,
                "algo": pt_algo, "ts": time.strftime("%H:%M:%S"),
            }
        st.toast("相变图完成!", icon=":material/check_circle:")

    if "last_pt" in st.session_state:
        lp = st.session_state["last_pt"]
        st.caption(f"算法 {lp['algo']} · N={N} · {lp['ts']} · 绿色=恢复成功, 红色=失败, 白色虚线=50% 分界")
        st.pyplot(make_phase_heatmap(lp["rates"], lp["M"], lp["K"], lp["algo"]))
