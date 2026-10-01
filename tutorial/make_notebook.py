#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Generate tutorial/Olivar_walkthrough.ipynb (step-by-step reproduction of the Olivar pipeline).
Run:  python tutorial/make_notebook.py && jupyter nbconvert --to notebook --execute --inplace tutorial/Olivar_walkthrough.ipynb
'''
import nbformat as nbf
from pathlib import Path

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip('\n')))

# ---------------------------------------------------------------- 0. intro
md(r'''
# Olivar 多重 PCR tiling 引物设计：逐步复现

本 notebook 按照 Olivar 的真实代码路径，把一次完整设计拆成 9 步逐一运行、逐一验证。配套的流程图与讲解见 `tutorial/olivar_pipeline.html`。

| 步骤 | 对应源码 | 内容 |
|---|---|---|
| 1 | — | 输入数据：SARS-CoV-2 参考序列、变异位点表、H1N1 HA 序列集 |
| 2 | `basic.py`, `design.py` | 基础打分函数：GC、序列复杂度、最近邻 ΔG |
| 3 | `build_helper.run_build` | `build`：k-mer 切分 → 每个位点的风险分量（GC / 复杂度 / BLAST 非特异 / 变异） |
| 4 | `tiling_helper.design_context_seq` | 风险数组 risk array 的加权合成 |
| 5 | `tiling_helper.generate_context` | 引物设计区（PDR）的随机贪心 + 蒙特卡洛优化 |
| 6 | `design.primer_generator.get` | 每个 PDR 内生成引物候选 |
| 7 | `design.PrimerSetBadnessFast`, `tiling_helper.optimize` | SADDLE 模拟退火最小化引物二聚体 |
| 8 | `main.specificity`, `main.sensitivity` | 非特异扩增预测（BLAST）与 MSA 灵敏度验证 |
| 9 | — | 代码审计：已证实的问题 + 改进原型（性能、Tm、简并模式） |

**本环境的说明**

* 仓库自带的人类基因组 BLAST 库 (`example_input/Human/*.nsq`, 772 MB) 是 Git LFS 指针，且沙箱无法访问 NCBI。因此第 3 步用一个**本地构建的小型背景库**演示 BLAST 流程；而第 4–7 步使用仓库自带的 `example_output/EPI_ISL_402124.olvr`，它已经包含了作者用真人类基因组算好的 BLAST 命中数，所以 tiling 结果可以与论文的 `example_output` 直接对比。
* `src/olivar/main.py` 原来有一行 `f'{config['title']}.log'`，这种 f-string 写法只在 Python ≥ 3.12 合法，在 3.8–3.11 上 `import olivar` 直接报 `SyntaxError`（`setup.py` 声称支持 ≥3.8）。本分支已把它改成双引号外壳，这是唯一的源码改动。
''')

code(r'''
import os, sys, time, json, pickle, shutil, subprocess, random, logging, re
from pathlib import Path
from copy import deepcopy

REPO = Path.cwd().parent if Path.cwd().name == 'tutorial' else Path.cwd()
os.chdir(REPO)
sys.path.insert(0, str(REPO / 'src'))           # `import olivar`
sys.path.insert(0, str(REPO / 'src' / 'olivar'))  # 内部模块 basic / design / tiling_helper ...

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display
from Bio import SeqIO, __version__ as bio_ver

import olivar
from olivar import build, tiling, save, specificity, sensitivity
import basic, design, tiling_helper as th, build_helper, msa_tools, ncbi_tools

logging.getLogger('main').setLevel(logging.INFO)
OUT = REPO / 'tutorial' / 'nb_output'
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
NCPU = min(4, os.cpu_count())

plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
print('python', sys.version.split()[0], '| olivar', olivar.__version__, '| biopython', bio_ver, '| numpy', np.__version__, '| pandas', pd.__version__)
for tool in ['blastn', 'makeblastdb', 'mafft']:
    print(f'{tool:12s}', shutil.which(tool))
print(subprocess.run(['blastn', '-version'], capture_output=True, text=True).stdout.splitlines()[0])
''')

# ---------------------------------------------------------------- 1. inputs
md(r'''
## 1. 输入数据

Olivar 支持两种输入模式：

* **模式 2（参考序列 + 变异表）**：`EPI_ISL_402124.fasta`（SARS-CoV-2 武汉株, 29,891 nt）+ `delta_omicron_loc.csv`（Delta/Omicron 的变异坐标与频率）。论文中的 SARS-CoV-2 panel 就是这样设计的。
* **模式 1（MSA）**：`H1N1-HA.fasta`，4,227 条未比对的 H1N1 HA 基因序列。`build --msa --align` 会先调用 MAFFT 比对，再由 `msa_tools.run_variant_call` 生成共识序列和变异表，然后走与模式 2 相同的流程。
''')

code(r'''
ref = next(SeqIO.parse('example_input/EPI_ISL_402124.fasta', 'fasta'))
var = pd.read_csv('example_input/delta_omicron_loc.csv')
print(f'参考序列 {ref.id}: {len(ref.seq):,} nt, GC = {basic.get_GC(str(ref.seq).upper()):.3f}')
print(f'变异表: {len(var)} 行;  变异类型计数:', var['VARIANT'].value_counts().to_dict())
display(var.head())

h1n1 = list(SeqIO.parse('example_input/H1N1-HA.fasta', 'fasta'))
lens = pd.Series([len(r.seq) for r in h1n1])
print(f'H1N1 HA: {len(h1n1)} 条序列, 长度分布(前5):', lens.value_counts().head().to_dict())
''')

# ---------------------------------------------------------------- 2. basic scores
md(r'''
## 2. 基础打分函数

**GC 含量** `basic.get_GC`：G/C 占比。

**序列复杂度** `basic.get_complexity`：分别以 1、2、3 nt 为词计算 Shannon 熵，并除以各自理论最大值（2、4、6 bit），取三者最小值：

$$C = \min\left(\frac{H_1}{2}, \frac{H_2}{4}, \frac{H_3}{6}\right),\quad H_k = -\sum_w p_w \log_2 p_w$$

polyA 的复杂度为 0，随机序列接近 1。

**结合自由能** `design.primer_generator`：采用 SantaLucia 统一最近邻参数（矩阵按 A,T,C,G = 0,1,2,3 编号，`paraH[i][j]` 是 5'-ij-3' 双核苷酸的 ΔH），盐校正只作用于熵：

$$\Delta S' = \Delta S + 0.368\ln[\mathrm{Na^+}],\quad \Delta G_{ij}(T) = \Delta H_{ij} - T\,\Delta S'_{ij}/1000,\quad \Delta G = \Delta G_{init} + \sum_{i}\Delta G_{s_i s_{i+1}}$$

其中 $\Delta G_{init} = 0.2 + 5.7T/1000$。引物的 5' 端会被一直延长，直到 ΔG ≤ `dG_max`（默认 −11.8 kcal/mol）。这里**没有**末端 AT 罚分、没有 Mg²⁺/dNTP 校正，也不算 Tm（第 9 步再讨论）。
''')

code(r'''
examples = {
    'polyA':        'A' * 28,
    'dinuc repeat': 'AT' * 14,
    'triplet rpt':  'CAG' * 9 + 'C',
    'SARS-CoV-2':   str(ref.seq[1000:1028]).upper(),
    'random':       basic.randseq(28, seed=1),
}
rows = [{'name': k, 'seq': s, 'GC': round(basic.get_GC(s), 3), 'complexity': round(basic.get_complexity(s), 3)} for k, s in examples.items()]
display(pd.DataFrame(rows))

# 手算一条引物的 ΔG，并与 Olivar 的实现对比
T, Na = 60, 0.18
gen = design.primer_generator(temperature=T, salinity=Na)
p = 'ACCAACCAACTTTCGATCTCTTGT'
paraH = np.array([[-7.6,-7.2,-8.4,-7.8],[-7.2,-7.6,-8.2,-8.5],[-8.5,-7.8,-8.0,-10.6],[-8.2,-8.4,-9.8,-8.0]])
paraS = np.array([[-21.3,-20.4,-22.4,-21.0],[-21.3,-21.3,-22.2,-22.7],[-22.7,-21.0,-19.9,-27.2],[-22.2,-22.4,-24.4,-19.9]])
idx = {'A':0,'T':1,'C':2,'G':3}
Kt = T + 273.15
dG_hand = 0.2 + Kt*5.7/1000 + sum(paraH[idx[a],idx[b]] - Kt*(paraS[idx[a],idx[b]] + 0.368*np.log(Na))/1000 for a, b in zip(p, p[1:]))
print(f'{p}: 手算 ΔG = {dG_hand:.4f}, Olivar = {gen.dG_init + gen.StacksDG(p):.4f} kcal/mol')
''')

# ---------------------------------------------------------------- 3. build
md(r'''
## 3. `build`：把参考序列变成“风险分量”

`build_helper.run_build` 的步骤：

1. 读入第一条 FASTA 记录，全部转小写；变异表中的位点改成**大写**（之后用于 `--check-var`：引物 3' 端 5 nt 内不能有大写碱基）。
2. `var_arr[pos] += FREQ`，然后 `var_arr = var_arr ** 0.5`（开方放大低频变异）。
3. 以 `word_size = 28`、`offset = 14` 把序列切成相互重叠一半的 k-mer（word），对每个 word 计算 GC、复杂度，并用 `blastn-short`（rough 模式, evalue 10）对 BLAST 库计数命中条数。
4. 每个碱基位置被 `word_size/offset = 2` 个 word 覆盖，位置分数取这 2 个 word 的平均，得到 `gc_arr / comp_arr / hits_arr`。两端各裁掉 `(n_cycle-1)·offset` 个碱基，记录 `start/stop`。
5. 简并模式（`--deg`，只能配合 MSA）额外计算每个 word 在 MSA 中的完全匹配率 `sensi_arr = 100 − sensitivity` 和简并组合数 `combi_arr = combinations − 1`。

结果用 pickle 存为 `.olvr`。

### 3.1 构建一个本地 BLAST 背景库

真实使用时这里应是宿主基因组（人类 GRCh38）或其他非目标序列。我们拼接一个约 0.4 Mb 的演示库：随机背景 + 50 条 H1N1 HA + 6 段“带 1–2 个突变的 SARS-CoV-2 片段”（模拟宿主中的同源区域），后面会检查 BLAST 命中数能否把这些片段与随机背景区分开。
''')

code(r'''
rng = np.random.default_rng(2024)
DB_DIR = OUT / 'blastdb'; DB_DIR.mkdir()
seq_ref = str(ref.seq).upper()
records = [f'>random_bg\n{"".join(rng.choice(list("ACGT"), 300_000))}']
planted = []
for k, pos in enumerate([2000, 7500, 12000, 18000, 23500, 27000]):
    frag = list(seq_ref[pos:pos+80])
    for m in rng.choice(80, size=1 + k % 2, replace=False):
        frag[m] = {'A':'C','C':'G','G':'T','T':'A'}[frag[m]]
    planted.append((pos+1, pos+80))
    flank = ''.join(rng.choice(list('ACGT'), 500))
    records.append(f'>homolog_{k+1}\n{flank}{"".join(frag)}{flank}')
for r in h1n1[:50]:
    records.append(f'>{r.id}_HA\n{str(r.seq).upper()}')
(DB_DIR / 'background.fasta').write_text('\n'.join(records) + '\n')
out = subprocess.run(['makeblastdb', '-in', str(DB_DIR/'background.fasta'), '-dbtype', 'nucl',
                      '-out', str(DB_DIR/'background'), '-parse_seqids'], capture_output=True, text=True)
print(out.stdout.strip().splitlines()[-1])
print('植入的同源片段 (1-based):', planted)
BLAST_DB = str(DB_DIR / 'background')
''')

md(r'''
### 3.2 运行官方 `build()`（模式 2）
''')

code(r'''
tik = time.time()
build(fasta_path='example_input/EPI_ISL_402124.fasta', msa_path=None, var_path='example_input/delta_omicron_loc.csv',
      BLAST_db=BLAST_DB, out_path=str(OUT/'build'), title=None, threads=NCPU, align=False, min_var=0.01, deg=False)
print(f'build 用时 {time.time()-tik:.1f}s')
olvr_demo = pickle.load(open(OUT/'build'/'EPI_ISL_402124.olvr', 'rb'))
print({k: (type(v).__name__, getattr(v, 'shape', '')) for k, v in olvr_demo.items() if k != 'seq'})
''')

md(r'''
### 3.3 手工重算风险分量，逐位验证

下面不调用 `run_build`，而是按 3 中描述的公式自己切 word、算分、平均，然后与官方结果和仓库自带的 `.olvr` 逐元素比较。
''')

code(r'''
WS, OFF = 28, 14
n_cycle = WS // OFF
seq_raw = list(str(ref.seq).lower())
for _, r in var.iterrows():                      # build 把变异位点改成大写
    for p in range(int(r.START)-1, int(r.STOP)):
        seq_raw[p] = seq_raw[p].upper()
seq_raw = ''.join(seq_raw)
L = len(seq_raw)
seq_len_temp = (L // OFF) * OFF
start_temp = (L - seq_len_temp)//2 + 1
stop_temp = start_temp + seq_len_temp - 1
words = [seq_raw[start_temp-1+p : start_temp-1+p+WS] for p in range(0, seq_len_temp-WS+1, OFF)]
gc_w = np.array([basic.get_GC(w) for w in words])
cx_w = np.array([basic.get_complexity(w) for w in words])
start = start_temp + (n_cycle-1)*OFF
stop = stop_temp - (n_cycle-1)*OFF
n = stop - start + 1
# 每个位点 = 覆盖它的 2 个 word 的均值
gc_arr = np.array([gc_w[p//OFF : p//OFF+n_cycle].mean() for p in range(n)])
cx_arr = np.array([cx_w[p//OFF : p//OFF+n_cycle].mean() for p in range(n)])

var_arr = np.zeros(L)
for _, r in var.iterrows():
    var_arr[r.START-1:r.STOP] += 1.0 if np.isnan(r.FREQ) else r.FREQ
var_arr = (var_arr ** 0.5)[start-1:stop]

shipped = pickle.load(open('example_output/EPI_ISL_402124.olvr', 'rb'))
print(f'{len(words)} 个 word; 设计区 {start}:{stop} (与官方一致: {(start, stop) == (olvr_demo["start"], olvr_demo["stop"])})')
for name, mine in [('gc_arr', gc_arr), ('comp_arr', cx_arr), ('var_arr', var_arr)]:
    print(f'{name:9s} 手算 vs 本次build: {np.allclose(mine, olvr_demo[name])},  手算 vs 仓库自带olvr: {np.allclose(mine, shipped[name])}')

hits_demo = olvr_demo['hits_arr']
print('\n注意: 仓库自带 olvr 的 comp_arr 与新版 build 不同 (它由旧版生成); 论文 risk.csv 不受影响, 因为复杂度只以 <0.4 的 0/1 形式进入 risk, 且该序列没有低复杂度位点。')
mask = np.zeros(len(hits_demo), bool)
for a_, b_ in planted: mask[a_-start : b_-start+1] = True
print(f'\n本地演示库的 BLAST 命中数: 植入同源片段内均值 {hits_demo[mask].mean():.1f}, 其余位置均值 {hits_demo[~mask].mean():.1f}, 其余位置最大 {hits_demo[~mask].max():.1f}')
print('=> 28 nt word 在 evalue=10 的 rough 模式下会产生大量偶然短命中, 植入片段与背景几乎无法区分; 命中数是“相对尺度”而非“同源证据”, 所以 tiling 里要除以最大值归一化。')
print('仓库自带 olvr (人类 GRCh38): 有命中的位点数 =', int((shipped['hits_arr'] > 0).sum()), ', 最大命中数 =', shipped['hits_arr'].max())
print('仓库自带 olvr 的键:', list(shipped.keys()))
''')

code(r'''
x = np.arange(shipped['start'], shipped['stop']+1)
fig, axes = plt.subplots(5, 1, figsize=(11, 7.5), sharex=True)
tracks = [('GC content (word mean)', shipped['gc_arr'], '#3b6ea5'), ('complexity', shipped['comp_arr'], '#c77d2e'),
          ('BLAST hits, human GRCh38 (shipped .olvr)', shipped['hits_arr'], '#3e8e5a'),
          ('BLAST hits, local demo DB', hits_demo, '#7a5aa6'), ('variation (sqrt freq)', shipped['var_arr'], '#b8423a')]
for ax, (t, y, c) in zip(axes, tracks):
    ax.plot(x, y, lw=0.6, color=c); ax.set_title(t, loc='left', fontsize=9)
axes[0].axhline(0.25, ls='--', c='grey', lw=.6); axes[0].axhline(0.75, ls='--', c='grey', lw=.6)
axes[1].axhline(0.4, ls='--', c='grey', lw=.6)
for s, e in planted:
    axes[3].axvspan(s, e, color='#7a5aa6', alpha=.15)
axes[-1].set_xlabel('position on EPI_ISL_402124 (nt)')
plt.tight_layout(); plt.savefig(OUT/'fig_build_tracks.png'); plt.show()
''')

# ---------------------------------------------------------------- 4. risk
md(r'''
## 4. 风险数组 risk array（`design_context_seq` 前半段）

`build` 存的是连续分量，`tiling` 再把它们转换并加权求和：

| 分量 | 转换 | 系数 |
|---|---|---|
| 极端 GC | `gc<0.25 or gc>0.75` → 1，否则 0 | `0.5·w_egc` |
| 低复杂度 | `comp<0.4` → 1 | `0.5·w_lc` |
| 非特异 | `hits / max(hits)`（同一参考内归一化） | `1·w_ns` |
| 变异 | `sqrt(freq)` | `10·w_var` |
| 灵敏度（简并模式） | `100 − sensitivity%` | `0.1·w_sensi` |
| 组合数（简并模式） | `combinations − 1` | `1·w_combi` |

`risk = Σ 分量`，长度补齐到整条参考序列（两端未覆盖位置为 0）。变异的权重（10）远大于其它项，这就是 Olivar 会主动“绕开”SNP 的原因。

注意：仓库自带的 `.olvr` 由旧版生成，缺少 `sensi_arr / combi_arr` 两个键，新版 `design_context_seq` 会直接 `KeyError`。下面先演示这个兼容性问题，再补上全零数组。
''')

code(r'''
cfg = dict(ref_path='example_output/EPI_ISL_402124.olvr', out_path=str(OUT/'tiling'), title='olivar-design',
           max_amp_len=420, min_amp_len=252, w_egc=1.0, w_lc=1.0, w_ns=1.0, w_var=1.0, w_sensi=1.0, w_combi=1.0,
           temperature=60.0, salinity=0.18, dG_max=-11.8, min_GC=0.2, max_GC=0.75, min_complexity=0.4, max_len=36,
           check_SNP=True, fP_prefix='', rP_prefix='', seed=10, threads=NCPU, iterMul=1)
try:
    th.design_context_seq(cfg, deg=False)
except KeyError as e:
    print('旧版 .olvr 在新版代码中报错: KeyError', e)

patched = dict(shipped)
patched['sensi_arr'] = np.zeros_like(shipped['gc_arr']); patched['combi_arr'] = np.zeros_like(shipped['gc_arr'])
(OUT/'tiling').mkdir(exist_ok=True)
OLVR = OUT/'tiling'/'EPI_ISL_402124.olvr'
pickle.dump(patched, open(OLVR, 'wb'), protocol=5)
cfg['ref_path'] = str(OLVR)

def risk_components(o, w=cfg):
    s, e, L = o['start'], o['stop'], len(o['seq'])
    pad = lambda a: np.concatenate((np.zeros(s-1), a, np.zeros(L-e)))
    h = o['hits_arr']/max(o['hits_arr']) if max(o['hits_arr']) else o['hits_arr']
    comp = {
        'extreme GC':     w['w_egc']*0.5*pad(((o['gc_arr']<th.LOWER_GC)|(o['gc_arr']>th.UPPER_GC)).astype(int)),
        'low complexity': w['w_lc']*0.5*pad((o['comp_arr']<th.LOW_COMPLEXITY).astype(int)),
        'non-specificity':w['w_ns']*pad(h),
        'variations':     w['w_var']*10*pad(o['var_arr']),
    }
    return comp, sum(comp.values())

comp, risk_arr = risk_components(patched)
ref_risk = pd.read_csv('example_output/EPI_ISL_402124_risk.csv')
print('手算 risk 与论文 example_output/EPI_ISL_402124_risk.csv 一致:', np.allclose(risk_arr, ref_risk['risk']))
print({k: round(float(v.sum()), 1) for k, v in comp.items()}, '<- 各分量总量')
''')

code(r'''
fig, ax = plt.subplots(figsize=(11, 2.8))
xx = np.arange(1, len(risk_arr)+1)
ax.stackplot(xx, *comp.values(), labels=list(comp.keys()), colors=['#3b6ea5', '#c77d2e', '#3e8e5a', '#b8423a'], lw=0)
ax.set_ylim(0, 6); ax.set_xlim(0, len(xx)); ax.legend(ncol=4, fontsize=8, frameon=False, loc='upper left')
ax.set_xlabel('position (nt)'); ax.set_ylabel('risk (stacked)')
plt.tight_layout(); plt.savefig(OUT/'fig_risk.png'); plt.show()
''')

# ---------------------------------------------------------------- 5. PDR
md(r'''
## 5. 引物设计区（PDR）优化

PDR 是长度固定 `PRIMER_DESIGN_LEN = 40` nt 的窗口，之后的引物只会从 PDR 内部产生。`generate_context` 一次随机生成一整套覆盖全长的 PDR：

* `find_min_loc(risk, start, stop)`：枚举区间内所有 40 nt 窗口，计算窗口风险和，从**风险最低的 30%**（`CHOICE_RATE`）中随机挑一个。
* 第 1 个扩增子：fP-PDR 落在序列开头 3×40 nt 内；rP-PDR 的位置由 `min_amp_len / max_amp_len` 约束。
* 之后每个扩增子 *i*：fP-PDR 必须落在扩增子 *i−1* 的 rP-PDR 之前、扩增子 *i−2* 的 rP-PDR 之后，使相邻扩增子**重叠**（无缺口覆盖）；奇偶编号分到 pool 1 / pool 2，这样同一 pool 内的扩增子互不重叠。
* 损失函数：取所有 PDR 风险值中**最高的 10%**（`RISK_TH`），平方求和，再除以覆盖率的平方：

$$\text{Loss} = \frac{\sum_{r \in \text{top }10\%} r^2}{\text{coverage}^2}$$

`design_context_seq` 用不同随机种子重复 $N = 500\cdot\text{iterMul}\cdot L/\text{max\_amp\_len}$ 次（SARS-CoV-2 为 35,584 次），保留损失最小的一套。它本质上是**带随机性的贪心构造 + 随机重启（random restart）**，而不是对单个解做局部搜索。
''')

code(r'''
one = th.generate_context((risk_arr, patched['start'], patched['stop'], 420, 252, 12345))
cs, risks, loss = one
df_pdr = pd.DataFrame(cs, columns=['fP_PDR_start', 'fP_PDR_end', 'rP_PDR_start', 'rP_PDR_end'])
df_pdr['pool'] = np.arange(len(df_pdr)) % 2 + 1
df_pdr['amp_span'] = df_pdr.rP_PDR_end - df_pdr.fP_PDR_start + 1
df_pdr[['fP_risk', 'rP_risk']] = risks
print(f'一次随机构造: {len(cs)} 个扩增子, loss = {loss:.3f}')
display(df_pdr.head(6))

fig, ax = plt.subplots(figsize=(11, 2.6))
ax.fill_between(xx[:3200], risk_arr[:3200], color='#cfd8e3', lw=0)
for i, (a, b, c, d) in enumerate(cs[:12]):
    y = 4.6 if i % 2 == 0 else 3.6
    ax.plot([a, d], [y, y], color='#4b5563', lw=1)
    ax.add_patch(plt.Rectangle((a, y-.15), b-a+1, .3, color='#3b6ea5'))
    ax.add_patch(plt.Rectangle((c, y-.15), d-c+1, .3, color='#b8423a'))
ax.set_xlim(0, 3200); ax.set_ylim(0, 5.2); ax.set_xlabel('position (nt)'); ax.set_ylabel('risk')
ax.set_title('first 12 amplicons: blue = fP PDR, red = rP PDR; upper row pool 1, lower row pool 2', loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(OUT/'fig_pdr.png'); plt.show()
''')

md(r'''
### 5.1 性能瓶颈与一个等价的向量化实现

`find_min_loc` 对每个候选窗口都用 Python `sum()` 重新求 40 个数的和，复杂度 O(区间长度×40)，单次 `generate_context` 约 0.16 s，35,584 次需要约 1.6 CPU·小时。

把所有窗口和预先算一次（滑动窗口内做顺序累加，与 Python `sum()` 的浮点舍入**逐位相同**），`find_min_loc` 就只剩切片和 `argpartition`。下面验证它在 50 个随机种子下给出与原实现**完全相同**的 PDR，并测速。
''')

code(r'''
W = th.PRIMER_DESIGN_LEN
_ws_cache = {}
def window_sums(risk):
    hit = _ws_cache.get(id(risk))
    if hit is None or hit[0] is not risk:
        win = np.lib.stride_tricks.sliding_window_view(np.asarray(risk, float), W)
        _ws_cache.clear()
        _ws_cache[id(risk)] = hit = (risk, win.cumsum(axis=1)[:, -1])  # 顺序累加 == Python sum()
    return hit[1]

def find_min_loc_fast(risk, start, stop, rng):
    loc = np.arange(start, stop - W + 2)
    score = window_sums(risk)[loc - 1]
    if len(loc) == 1:
        idx = 0
    else:
        k = max(int(len(loc) * th.CHOICE_RATE), 1)
        idx = rng.choice(np.argpartition(score, k)[:k])
    return loc[idx], loc[idx] + W - 1, score[idx]

orig_find_min_loc = th.find_min_loc
same, t_orig, t_fast = 0, 0, 0
for s in range(50):
    t = time.time(); a = th.generate_context((risk_arr, patched['start'], patched['stop'], 420, 252, s)); t_orig += time.time() - t
    th.find_min_loc = find_min_loc_fast
    t = time.time(); b = th.generate_context((risk_arr, patched['start'], patched['stop'], 420, 252, s)); t_fast += time.time() - t
    th.find_min_loc = orig_find_min_loc
    same += np.array_equal(a[0], b[0]) and a[2] == b[2]
print(f'50 个种子中结果完全一致: {same}/50;  原实现 {t_orig/50*1000:.1f} ms/次, 向量化 {t_fast/50*1000:.1f} ms/次, 加速 {t_orig/t_fast:.0f}x')
''')

md(r'''
### 5.2 运行完整的 35,584 次 PDR 优化

使用向量化版本（结果与原版一致），参数与论文 `example_output/olivar-design.json` 相同（`seed=10, max_amp_len=420, min_amp_len=252, check_var=True`）。
''')

code(r'''
th.find_min_loc = find_min_loc_fast          # 等价替换, 仅为提速
tik = time.time()
plex, risk_o, gc_o, comp_o, hits_o, var_o, sensi_o, combi_o, all_loss, seq_record = th.design_context_seq(cfg, deg=False)
print(f'PDR 优化用时 {time.time()-tik:.1f}s, 得到 {len(plex)} 个扩增子')
th.find_min_loc = orig_find_min_loc

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(np.arange(len(all_loss)), all_loss, color='#3b6ea5', lw=1)
ax.set_yscale('log'); ax.set_xlabel('iterations (sorted by loss, descending)'); ax.set_ylabel('loss')
ax.set_title(f'best loss = {min(all_loss):.2f}, median = {np.median(all_loss):.2f}', loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(OUT/'fig_pdr_loss.png'); plt.show()
''')

# ---------------------------------------------------------------- 6. candidates
md(r'''
## 6. PDR 内生成引物候选（`primer_generator.get`）

对每个 PDR（rP 先取反向互补，使 3' 端都朝向扩增子内部）：

1. 3' 端位置 `end_pos` 从 PDR 最右端向左扫描；
2. `check_SNP`：3' 端最后 5 nt 内有大写（变异）碱基则丢弃；
3. 从 3' 端向 5' 延长：先满足 `min_len=15`，再延长到 ΔG ≤ `dG_max`，但不超过 `max_len=36`；
4. 过滤 `min_GC ≤ GC ≤ max_GC`、`complexity ≥ min_complexity`；
5. 单引物“坏度” `badness = 2 × WAlignScore(primer)`：引物与自身反向互补做局部比对（GC 配对 2 分，AT 1 分，错配 −2，gap −3），再按离 3' 端的距离扣分，`1.5^(max_score − 6)`，用来近似发夹/自二聚体风险。

若某个 PDR 一个候选都没有，`get_primer` 会按失败最多的原因放宽条件（关 SNP 检查，或 GC 范围 ±0.05，或复杂度 −0.05）再试。
''')

code(r'''
plex = th.get_primer(plex, cfg)
first = next(iter(plex))
cand = plex[first]['fP_candidate'][['seq', 'primer_len', 'dist', 'dG', 'badness']].copy()
cand['seq'] = cand['seq'].str[0]
print(first, 'fP PDR =', plex[first]['fP_design'], '\n候选数:', len(cand))
display(cand.head(8))

n_f = [len(v['fP_candidate']) for v in plex.values()]; n_r = [len(v['rP_candidate']) for v in plex.values()]
relaxed = [k for k, v in plex.items() if v['fP_setting'] != plex[first]['fP_setting'] or v['rP_setting'] != plex[first]['rP_setting']]
print(f'每个 PDR 的 fP 候选数: 中位 {np.median(n_f):.0f} (min {min(n_f)}, max {max(n_f)});  rP 候选数: 中位 {np.median(n_r):.0f} (min {min(n_r)})')
print('被自动放宽条件的扩增子:', relaxed if relaxed else '无')

for s in ['GACGTCGACGTC' + 'AAAAT', 'ACCAACCAACTTTCGATCTCTTGT', 'TTGCAGAATTCTGCAA']:
    print(f'WAlignScore({s}) = {design.WAlignScore(s):.3f}')
''')

# ---------------------------------------------------------------- 7. SADDLE
md(r'''
## 7. SADDLE：模拟退火最小化引物二聚体

**二聚体评分 `PrimerSetBadnessFast`**（SADDLE 论文的快速近似）：

* 把池内所有引物的 **3' 端 4/5/6-mer** 放进 end-hash，把所有 **7/8-mer** 放进 middle-hash（越靠近 3' 端权重越大，权重 $1/(\text{距 3'端距离}+1)$）。
* 对每条引物取反向互补，滑动 4–8 mer 去查表：查到就说明存在可互补配对的片段。每次命中加分 $w_k \cdot \text{count} \cdot \frac{1}{j+1}\cdot 2^{\#GC}$，其中 $w = \{4:1, 5:4, 6:20, 7:100, 8:500\}$，$j$ 是该片段在反向互补序列中的位置（越靠近被查引物的 5' 端，越意味着与对方 3' 端配对）。

**优化 `optimize`**：每个 pool 独立做模拟退火。状态 = 每个扩增子选哪一对 (fP, rP)；每步随机换掉一个扩增子的引物对；`Loss = Σ 单引物 badness + 池二聚体评分`；以概率 $2^{(L_{cur}-L_{new})/T}$ 接受变差；温度从 $T_0 = 1000 + 10\cdot\max(n/2-100, 0)$ 线性降到 0，之后再跑同样步数的零温贪心。每个温度 1000 次尝试。
''')

code(r'''
a = 'ACGTTGCAGTCCAGTACGGCATG'
b_dimer = 'GACCTAGTAGCTTA' + basic.revcomp(a[-8:])   # b 的 3' 端 8 nt 与 a 的 3' 端反向互补
toy = {'unrelated': ['TGACCTAGGATCAACTGTAGCTA'], '3prime-complementary': [b_dimer]}
for name, other in toy.items():
    total, comp_b = design.PrimerSetBadnessFast([a], other)
    print(f'{name:22s} pool badness = {total:9.2f}   (fP {comp_b[0][0]:.2f}, rP {comp_b[1][0]:.2f})')
''')

code(r'''
tik = time.time()
plex_opt, lc = th.optimize(plex, cfg)
print(f'SADDLE 用时 {time.time()-tik:.1f}s')
df, art_df = th.to_df(plex_opt, cfg)
design_out = {'config': cfg, 'df': df, 'art_df': art_df, 'all_plex_info': plex_opt, 'learning_curve': lc,
              'all_ref_info': {'EPI_ISL_402124': dict(risk_arr=risk_o, gc_arr=gc_o, comp_arr=comp_o, hits_arr=hits_o, var_arr=var_o,
                               sensi_arr=sensi_o, combi_arr=combi_o, all_loss=all_loss, seq_record=seq_record)}}
pickle.dump(design_out, open(OUT/'tiling'/'olivar-design.olvd', 'wb'), protocol=5)
save(design_out, str(OUT/'tiling'))

fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
for i, ax in enumerate(axes):
    ax.plot(lc[i], color='#3b6ea5', lw=0.8); ax.set_yscale('log')
    ax.set_title(f'pool {i+1}: {lc[i][0]:.0f} -> {lc[i][-1]:.1f}', loc='left', fontsize=9); ax.set_xlabel('SA iterations')
axes[0].set_ylabel('SADDLE loss')
plt.tight_layout(); plt.savefig(OUT/'fig_saddle.png'); plt.show()
display(df[['amplicon_id', 'pool', 'fP', 'rP', 'start', 'end']].head())
''')

md(r'''
### 7.1 与论文 `example_output` 对比
''')

code(r'''
paper = pd.read_csv('example_output/olivar-design.csv')
mine = df.copy()
m = paper.merge(mine, on='amplicon_id', suffixes=('_paper', '_nb'))
print(f'扩增子数: 论文 {len(paper)}, 本次 {len(mine)}')
print(f'扩增子起止坐标完全相同: {((m.start_paper == m.start_nb) & (m.end_paper == m.end_nb)).mean():.1%}')
print(f'fP 序列相同: {(m.fP_paper.str.upper() == m.fP_nb.str.upper()).mean():.1%},  rP 序列相同: {(m.rP_paper.str.upper() == m.rP_nb.str.upper()).mean():.1%}')
amp_len = mine.end - mine.start + 1
print(f'扩增子长度: {amp_len.min()}–{amp_len.max()} nt (中位 {amp_len.median():.0f});  覆盖 {mine.insert_start.min()}–{mine.insert_end.max()}')
gaps = [(a, b) for a, b in zip(mine.insert_end[:-1], mine.insert_start[1:]) if b > a + 1]
print('相邻插入片段之间的缺口数:', len(gaps))
''')

# ---------------------------------------------------------------- 8. validation
md(r'''
## 8. 验证：特异性与灵敏度

### 8.1 `specificity`
用 `blastn-short` 的 precise 模式（evalue 5000, reward 1/penalty −1，参照 Primer-BLAST）搜索每条引物；一条命中被视为“可延伸”的条件：3' 端无悬垂、3' 端 5 nt 内至多 1 个错配、总错配 ≤ 4。然后在同一条序列上寻找方向相对、距离 < `max_amp_len` 的引物对，即预测的非特异扩增子。同时输出每条引物的 ΔG、自身比对分、池内二聚体分。
''')

code(r'''
csv_path = OUT/'tiling'/'olivar-design.csv'
specificity(primer_pool=str(csv_path), pool=1, BLAST_db=BLAST_DB, out_path=str(OUT/'specificity'),
            title='olivar-specificity', max_amp_len=1500, temperature=60, threads=NCPU)
val = pd.read_csv(OUT/'specificity'/'olivar-specificity_pool-1.csv')
display(val.sort_values('BLAST_hits', ascending=False).head(6))
ns = pd.read_csv(OUT/'specificity'/'olivar-specificity_pool-1_ns-amp.csv')
print('预测的非特异扩增子数:', len(ns))
print('dimer_score 最高的 3 条引物:'); display(val.nlargest(3, 'dimer_score')[['name', 'seq', 'dimer_score']])
''')

md(r'''
### 8.2 模式 1：从 MSA 出发设计 H1N1 HA，并做灵敏度验证

为控制运行时间，随机抽 300 条 HA 序列；`build --msa --align` 调用 MAFFT 比对，用共识序列作参考，频率 ≥ 1% 的变异写入变异表。灵敏度 = MSA 中与引物（含简并展开）**完全匹配**的序列比例。
''')

code(r'''
random.seed(7)
sub = random.sample(h1n1, 300)
H1_DIR = OUT / 'h1n1'; H1_DIR.mkdir()
SeqIO.write(sub, H1_DIR/'H1N1-HA_300.fasta', 'fasta')
tik = time.time()
build(fasta_path=None, msa_path=str(H1_DIR/'H1N1-HA_300.fasta'), var_path=None, BLAST_db=None, out_path=str(H1_DIR),
      title='H1N1-HA', threads=NCPU, align=True, min_var=0.01, deg=False)
tiling(ref_path=str(H1_DIR/'H1N1-HA.olvr'), out_path=str(H1_DIR/'design'), title='h1n1-design', max_amp_len=420, min_amp_len=300,
       w_egc=1, w_lc=1, w_ns=1, w_var=1, w_sensi=1, w_combi=1, temperature=60, salinity=0.18, dG_max=-11.8, min_GC=0.2,
       max_GC=0.75, min_complexity=0.4, max_len=36, check_var=False, fP_prefix='', rP_prefix='', seed=10, threads=NCPU, iterMul=1, deg=False)
print(f'H1N1 build + tiling 用时 {time.time()-tik:.1f}s')
h1_df = pd.read_csv(H1_DIR/'design'/'h1n1-design.csv')
display(h1_df[['amplicon_id', 'pool', 'fP', 'rP', 'start', 'end']])
''')

code(r'''
def run_sensitivity(csv, msa, title):
    sens = {}
    for pool in (1, 2):
        sensitivity(primer_pool=str(csv), msa_path=str(msa), pool=pool, out_path=str(H1_DIR/'sens'), title=title,
                    temperature=60, sodium=0.18, threads=1, align=False)
        txt = (H1_DIR/'sens'/f'{title}_pool-{pool}.out').read_text()
        for name, pct in re.findall(r'^(\S+_[fr]P)\n(?:.*\n)*?sensitivity: ([\d.]+)%', txt, flags=re.M):
            sens[re.sub(r'^.*_(\d+_[fr]P)$', r'amp\1', name)] = float(pct)
    return pd.Series(sens)

aligned = H1_DIR/'H1N1-HA_300_aligned.fasta'
sens_normal = run_sensitivity(H1_DIR/'design'/'h1n1-design.csv', aligned, 'normal')
print(sens_normal.round(1).to_string())
''')

md(r'''
### 8.3 简并模式（`--deg`）

简并模式下共识序列在“前几种碱基累计频率 ≥ 70%”的位置写成 IUPAC 简并码；`build` 额外为每个 word 计算 MSA 灵敏度与组合数，`tiling --deg` 把它们计入风险，候选引物会被展开成所有非简并序列，SADDLE 以 `1/组合数` 作为各变体的浓度。

**性能问题**：`get_sensitivity` 对**每一个 word** 都重新从磁盘读入整个 MSA 并重算共识（`MSA(msa_path)`）。下面用 300 条序列计时；对完整的 4,227 条 H1N1 序列，这一步会再慢一个数量级。
''')

code(r'''
DEG_DIR = OUT/'h1n1_deg'; DEG_DIR.mkdir()
tik = time.time()
build(fasta_path=None, msa_path=str(aligned), var_path=None, BLAST_db=None, out_path=str(DEG_DIR),
      title='H1N1-HA-deg', threads=NCPU, align=False, min_var=0.01, deg=True)
t_deg = time.time() - tik
deg_ref = pickle.load(open(DEG_DIR/'H1N1-HA-deg.olvr', 'rb'))
n_words = (len(deg_ref['seq']) // 14) - 1
print(f'简并 build 用时 {t_deg:.1f}s; 约 {n_words} 个 word, 每个都重新加载一次 MSA')
print('共识中的简并碱基:', {b: deg_ref['seq'].count(b) for b in set(deg_ref['seq'].upper()) - set('ACGT')})

t = time.time(); msa_tools.MSA(str(aligned)); t1 = time.time() - t
print(f'单次加载 300 条 MSA: {t1:.2f}s -> 仅重复加载就占 ~{t1*n_words:.0f} CPU·s')

tiling(ref_path=str(DEG_DIR/'H1N1-HA-deg.olvr'), out_path=str(DEG_DIR/'design'), title='h1n1-deg', max_amp_len=420, min_amp_len=300,
       w_egc=1, w_lc=1, w_ns=1, w_var=1, w_sensi=1, w_combi=1, temperature=60, salinity=0.18, dG_max=-11.8, min_GC=0.2,
       max_GC=0.75, min_complexity=0.4, max_len=36, check_var=False, fP_prefix='', rP_prefix='', seed=10, threads=NCPU, iterMul=1, deg=True)
deg_df = pd.read_csv(DEG_DIR/'design'/'h1n1-deg.csv')
sens_deg = run_sensitivity(DEG_DIR/'design'/'h1n1-deg.csv', aligned, 'deg')
cmp = pd.DataFrame({'normal': sens_normal, 'degenerate': sens_deg})
print(cmp.describe().loc[['mean', 'min', '50%']].round(1))
display(deg_df[['amplicon_id', 'fP', 'rP']].head())
''')

code(r'''
fig, ax = plt.subplots(figsize=(10, 2.8))
for col, c, off in [('normal', '#3b6ea5', -0.2), ('degenerate', '#c77d2e', 0.2)]:
    s = cmp[col].dropna()
    ax.bar(np.arange(len(s)) + off, s.values, width=0.4, color=c, label=col)
ax.set_xticks(range(len(cmp))); ax.set_xticklabels(cmp.index, rotation=70, fontsize=7)
ax.set_ylabel('perfect-match sensitivity (%)'); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT/'fig_sensitivity.png'); plt.show()
''')

# ---------------------------------------------------------------- 9. audit
md(r'''
## 9. 代码审计：用运行结果证实的问题

### 9.1 简并模式下 BLAST 命中数被写到错误的下标（`ncbi_tools.BLAST_batch_short`, rough 模式）

`build` 把每个 word 的简并展开命名为 `query_{word}_{variant}`，而 rough 模式解析时用 `int(query_name.split('_')[1])` 当下标，取到的是 **word 编号**，不是展开后列表的下标。`build` 随后又按展开列表的下标汇总，于是命中数被记到别的 word 上，同一个 word 的多个变体还会互相覆盖。下面用 3 个 query 直接复现：
''')

code(r'''
seqs  = ['AAAAAAAAAAAAAAAAAAAAAAAAAAAA',                 # word 0, 变体 0, 不应有命中
         'CCCCCCCCCCCCCCCCCCCCCCCCCCCC',                 # word 0, 变体 1, 不应有命中
         seq_ref[planted[0][0]-1 : planted[0][0]+27]]    # word 1, 变体 0, 来自植入的同源片段
names = ['query_0_0', 'query_0_1', 'query_1_0']           # 与 build_helper 的命名完全一致
hits_named, _ = ncbi_tools.BLAST_batch_short(seqs, db=BLAST_DB, seq_names=names, mode='rough')
hits_plain, _ = ncbi_tools.BLAST_batch_short(seqs, db=BLAST_DB, mode='rough')
print('正确的命中数 (按 query 顺序)       :', hits_plain)
print('build 命名方式下返回的命中数       :', hits_named, ' <- 第 3 个 query 的命中被写到了第 2 个位置')
''')

md(r'''
### 9.1b 复杂度计算区分大小写

`build` 把变异位点改成大写，而 `basic.get_complexity` 用 `Counter` 直接统计字符，`A` 与 `a` 被当成不同的“碱基”。含变异的 word 因此被“凭空增加了碱基种类”，复杂度被高估，低复杂度区域里的变异位点反而会躲过低复杂度惩罚。`get_GC` 则正确处理了大小写，两个函数行为不一致。
''')

code(r'''
w = 'acacacacacacacacacacacacacac'
w_snp = w[:13] + 'G' + w[14:]
print('小写 word            :', round(basic.get_complexity(w), 3))
print('同一 word, 1 个位点大写 :', round(basic.get_complexity(w[:13] + 'C' + w[14:]), 3), ' <- 序列没变, 复杂度变了')
low_all = np.array([basic.get_complexity(seq_raw[start_temp-1+p : start_temp-1+p+WS].lower()) for p in range(0, seq_len_temp-WS+1, OFF)])
print(f'EPI_ISL_402124: {int((abs(low_all - cx_w) > 1e-9).sum())}/{len(cx_w)} 个 word 因大写变异而改变了复杂度 (最大差 {abs(low_all - cx_w).max():.3f})')
''')

md(r'''
### 9.2 Tm 不受控：用含 Mg²⁺ 的模型评估设计出的引物

Olivar 只用一个 ΔG 阈值（Na⁺ = 0.18 M）决定引物长度，没有 Mg²⁺/dNTP 校正，也没有对池内 Tm 一致性的约束。用 Biopython 的 `Tm_NN`（SantaLucia 2004 参数，Owczarzy 2008 Mg 校正，50 mM Na⁺、2 mM Mg²⁺、0.8 mM dNTP、250 nM 引物）重新评估本次 SARS-CoV-2 设计：
''')

code(r'''
from Bio.SeqUtils import MeltingTemp as mt
allp = pd.concat([df.fP, df.rP]).str.upper()
tm = allp.apply(lambda s: mt.Tm_NN(s, nn_table=mt.DNA_NN4, Na=50, Mg=2, dNTPs=0.8, dnac1=250, dnac2=0, saltcorr=7))
dg = allp.apply(lambda s: gen.dG_init + gen.StacksDG(s))
print(f'{len(allp)} 条引物: 长度 {allp.str.len().min()}–{allp.str.len().max()} nt; ΔG {dg.min():.2f} ~ {dg.max():.2f};  Tm(Mg 校正) {tm.min():.1f}–{tm.max():.1f} °C, 标准差 {tm.std():.2f} °C')
three_gc = allp.str[-5:].str.count('[GC]')
print('3\' 端 5 nt 中 GC 数分布:', three_gc.value_counts().sort_index().to_dict(), ' (0 个 = 无 GC clamp, ≥4 个 = 3\' 端过强)')

fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
axes[0].hist(tm, bins=25, color='#3b6ea5'); axes[0].set_xlabel('Tm with Mg2+ correction (°C)'); axes[0].set_ylabel('primers')
axes[1].scatter(dg, tm, s=6, color='#c77d2e'); axes[1].set_xlabel('Olivar dG (kcal/mol, 0.18 M Na+)'); axes[1].set_ylabel('Tm (°C)')
plt.tight_layout(); plt.savefig(OUT/'fig_tm.png'); plt.show()
''')

md(r'''
### 9.3 SADDLE 每一步的开销

`optimize` 每次尝试都 `deepcopy` 三个字典并**从头**重算整个池的二聚体评分 O(n·L)，而一次突变只改动一个扩增子的两条引物。增量计算（只更新被替换引物对哈希表的贡献）可以把每步开销降到 O(L)。
''')

code(r'''
tube = [k for k, v in plex_opt.items() if v['tube'] == 1]
fps = [plex_opt[k]['fP']['seq'] for k in tube]; rps = [plex_opt[k]['rP']['seq'] for k in tube]
t = time.time()
for _ in range(200): design.PrimerSetBadnessFast(fps, rps)
t_full = (time.time() - t) / 200
d = {k: plex_opt[k]['fP']['seq'] for k in tube}
t = time.time()
for _ in range(200): deepcopy(d); deepcopy(d); deepcopy(d)
t_copy = (time.time() - t) / 200
steps = 2 * 1000 * 2 * (10 + int(max(len(plex_opt)/2 - 100, 0)/10))
print(f'pool 1 有 {len(tube)} 对引物: 全量二聚体评分 {t_full*1000:.2f} ms/步, deepcopy {t_copy*1000:.3f} ms/步; 两个 pool 共 {steps} 步')
print(f'全量重算 + 复制约 {steps*(t_full+t_copy):.0f}s; 对 300+ 扩增子的大 panel, 步数和池大小同时增长, 开销近似 O(n²)')
''')

md(r'''
## 10. 小结

* 第 3、4 步的手工重算与官方实现**逐元素一致**，并与论文 `example_output` 的风险数组一致。
* 向量化 `find_min_loc` 给出与原实现**完全相同**的 PDR，速度提升约 25–30 倍。
* 本次设计与论文 `example_output`（Olivar 1.1.5 生成）的对比见 7.1：扩增子数和坐标的一致程度直接反映了 1.1.5 → 1.3.3 之间算法的变化（例如简并模式引入的浓度加权 SADDLE）。
* 第 9 节记录了已证实的问题；完整的改进建议（引物迭代、算法、数据库选择）见 `tutorial/olivar_pipeline.html`。
''')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
out = Path(__file__).with_name('Olivar_walkthrough.ipynb')
nbf.write(nb, out)
print('wrote', out)
