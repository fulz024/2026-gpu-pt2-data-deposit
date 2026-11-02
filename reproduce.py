"""Recreate the numerical figures from the accompanying source data."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from verify import verify as check_data, write_tables, variance_summary

BASE = Path(__file__).resolve().parent
SETTINGS = json.loads((BASE / "plot_settings.json").read_text(encoding="utf-8"))
BLUE, PURPLE, GOLD = (SETTINGS["colors"][k] for k in ("blue", "purple", "gold"))
INK = MUTED = "#000000"
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
OUT = None
FORMATS = ("pdf", "png")

def read_csv(name):
    with (BASE / "data" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def save(fig, name, **kwargs):
    for extension in FORMATS:
        fig.savefig(OUT / (name + "." + extension), dpi=450, facecolor="white", **kwargs)
    plt.close(fig)

def geometry():
    rows = read_csv("figure5_geometries.csv")
    structures = []
    for system in ("feo", "fep"):
        selected = sorted((r for r in rows if r["system"] == system), key=lambda r: int(r["atom"]))
        atoms = [r["element"] for r in selected]
        xyz = np.array([[float(r[axis + "_angstrom"]) for axis in "xyz"] for r in selected])
        structures.append((atoms, xyz))
    return structures

def science_points():
    rows = read_csv("figure5.csv")
    return [dict(case=r["case"], Nvar=int(r["Nvar"]), energy_Ha=float(r["energy_Ha"]),
                 E2_Ha=float(r["E2_Ha"]), reported_SE_Ha=float(r["reported_SE_Ha"])) for r in rows]
def style():
    plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.spines.bottom': True, 'axes.spines.left': True, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'axes.grid': False, 'axes.linewidth': 0.65, 'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 3, 'ytick.major.size': 3, 'lines.linewidth': 1.15, 'lines.markersize': 4, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'savefig.facecolor': 'white'})

def panel(ax, letter, title):
    ax.text(-0.02, 1.035, f'({letter})', transform=ax.transAxes, ha='left', va='bottom', fontsize=9)

def figure3():
    rows = read_csv('figure3.csv')
    lookup = {(r['system'], r['method']): r for r in rows}
    systems = ['cr2', 'benzene', 'pyrazine', 'pyrimidine']
    names = ['Cr$_2$', 'Benzene', 'Pyrazine', 'Pyrimidine']
    assert len(rows) == 8
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.0))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.19, top=0.92, wspace=0.34)
    methods = [('gpu', BLUE, 'GPU sketch'), ('arrow', SETTINGS['figure3']['comparison_color'], 'Arrow')]
    x = np.arange(len(systems))
    export = []
    for (method, color, label), offset in zip(methods, [-0.18, 0.18]):
        times = [float(lookup[s, method]['seconds']) for s in systems]
        errors = [float(lookup[s, method]['reported_SE_microHa']) for s in systems]
        a.bar(x + offset, times, width=0.32, color=color, edgecolor=color, linewidth=0.55, label=label, zorder=2)
        b.bar(x + offset, errors, width=0.32, color=color, edgecolor=color, linewidth=0.55, zorder=2)
        for system, seconds, error in zip(systems, times, errors):
            export.append(dict(system=system, method=method, seconds=seconds, reported_SE_microHa=error))
    a.set_yscale('log')
    a.set_ylim(10, 50000)
    a.set_yticks([10, 100, 1000, 10000], ['10', '100', '1,000', '10,000'])
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel('Complete PT2 time (s)', labelpad=6)
    for position, system in zip(x, systems):
        arrow_time = float(lookup[system, 'arrow']['seconds'])
        gpu_time = float(lookup[system, 'gpu']['seconds'])
        a.text(position + 0.18, arrow_time * 1.28, f'{arrow_time/gpu_time:.2f}\u00d7',
               ha='center', va='bottom', fontsize=8.5, color=INK, zorder=5)
    b.axhline(10, color=INK, ls=(0, (3, 2)), lw=0.8, zorder=3)
    b.text(0.97, 10.2, '10 $\\mu$Ha', transform=b.get_yaxis_transform(), ha='right', va='bottom', fontsize=7.5)
    b.set_ylim(0, 12.5)
    b.set_yticks([0, 2, 4, 6, 8, 10, 12])
    b.set_ylabel('Reported SE ($\\mu$Ha)')
    for ax, letter in zip([a, b], 'ab'):
        ax.set_xticks(x, names)
        ax.set_xlim(-0.55, 3.55)
        ax.tick_params(axis='x', labelsize=8)
        panel(ax, letter, '')
    a.legend(loc='upper left', fontsize=7.5, labelspacing=0.3, borderaxespad=0.45, handlelength=1.2, handletextpad=0.5)
    save(fig, 'figure3', bbox_inches='tight', pad_inches=0.08)

def figure4(include_si=True):
    rows = read_csv('figure4.csv')
    assert len(rows) == 26 and all((int(r['repeats']) == 5 for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw={'width_ratios': [1, 1.22]})
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.2, top=0.92, wspace=0.34)
    colors = SETTINGS['figure4']['reference_colors']
    markers = ['o', 's', '^']
    supports = [29715, 100200, 298352]
    comparisons = []
    memory_bars = []
    for group, (n, color, marker) in enumerate(zip(supports, colors, markers)):
        subset = lambda mode: sorted([r for r in rows if int(r['Nvar']) == n and r['mode'] == mode], key=lambda r: int(r['GPUs']))
        for mode in ['sketch']:
            r = subset(mode)
            x = np.array([int(p['GPUs']) for p in r])
            t = np.array([float(p['median_seconds']) for p in r])
            low = np.array([float(p['min_seconds']) for p in r])
            high = np.array([float(p['max_seconds']) for p in r])
            ls, face = ('-', color) if mode == 'sketch' else ((0, (3, 2)), 'white')
            for ax, values in [(axes[0], t)]:
                ax.plot(x, values, color=color, marker=marker, ls=ls, mfc=face, mec=color, mew=0.8, lw=1.25 if mode == 'sketch' else 0.85, zorder=3 if mode == 'sketch' else 2)
            axes[0].errorbar(x, t, yerr=[t - low, high - t], color=color, fmt='none', lw=0.65, capsize=1.8, zorder=1)
        sketch = {int(p['GPUs']): p for p in subset('sketch')}
        exact = subset('exact')
        for p in exact:
            s = sketch[int(p['GPUs'])]
            memory = float(p['peak_allocated_GiB']) / float(s['peak_allocated_GiB'])
            time = float(p['median_seconds']) / float(s['median_seconds'])
            comparisons.append(dict(Nvar=n, GPUs=int(p['GPUs']), exact_over_sketch_memory=memory, exact_over_sketch_seconds=time))
        for index, gpu in enumerate([1, 2, 4, 8, 16]):
            center = group * 6 + index
            for mode, offset in [('exact', -0.18), ('sketch', 0.18)]:
                record = next((p for p in subset(mode) if int(p['GPUs']) == gpu), None)
                if record is None:
                    continue
                value = float(record['peak_allocated_GiB'])
                axes[1].bar(center + offset, value, width=0.33, facecolor='white' if mode == 'exact' else color, edgecolor=color, linewidth=0.65, zorder=3)
                memory_bars.append(dict(Nvar=n, GPUs=gpu, mode=mode, peak_allocated_GiB=value))
    for ax in axes[:1]:
        ax.set_xscale('log', base=2)
        ax.set_xlim(0.86, 19)
        ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
        ax.set_xlabel('A100 GPUs')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.011, 2.1)
    axes[0].set_yticks([0.02, 0.1, 0.5, 2], ['0.02', '0.1', '0.5', '2'])
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel('PT2 evaluation time (s)')
    axes[1].set_xlim(-0.7, 16.7)
    axes[1].set_ylim(0, 17.4)
    axes[1].set_yticks([0, 4, 8, 12, 16])
    positions = [group * 6 + i for group in range(3) for i in range(5)]
    axes[1].set_xticks(positions, [str(g) for _ in supports for g in [1, 2, 4, 8, 16]])
    axes[1].tick_params(axis='x', labelsize=7.5, pad=3)
    axes[1].set_xlabel('A100 GPUs')
    axes[1].set_ylabel('Peak tensor memory / GPU (GiB)')
    axes[1].legend([Patch(facecolor='white', edgecolor=INK, lw=0.65), Patch(facecolor=INK, edgecolor=INK, lw=0.65)], ['Exact', 'Sketch'], loc='upper right', ncol=2, handlelength=1.4, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.25, fontsize=7.5)
    for ax, letter, title in zip(axes, 'ab', ['Sketch strong scaling', 'Tensor-memory use']):
        panel(ax, letter, title)
    h1 = [Line2D([], [], color=c, marker=m, lw=1.1) for c, m in zip(colors, markers)]
    axes[0].legend(h1, ['29,715', '100,200', '298,352'], title='$N_{\\rm ref}$', loc='upper right', fontsize=7.5, title_fontsize=7.5, labelspacing=0.25, handlelength=1.5, handletextpad=0.45, borderaxespad=0.4)
    save(fig, 'figure4')
    if include_si:
        figure_si_exact(rows, supports, colors, markers)

def figure_si_exact(rows, supports, colors, markers):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.65), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.2, top=0.92, wspace=0.18)
    for ax, n, color, marker, letter in zip(axes, supports, colors, markers, 'abc'):
        for mode in ['exact', 'sketch']:
            selected = sorted([r for r in rows if int(r['Nvar']) == n and r['mode'] == mode], key=lambda r: int(r['GPUs']))
            x = [int(r['GPUs']) for r in selected]
            t = np.array([float(r['median_seconds']) for r in selected])
            low = np.array([float(r['min_seconds']) for r in selected])
            high = np.array([float(r['max_seconds']) for r in selected])
            ax.errorbar(x, t, yerr=[t - low, high - t], color=color, marker=marker, mfc=color if mode == 'sketch' else 'white', mec=color, mew=0.8, ls='-' if mode == 'sketch' else '--', capsize=1.6, elinewidth=0.65)
        panel(ax, letter, '$N_{\\rm ref}$ = ' + f'{n:,}')
        ax.text(0.04, 0.96, '$N_{\\rm ref}$ = ' + f'{n:,}', transform=ax.transAxes, va='top', fontsize=7.5)
        ax.set_xscale('log', base=2)
        ax.set_xlim(0.85, 19)
        ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
        ax.set_xlabel('A100 GPUs')
        ax.set_yscale('log')
        ax.set_ylim(0.011, 3.4)
        ax.set_yticks([0.02, 0.1, 0.5, 2], ['0.02', '0.1', '0.5', '2'])
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel('PT2 evaluation time (s)')
    axes[0].legend([Line2D([], [], color=INK, marker='o', mfc=INK), Line2D([], [], color=INK, marker='o', mfc='white', ls='--')], ['Sketch', 'Exact'], loc='upper right', bbox_to_anchor=(0.99, 0.85), fontsize=7.5, handlelength=1.5, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
    save(fig, 'figureS2')


def figure_si_variance():
    summary = variance_summary()
    scatter, points = summary['scatter'], summary['conditional_points']
    settings = SETTINGS['figureS1']
    blue, sand = settings['colors']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'xtick.top': False, 'ytick.right': False,
                         'xtick.direction': 'out', 'ytick.direction': 'out'}):
        fig, axes = plt.subplots(1, 2, figsize=settings['size_inches'], constrained_layout=True)
        x = np.arange(3)
        empirical = [row['empirical_SD_Ha']*1e6 for row in scatter]
        reported = [row['RMS_reported_uncertainty_Ha']*1e6 for row in scatter]
        ax = axes[0]
        for offset, values, color, label in [(-.19, empirical, blue, 'Repeated-run SD'),
                                             (.19, reported, sand, 'RMS reported uncertainty')]:
            bars = ax.bar(x+offset, values, width=.36, color=color, label=label)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xticks(x, ['Complete PT', 'Control\n(guard)', 'Sampled\ncorrection'])
        ax.set_ylabel(r'Standard deviation / SE ($\mu E_h$)')
        ax.set_ylim(0, 2.7)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.set_title('(a) Complete calculations', loc='left', fontsize=9, pad=10)
        ax = axes[1]
        b = np.array([point['B'] for point in points])
        v = np.array([point['collision_variance_Ha2'] for point in points])*1e12
        ax.loglog(b, v, 'o', color=blue, markersize=6, label='Conditional collision variance')
        fitted = np.exp(np.polyval(np.polyfit(np.log(b), np.log(v), 1), np.log(b)))
        ax.plot(b, fitted, '--', color=blue, alpha=.8,
                label=f"Fit: slope {summary['log_variance_log_B_slope']:.3f}")
        ax.axhline(points[-1]['source_batch_variance_Ha2']*1e12, color=sand, linestyle=':',
                   label='Source/batch estimate\nat largest B')
        ax.set_xscale('log', base=2)
        ax.set_xticks(b, [rf'$2^{{{int(np.log2(value))}}}$' for value in b])
        ax.set_xlabel('Number of buckets B')
        ax.set_ylabel(r'Single-iteration variance ($(\mu E_h)^2$)')
        ax.legend(frameon=False, fontsize=7.5, loc='upper right')
        ax.set_title('(b) Collision variance versus B', loc='left', fontsize=9, pad=10)
        save(fig, 'figureS1')

def molecule(ax, atoms, xyz, system):
    p = xyz - xyz[atoms.index('Fe')]

    def rotate(deg, axis):
        t = np.deg2rad(deg)
        c, s = (np.cos(t), np.sin(t))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == 'x' else np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if system == 0:
        p = p @ rotate(34, 'x').T @ rotate(-24, 'z').T
    else:
        p = p @ rotate(24, 'x').T @ rotate(10, 'z').T
    cov = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'Fe': 1.32}
    radii = {'H': 0.125, 'C': 0.19, 'N': 0.22, 'O': 0.24, 'Fe': 0.33}
    colors = {'H': '#F7F6F2', 'C': '#727B82', 'N': BLUE, 'O': '#AE777C', 'Fe': '#AB9271'}
    edges = []
    for i in range(len(atoms)):
        for j in range(i):
            if atoms[i] == atoms[j] == 'H':
                continue
            d = np.linalg.norm(xyz[i] - xyz[j])
            if 0.5 < d < 1.2 * (cov[atoms[i]] + cov[atoms[j]]):
                edges.append((i, j))
    fe = atoms.index('Fe')
    assert sum((fe in pair for pair in edges)) == (6 if system == 0 else 4)
    for i, j in edges:
        ax.plot(p[[i, j], 0], p[[i, j], 1], color='#9DAAB3', lw=1.7, solid_capstyle='round', zorder=5 + (p[i, 2] + p[j, 2]) * 0.5)
    for i in np.argsort(p[:, 2]):
        element = atoms[i]
        x, y, z = p[i]
        radius = radii[element]
        ax.add_patch(Circle((x, y), radius, facecolor=colors[element], edgecolor='#586B78', lw=0.5, zorder=5 + z + 0.05))
        ax.add_patch(Circle((x - radius * 0.27, y + radius * 0.27), radius * 0.24, facecolor='white', alpha=0.4, lw=0, zorder=5 + z + 0.06))
    ax.set_aspect('equal')
    ax.set_xlim(p[:, 0].min() - 0.5, p[:, 0].max() + 0.5)
    ax.set_ylim(p[:, 1].min() - 0.5, p[:, 1].max() + 0.5)
    ax.axis('off')

def extrapolation_inset(parent,system,points,fits,offset):
    bounds = SETTINGS['figure5']['layout'][system]['extrapolation_inset']
    ax=parent.inset_axes(bounds)
    count=len([r for r in points if r['case']==system+'_triplet'])
    for spin,color,marker in [('triplet',BLUE,'o'),('quintet',PURPLE,'s')]:
        rows=[r for r in points if r['case']==system+'_'+spin]
        x=np.array([-r['E2_Ha']*1000 for r in rows])
        y=np.array([(r['Etotal_Ha']-offset)*1000 for r in rows])
        fit=fits[system][spin]
        intercept=(fit['intercept_Ha']-offset)*1000
        xx=np.linspace(0,max(x),200)
        ax.plot(xx,intercept+fit['slope']*xx,color=color,lw=.75,ls=(0,(3,2)))
        ax.plot(x,y,ls='none',marker=marker,color=color,mfc=color,ms=2.5,mew=.45)
    ax.axvline(0,color='#AAAAAA',lw=.45,ls=':',zorder=0)
    ax.set_xlabel(r'$|E^{(2)}|$ (mHa)',fontsize=6.4,labelpad=1)
    ax.set_ylabel(r'$E-E_0$',fontsize=6.4,labelpad=1)
    ax.tick_params(labelsize=6,length=2,width=.5,pad=1.5)
    for spine in ax.spines.values():spine.set_linewidth(.5)
    if system=='feo':
        ax.set_xlim(-9,263);ax.set_ylim(-65,105)
        ax.set_xticks([0,125,250]);ax.set_yticks([-50,0,50,100])
    else:
        ax.set_xlim(-1,29);ax.set_ylim(0,28)
        ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
    return dict(points_per_state=count,gap_mHa=fits[system]['gap_mHa'],
                fit='All accepted state energies; ordinary unweighted linear regression versus abs(E2)',
                alternative_fit_or_uncertainty_displayed=False)


def reference_size_panel(ax,system,points,fits,geometry):
    offset={'feo':-1619.60,'fep':-2245.22}[system]
    for spin,color,marker in [('triplet',BLUE,'o'),('quintet',PURPLE,'s')]:
        rows=[r for r in points if r['case']==system+'_'+spin]
        x=np.array([r['Nvar']/1e6 for r in rows])
        e=np.array([r['Evar_Ha'] for r in rows])
        total=np.array([r['Etotal_Ha'] for r in rows])
        ax.plot(x,(e-offset)*1000,color=color,marker=marker,mfc='white',mec=color,
                mew=.85,ms=4.2,ls=(0,(4,2)),lw=1.05,zorder=3)
        ax.plot(x,(total-offset)*1000,color=color,marker=marker,mfc=color,mec=color,
                mew=.7,ms=4.2,ls='-',lw=1.3,zorder=3)
    ax.set_xscale('log')
    ticks=sorted({r['Nvar']//1_000_000 for r in points if r['case']==system+'_triplet'})
    ax.set_xticks(ticks,[str(n) for n in ticks]);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(.82,max(ticks)*1.2333333333333334)
    layout = SETTINGS['figure5']['layout'][system]
    ax.set_ylim(layout['y_limits_mHa'])
    ax.set_yticks(layout['y_ticks_mHa'])
    ax.set_ylabel(r'$E-E_0$ (mHa)')
    ax.set_xlabel('Reference determinants (millions)',labelpad=5)
    title=(r'[FeO(NH$_3$)$_5$]$^{2+}$'+' (60e, 187o)') if system=='feo' else 'Fe-porphyrin (40e, 42o)'
    ax.text(.075,1.039,title,transform=ax.transAxes,va='bottom',fontsize=8.7)
    ax.text(.04,.025,r'$E_0$ = '+f'{offset:.2f} Ha',transform=ax.transAxes,fontsize=7.6)
    inset=extrapolation_inset(ax,system,points,fits,offset)
    molax=ax.inset_axes(layout['molecule_inset'])
    molecule_info=molecule(molax,*geometry,0 if system=='feo' else 1)
    # Retain the original paper's single vertical PT2 illustration.
    reference_size=10_000_000
    r=next(r for r in points if r['case']==system+'_triplet' and r['Nvar']==reference_size)
    xpos=12.2
    high=(r['Evar_Ha']-offset)*1000
    low=(r['Etotal_Ha']-offset)*1000
    ax.annotate('',xy=(xpos,low+.8),xytext=(xpos,high-.8),
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=.8,mutation_scale=8))
    ax.text(xpos*(.93 if system=='feo' else 1.07),(high+low)/2,'PT2',
            fontsize=7.7,va='center',ha='right' if system=='feo' else 'left')
    return dict(offset_Ha=offset,structures=molecule_info,extrapolation=inset)


def signed_2(value):
    # Printed literature gaps can lie on a half-cent boundary after subtraction.
    rounded = Decimal(str(round(float(value), 9))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return f'{rounded:+.2f}'


def comparison_panel(fig,rect,system,points,fits,literature,letter):
    left,bottom,width,height=rect
    labels=fig.add_axes([left,bottom,width*.405,height])
    ax=fig.add_axes([left+width*.422,bottom,width*.462,height])
    values=fig.add_axes([left+width*.903,bottom,width*.10,height])
    pair=latest_pair(points,system)
    count=len([r for r in points if r['case']==system+'_triplet'])
    rows=[
        dict(label=f"SCI ({pair['Nvar']//1_000_000}M)",gap_mHa=pair['SCI_gap_mHa'],color=GOLD,marker='D',filled=False,own=True),
        dict(label=f"SCI+PT2 ({pair['Nvar']//1_000_000}M)",gap_mHa=pair['PT2_gap_mHa'],color=INK,marker='D',filled=True,own=True),
        dict(label='Extrapolated',gap_mHa=fits[system]['gap_mHa'],color=BLUE,marker='D',filled=True,own=True),
    ]
    names={
        'feo':[('HCI+PT2','22e168o, RI'),('HCI extrapolated','22e168o, RI'),('CCSD(T)/CBS','outer-core correction')],
        'fep':[('ASCI+PT2 (5M)',''),('ASCI extrapolated',''),('iCISCF(2) extrap.','reoptimized orbitals'),('FCIQMC–CASPT2','26e27o; different triplet')],
    }
    assert len(names[system])==len(literature[system])
    for original,(label,detail) in zip(literature[system],names[system]):
        rows.append(dict(original,label=label,detail=detail,color="#777777",
                         marker='D' if 'extrap' in label else 'o',filled=False,own=False))
    yy=np.arange(len(rows))[::-1].astype(float)
    for a in (labels,ax,values):a.set_ylim(-.60,len(rows)-.10)
    labels.axis('off');values.axis('off')
    labels.text(-.02,1.08,f'({letter})',transform=labels.transAxes,fontsize=9.2,va='bottom')
    labels.text(0,len(rows)-.13,'This work',color="#777777",fontsize=7,va='bottom')
    ax.set_yticks([]);ax.tick_params(axis='y',left=False,right=False)
    ax.tick_params(axis='x',top=False,labelsize=7.4,length=2.7)
    for name in ('top','left','right'):ax.spines[name].set_visible(False)
    ax.set_xlim((-28,2) if system=='feo' else (-5,9))
    ax.set_xticks([-25,-15,-5,0] if system=='feo' else [-4,0,4,8])
    ax.axvline(0,color='#A6A6A6',ls=':',lw=.65,zorder=0)
    ax.set_xlabel(r'$E_{\rm Q}-E_{\rm T}$ (mHa)',labelpad=4,fontsize=8.3)
    for a in (labels,ax,values):a.axhline(len(rows)-3.5,color='#CCCCCC',lw=.5)
    for y,row in zip(yy,rows):
        detail=row.get('detail','')
        labels.text(0,y+(.12 if detail else 0),row['label'],fontsize=7.35,va='center')
        if detail:labels.text(0,y-.19,detail,color="#777777",fontsize=6.3,va='center')
        color=row['color']
        ax.scatter([row['gap_mHa']],[y],s=26 if row['own'] else 21,marker=row['marker'],
                   facecolor=color if row['filled'] else 'white',edgecolor=color,lw=.85,zorder=4)
        values.text(0,y,signed_2(row['gap_mHa']),fontsize=7.3,va='center',color=color if row['own'] else INK)
        if row['label']=='FCIQMC–CASPT2':
            for a in (labels,ax,values):a.axhline(y+.51,color='#CCCCCC',lw=.5,ls=(0,(2,2)))
    return rows



def latest_pair(points, system):
    n = max(r['Nvar'] for r in points if r['case'].startswith(system + '_'))
    t, q = [next(r for r in points if r['case'] == system + '_' + s and r['Nvar'] == n)
            for s in ('triplet', 'quintet')]
    return dict(Nvar=n, SCI_gap_mHa=(q['Evar_Ha']-t['Evar_Ha'])*1000,
                PT2_gap_mHa=(q['Etotal_Ha']-t['Etotal_Ha'])*1000)


def figure5():
    points = [dict(r, Evar_Ha=r['energy_Ha'], Etotal_Ha=r['energy_Ha']+r['E2_Ha'])
              for r in science_points()]
    fits = {}
    for r in read_csv('figure5_extrapolation.csv'):
        fits.setdefault(r['system'], {})[r['state']] = dict(intercept_Ha=float(r['intercept_Ha']), slope=float(r['slope']))
        fits[r['system']]['gap_mHa'] = float(r['gap_Q_minus_T_mHa'])
    literature = {s: [dict(r, gap_mHa=float(r['gap_mHa'])) for r in read_csv('figure5_literature.csv') if r['system']==s]
                  for s in ('feo', 'fep')}
    structures = geometry()
    fig = plt.figure(figsize=(8.2, 7.0), facecolor='white')
    for i, system in enumerate(('feo', 'fep')):
        left, width = [(.09, .385), (.585, .385)][i]
        ax = fig.add_axes([left, .385, width, .54])
        reference_size_panel(ax, system, points, fits, structures[i])
        panel(ax, 'ab'[i], '')
        comparison_panel(fig, [left, .060, width, .225], system, points, fits, literature, 'cd'[i])
    handles, names = [], []
    for method in ('SCI', 'SCI+PT2'):
        for spin, color, marker in [('Triplet', BLUE, 'o'), ('Quintet', PURPLE, 's')]:
            handles.append(Line2D([], [], color=color, marker=marker, mfc='white' if method=='SCI' else color,
                                  mec=color, mew=.8, ms=4, ls='--' if method=='SCI' else '-', lw=1.1))
            names.append(spin+', '+method)
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.535, .998), ncol=4,
               fontsize=7.7, handlelength=1.8, handletextpad=.5, columnspacing=1.8, borderaxespad=.1)
    save(fig, 'figure5')

def figure6():
    rows = read_csv('fep_cost_memory.csv')
    figure_settings = SETTINGS['figure6']
    states = {state: sorted((r for r in rows if r['case'] == 'fep_'+state), key=lambda r: int(r['Nvar']))
              for state in ('triplet', 'quintet')}
    memory_rows = states[figure_settings['memory_state']]
    x = np.array([int(r['Nvar'])/1e6 for r in memory_rows])
    fonts = figure_settings['font_sizes']
    fig, axes = plt.subplots(1, 3, figsize=figure_settings['size_inches'])
    fig.subplots_adjust(**figure_settings['layout'])
    a, b, c = axes
    for state, color, marker in [('triplet', BLUE, 'o'), ('quintet', PURPLE, 's')]:
        points = states[state]
        a.plot([int(r['Nvar'])/1e6 for r in points], [float(r['PT_seconds']) for r in points],
               color=color, marker=marker, mfc=color if state == 'triplet' else 'white',
               mec=color, mew=.8, ms=4.3, label=state.capitalize())
    a.set_ylim(0, 100)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel('PT2 time (s)', fontsize=fonts['labels'], labelpad=2)
    a.legend(loc='upper left', fontsize=fonts['legend'], handlelength=1.4,
             labelspacing=.18, handletextpad=.4, borderaxespad=.35)

    bottom = np.zeros(len(x))
    for component in figure_settings['memory_components']:
        values = np.array([float(r[component['column']]) for r in memory_rows])
        b.bar(x, values, width=x*.47, bottom=bottom, label=component['label'],
              color=component['color'], edgecolor='white', linewidth=.3)
        bottom += values
    b.set_ylim(0, 10)
    b.set_yticks([0, 2, 4, 6, 8, 10])
    b.set_ylabel('Base memory (GiB)', fontsize=fonts['labels'], labelpad=2)
    b.legend(loc='upper left', fontsize=fonts['legend'], handlelength=.9,
             handletextpad=.4, labelspacing=.15, borderaxespad=.35)

    for column, label, color, marker in [
        ('control_peak_GiB', 'Control', BLUE, 'o'),
        ('stochastic_peak_GiB', 'Stochastic', PURPLE, 's')]:
        c.plot(x, [float(r[column]) for r in memory_rows], color=color, marker=marker,
               mfc=color if marker == 'o' else 'white', mec=color, mew=.8, ms=4.3, label=label)
    c.set_ylim(0, 70)
    c.set_yticks([0, 20, 40, 60])
    c.set_ylabel('Peak memory (GiB)', fontsize=fonts['labels'], labelpad=2)
    c.legend(loc='lower left', fontsize=fonts['legend'], handlelength=1.4,
             handletextpad=.4, labelspacing=.18, borderaxespad=.35)
    for ax, letter in zip(axes, 'abc'):
        ax.set_xscale('log')
        ax.set_xlim(.55, 180)
        ax.set_xticks(x, [f'{value:g}' for value in x])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=fonts['ticks'], pad=2.5)
        ax.text(-.02, 1.025, f'({letter})', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=fonts['panel'])
    fig.supxlabel('Reference determinants (million)', x=.537, y=.025, fontsize=fonts['labels'])
    save(fig, 'figure6')


def figure7():
    rows = read_csv("figure7.csv")
    systems = [("h2o_ccpvdz", r"H$_2$O"), ("nh3_ccpvdz", r"NH$_3$"), ("hcn_ccpvdz", "HCN")]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.9), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.84, wspace=.12)
    handles = [Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", linewidth=1.1,
                      markersize=3.5, label=r"Reference $E_{\mathcal{S}}$"),
               Line2D([], [], color=PURPLE, marker="s", linewidth=1.1, markersize=3.5,
                      label=r"$E_{\mathrm{NNQS+PT2}}$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=2, frameon=False, fontsize=8.5, handlelength=2, columnspacing=2.2)
    for ax, letter, (system, label) in zip(axes, "abc", systems):
        points = sorted((r for r in rows if r["system"] == system), key=lambda r: int(r["update"]))
        x = [int(r["update"]) for r in points]
        ref = [(float(r["reference_energy_Ha"]) - float(r["literature_reference_Ha"])) * 1000 for r in points]
        corrected = [(float(r["PT2_energy_Ha"]) - float(r["literature_reference_Ha"])) * 1000 for r in points]
        ax.axhline(0, color=".45", linewidth=.55, zorder=1)
        ax.plot(x, ref, color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=.9,
                markersize=3.5, linewidth=1.1, zorder=3)
        ax.plot(x, corrected, color=PURPLE, marker="s", markersize=3.2, linewidth=1.1, zorder=3)
        ax.set_xscale("log"); ax.set_xlim(80, 40000)
        ax.set_yscale("symlog", linthresh=SETTINGS["figure7"]["symlog_linthresh_mHa"], linscale=.8, base=10)
        ax.set_ylim(-2, 110)
        ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000, 30000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: {100:"100", 1000:"1k", 10000:"10k", 30000:"30k"}.get(value, "")))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(width=.65, length=3, pad=3)
        ax.text(0, 1.025, f"({letter})", transform=ax.transAxes, va="bottom", fontweight="bold")
        ax.text(.955, .96, label, transform=ax.transAxes, va="top", ha="right")
        ax.set_xlabel("Parameter updates")
        if ax is axes[0]: ax.set_ylabel(r"$E-E_{\mathrm{ref}}$ (mHa)")
    save(fig, "figure7")


def main():
    global OUT, FORMATS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "output")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    check_data()
    if args.check_data: return
    FORMATS = tuple(args.formats.split(","))
    if not FORMATS or any(x not in ("pdf", "png", "svg") for x in FORMATS):
        parser.error("Choose formats from pdf,png,svg")
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    figure3(); figure4(); figure5(); figure6(); figure7(); figure_si_variance()
    write_tables(OUT)
    print("Figures written to", OUT)

if __name__ == "__main__":
    main()
