"""Manuscript survival layout: arm medians, event counts and binary interactions."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
def number(value):
    return 'NE' if np.isnan(value) else 'NR' if not np.isfinite(value) else f'{value:.2f}'


def median_label(row):
    return number(row['median']) + '\n(' + number(row.median_ci_low) + '-' + number(row.median_ci_high) + ')'


def draw_km(fig, endpoint, col, cfg, data):
    left = 0.015 + col * 0.5
    (x0, width) = (left + 0.113, 0.352)
    ax = fig.add_axes([x0, 0.535, width, 0.385])
    for key in ['B_0', 'B_1', 'NB_0', 'NB_1']:
        curve = data['km_curves']
        curve = curve[(curve.endpoint == endpoint) & (curve.group_key == key)].sort_values('time')
        color = cfg['colors'][key]
        ax.step(curve.time, curve.survival, where='post', color=color, linestyle=cfg['linestyles'][key], linewidth=2.3 if key.startswith('NB') else 1.7)
        censored = curve[curve.censored.gt(0)]
        ax.plot(censored.time, censored.survival, linestyle='none', marker='|', color=color, markersize=6, markeredgewidth=0.85)
    ax.set_xlim(0, cfg['time_limit_months'])
    ax.set_ylim(0, 1.025)
    ax.set_xticks(cfg['time_ticks_months'])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    ax.set_xlabel('Time (months)', fontsize=13, labelpad=11)
    ax.set_ylabel(endpoint + ' probability', fontsize=13, labelpad=10)
    ax.tick_params(labelsize=11, length=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#555555')
    for y in [0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y, color='#ECEEF0', linewidth=0.55, zorder=-1)
    fig.text(left, 0.948, 'A' if col == 0 else 'C', fontsize=19, fontweight='bold')
    fig.text(x0 + width / 2, 0.952, 'Progression-free survival' if endpoint == 'PFS' else 'Overall survival', ha='center', va='center', fontsize=15, fontweight='bold')
    riskax = fig.add_axes([x0, 0.323, width, 0.116])
    riskax.set_xlim(ax.get_xlim())
    riskax.set_ylim(0, 4)
    riskax.axis('off')
    fig.text(left, 0.453, 'No. at risk (cumulative censored)', fontsize=10.5, color='#30343B')
    label_positions = []
    for (i, key) in enumerate(['B_0', 'B_1', 'NB_0', 'NB_1']):
        y = 3.5 - i
        ypos = 0.323 + y / 4 * 0.116
        fig.text(left, ypos, cfg['group_labels'][key], fontsize=10.5, color=cfg['colors'][key], fontweight='bold' if key.startswith('NB') else 'normal', va='center')
        selected = data['risk_table']
        selected = selected[(selected.endpoint == endpoint) & (selected.group_key == key)]
        for row in selected.itertuples():
            riskax.text(row.time, y, f'{row.at_risk} ({row.cumulative_censored})', ha='center', va='center', fontsize=10)
            assert abs(ax.transData.transform((row.time, 0))[0] - riskax.transData.transform((row.time, 0))[0]) < 1e-08
            label_positions.append(dict(endpoint=endpoint, group_key=key, time=row.time, curve_x_pixels=float(ax.transData.transform((row.time, 0))[0]), risk_x_pixels=float(riskax.transData.transform((row.time, 0))[0])))
    return label_positions


def draw_forest(fig, endpoint, col, cfg, data):
    left = 0.015 + col * 0.5
    fig.text(left, 0.282, 'B' if col == 0 else 'D', fontsize=19, fontweight='bold')
    fig.text(left + 0.035, 0.286, endpoint + ' treatment effects', fontsize=13, fontweight='bold', va='center')
    columns = {'subgroup': left, 'ici_events': left + 0.06, 'ici_median': left + 0.132, 'chemo_events': left + 0.204, 'chemo_median': left + 0.276, 'hr_text': left + 0.436}
    fig.text(left + 0.096, 0.25, 'ICI+CT', ha='center', fontsize=10.5, fontweight='bold')
    fig.text(left + 0.24, 0.25, 'CT', ha='center', fontsize=10.5, fontweight='bold')
    for (name, text) in [('subgroup', 'Subgroup'), ('chemo_events', 'Events/\npatients'), ('chemo_median', 'Median months\n(95% CI)'), ('ici_events', 'Events/\npatients'), ('ici_median', 'Median months\n(95% CI)'), ('hr_text', 'HR\n(95% CI)')]:
        fig.text(columns[name], 0.22, text, ha='left' if name == 'subgroup' else 'center', va='center', fontsize=10)
    fig.add_artist(Line2D([left, left + 0.472], [0.202, 0.202], transform=fig.transFigure, color='#555555', linewidth=0.8))
    ax = fig.add_axes([left + 0.323, 0.073, 0.066, 0.132])
    ax.set_xscale('log')
    estimates=data['treatment_hr']; limits=estimates[estimates.endpoint.eq(endpoint)][['ci_low','ci_high']].to_numpy()
    lower=min(.2,float(np.nanmin(limits))*.9);upper=max(4.,float(np.nanmax(limits))*1.1)
    ax.set_xlim(lower,upper)
    ax.set_ylim(-0.5, 2.5)
    ax.axvline(1, color='#AEB3B8', linewidth=0.8, linestyle='--')
    ticks=[v for v in [.05,.1,.25,.5,1,2,4,8,16,32] if lower<=v<=upper]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x', labelsize=9, rotation=45, length=3)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#B0B0B0')
    for (i, subgroup) in enumerate(['Overall', 'B', 'NB']):
        y = 2 - i
        ypos = 0.073 + (y + 0.5) / 3 * 0.132
        color = cfg['forest_colors'][subgroup]
        hr = data['treatment_hr']
        row = hr[(hr.endpoint == endpoint) & (hr.subgroup == subgroup)].iloc[0]
        if not np.isfinite(row.hr):raise ValueError('Forest estimate is not estimable; inspect the saved statistics')
        ax.errorbar(row.hr, y, xerr=[[row.hr - row.ci_low], [row.ci_high - row.hr]], fmt='s', color=color, markersize=7.5 if subgroup == 'NB' else 6, linewidth=1.2, capsize=2)
        fig.text(columns['subgroup'], ypos, subgroup, va='center', fontsize=11, fontweight='bold' if subgroup == 'NB' else 'normal', color=color)
        for (treatment, stem) in [(0, 'chemo'), (1, 'ici')]:
            summary = data['survival_summary']
            s = summary[(summary.endpoint == endpoint) & (summary.subgroup == subgroup) & (summary.treatment == treatment)].iloc[0]
            fig.text(columns[stem + '_events'], ypos, f'{int(s.events)}/{int(s.n)}', ha='center', va='center', fontsize=10)
            fig.text(columns[stem + '_median'], ypos, median_label(s), ha='center', va='center', fontsize=9.5, linespacing=1.2)
        fig.text(columns['hr_text'], ypos, f'{row.hr:.2f}\n({row.ci_low:.2f}-{row.ci_high:.2f})', ha='center', va='center', fontsize=10, linespacing=1.2)
    row = data.get('interaction', data.get('altman_bland_interaction')).set_index('endpoint').loc[endpoint]
    assert row.ratio_direction == 'HR_B / HR_NB'
    p = float(row.p)
    ratio_label = f'Ratio of treatment HRs (95% CI): {row.ratio:.2f} ({row.ci_low:.2f}-{row.ci_high:.2f})'
    fig.text(left + 0.003, 0.04, ratio_label, fontsize=10.5)
    test = 'Altman-Bland P' if 'Altman' in str(row.get('method', '')) else 'Interaction P'
    label = test+' < 0.001' if p < 0.001 else f'{test} = {p:.3f}'
    fig.text(left + 0.003, 0.018, 'B/NB; ' + label, fontsize=11)
    # Align the two direction arrows with HR=1 on the logarithmic forest axis.
    neutral = left + 0.323 + 0.066 * (math.log(1 / lower) / math.log(upper / lower))
    arrow_y = 0.016
    for tip, marker, label, ha, offset in [
            (left + 0.302, '<', 'favor ICI+CT', 'right', -0.004),
            (left + 0.413, '>', 'favor CT', 'left', 0.004)]:
        start = neutral + (-0.002 if tip < neutral else 0.002)
        fig.add_artist(Line2D([start, tip], [arrow_y, arrow_y],
                             transform=fig.transFigure, color='#30343B', linewidth=0.9,
                             marker=marker, markevery=[1], markersize=4.5))
        fig.text(tip + offset, arrow_y, label, ha=ha, va='center', fontsize=9.5)


