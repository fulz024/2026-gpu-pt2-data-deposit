"""Validate the frozen paper data and regenerate numerical tables (standard library only)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
DATA_FILES = {
    'figure3.csv': 8, 'figure4.csv': 26, 'figure5.csv': 20,
    'figure5_extrapolation.csv': 4, 'figure5_literature.csv': 7,
    'figure5_geometries.csv': 59, 'fep_cost_memory.csv': 10, 'figure7.csv': 39, 'validation.csv': 20,
    'trimci.csv': 4, 'sampling.csv': 4, 'nnqs_benchmarks.csv': 3, 'hcn_components.csv': 4,
    'variance_complete_runs.csv': 64, 'variance_conditional_hashes.csv': 512,
    'variance_components.csv': 4, 'sketch_accuracy.csv': 3,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    with (BASE / 'data' / name).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, tolerance, name):
    require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance), name)


def fit(rows):
    """Centered ordinary least squares in Ha, against |E2| in Ha."""
    x = [abs(float(r['E2_Ha'])) for r in rows]
    y = [float(r['energy_Ha']) + float(r['E2_Ha']) for r in rows]
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    slope = math.fsum((a-mx)*(b-my) for a, b in zip(x, y)) / math.fsum((a-mx)**2 for a in x)
    return my - slope * mx, slope


def gaps():
    points = read('figure5.csv')
    rows = []
    for system in ('feo', 'fep'):
        for n in sorted({int(r['Nvar']) for r in points if r['case'].startswith(system + '_')}):
            t, q = [next(r for r in points if r['case'] == system+'_'+state and int(r['Nvar']) == n)
                    for state in ('triplet', 'quintet')]
            sci = (float(q['energy_Ha']) - float(t['energy_Ha'])) * 1000
            rows.append(dict(system=system, Nvar=n, SCI_gap_mHa=sci,
                PT2_gap_mHa=sci + (float(q['E2_Ha'])-float(t['E2_Ha']))*1000,
                combined_reported_SE_mHa=math.hypot(float(q['reported_SE_Ha']), float(t['reported_SE_Ha']))*1000))
    return rows


def variance_summary():
    """Recompute the displayed scatter and crossed source/hash decomposition."""
    complete = read('variance_complete_runs.csv')
    for row in complete:
        close(float(row['control_Ha']) + float(row['sampled_mean_Ha']), row['E2_Ha'],
              2e-17, 'Control plus sampled correction')
        close(float(row['control_variance_Ha2']) + float(row['tail_variance_Ha2']),
              float(row['reported_SE_Ha'])**2, 1e-26, 'Combined reported SE')
    require(len({row['seed'] for row in complete}) == 64, 'Independent complete-run seeds')
    require(all(int(row['control_passes']) == 1 and int(row['tail_windows']) == 6
                for row in complete), 'Actual variance-check stopping counts')
    quantities = [
        ('Complete PT', 'E2_Ha', None),
        ('Control', 'control_Ha', 'control_variance_Ha2'),
        ('Sampled correction', 'sampled_mean_Ha', 'tail_variance_Ha2'),
    ]
    scatter = []
    for label, value_key, variance_key in quantities:
        variance = [float(row[variance_key]) if variance_key else float(row['reported_SE_Ha'])**2
                    for row in complete]
        scatter.append(dict(component=label,
            empirical_SD_Ha=statistics.stdev(float(row[value_key]) for row in complete),
            RMS_reported_uncertainty_Ha=math.sqrt(statistics.mean(variance))))

    observations = read('variance_conditional_hashes.csv')
    components = {int(row['B']): row for row in read('variance_components.csv')}
    buckets = [2**k for k in (16, 20, 24, 28)]
    draws = sorted({int(row['draw_id']) for row in observations})
    hashes = sorted({int(row['hash_seed']) for row in observations})
    require(len(draws) == 8 and len(hashes) == 16, 'Crossed variance design')
    require(len({(row['B'], row['draw_id'], row['hash_seed']) for row in observations})
            == len(observations), 'Repeated crossed variance observations')
    for draw in draws:
        selected = [row for row in observations if int(row['draw_id']) == draw]
        require(len({(row['draw_seed'], row['partition_seed'], row['unique_sampled_sources'],
                      row['external_edges']) for row in selected}) == 1,
                'Source draw and external batch held fixed across hashes and B')
    points = []
    for b in buckets:
        records = {(int(row['draw_id']), int(row['hash_seed'])): float(row['estimate_Ha'])
                   for row in observations if int(row['B']) == b}
        require(len(records) == 128, 'Missing crossed observations')
        matrix = [[records[draw, seed] for seed in hashes] for draw in draws]
        row_means = [statistics.mean(row) for row in matrix]
        col_means = [statistics.mean(row[j] for row in matrix) for j in range(16)]
        grand = statistics.mean(row_means)
        interaction = math.fsum((matrix[i][j]-row_means[i]-col_means[j]+grand)**2
                               for i in range(8) for j in range(16)) / (7*15)
        ms_draw = 16 * statistics.variance(row_means)
        ms_hash = 8 * statistics.variance(col_means)
        source = (ms_draw-interaction)/16
        collision = statistics.mean(statistics.variance(row) for row in matrix)
        recomputed = dict(source_batch_variance_unclipped_Ha2=source,
            hash_main_variance_unclipped_Ha2=(ms_hash-interaction)/8,
            interaction_variance_Ha2=interaction,
            conditional_hash_variance_direct_Ha2=collision,
            conditional_hash_variance_ANOVA_Ha2=(ms_hash-interaction)/8+interaction,
            MS_draw=ms_draw, MS_hash=ms_hash, MS_interaction=interaction,
            mean_tail_Ha=grand, estimated_total_window_variance_Ha2=source+collision)
        for key, value in recomputed.items():
            require(math.isclose(value, float(components[b][key]), rel_tol=2e-11, abs_tol=1e-25),
                    'Crossed variance arithmetic: ' + key)
        points.append(dict(B=b, source_batch_variance_Ha2=source, collision_variance_Ha2=collision))
    x = [math.log(point['B']) for point in points]
    y = [math.log(point['collision_variance_Ha2']) for point in points]
    mx, my = statistics.mean(x), statistics.mean(y)
    slope = math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/math.fsum((a-mx)**2 for a in x)
    return dict(complete_runs=64, scatter=scatter, conditional_points=points,
                log_variance_log_B_slope=slope)


def verify():
    expected_names = {'data/' + name for name in DATA_FILES} | {'parameters.json', 'plot_settings.json'}
    require({p.name for p in (BASE/'data').iterdir()} == set(DATA_FILES), 'Unexpected or missing data file')
    checksums = {}
    for line in (BASE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(name in expected_names and name not in checksums, 'Unexpected checksum entry: ' + name)
        path = BASE / name
        require(not path.is_symlink() and path.resolve().is_relative_to(BASE), 'Unsafe data path: ' + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Checksum mismatch: ' + name)
        checksums[name] = digest
    require(set(checksums) == expected_names, 'Incomplete data checksum manifest')
    tables = {name: read(name) for name in DATA_FILES}
    for name, count in DATA_FILES.items():
        require(len(tables[name]) == count, 'Wrong row count: ' + name)
        for row in tables[name]:
            for key, value in row.items():
                if key.endswith(('_Ha', '_mHa', '_microHa', '_seconds', '_GiB')) or key == 'seconds':
                    if value:
                        require(math.isfinite(float(value)), 'Nonfinite value: ' + name + ':' + key)

    f3, f4, f5, f7 = [tables['figure'+n+'.csv'] for n in ('3', '4', '5', '7')]
    require(len({(r['system'], r['method']) for r in f3}) == 8, 'Repeated performance rows')
    require(all(float(r['seconds']) > 0 and 0 <= float(r['reported_SE_microHa']) <= 10 for r in f3), 'Invalid performance data')
    require(len({(r['Nvar'], r['mode'], r['GPUs']) for r in f4}) == 26, 'Repeated scaling rows')
    require(sum(r['mode'] == 'sketch' for r in f4) == 15, 'Missing sketch scaling rows')
    for row in f4:
        require(int(row['repeats']) == 5, 'Scaling repeat count')
        require(0 < float(row['min_seconds']) <= float(row['median_seconds']) <= float(row['max_seconds']), 'Scaling range')
        require(float(row['peak_allocated_GiB']) > 0, 'Scaling memory')
        require(int(row['nodes']) == (2 if int(row['GPUs']) == 16 else 1), 'Scaling node count')

    extrap = tables['figure5_extrapolation.csv']
    for system, sizes in [('feo', [1, 3, 10, 30, 60]), ('fep', [1, 3, 10, 30, 100])]:
        intercepts = {}
        for state in ('triplet', 'quintet'):
            selected = [r for r in f5 if r['case'] == system + '_' + state]
            require(sorted(int(r['Nvar']) for r in selected) == [n*1000000 for n in sizes], 'SCI support sequence')
            require(all(float(r['E2_Ha']) < 0 and float(r['reported_SE_Ha']) >= 0 for r in selected), 'SCI/PT2 values')
            a, b = fit(selected)
            saved = next(r for r in extrap if r['system'] == system and r['state'] == state)
            require(int(saved['points']) == len(sizes), 'Extrapolation point count')
            close(a, saved['intercept_Ha'], 2e-9, system + ' extrapolation intercept')
            close(b, saved['slope'], 1e-7, system + ' extrapolation slope')
            intercepts[state] = a
        gap = (intercepts['quintet']-intercepts['triplet'])*1000
        for saved in [r for r in extrap if r['system'] == system]:
            close(gap, saved['gap_Q_minus_T_mHa'], 2e-6, system + ' extrapolation gap')

    settings = json.loads((BASE / 'parameters.json').read_text(encoding='utf-8'))
    costs = tables['fep_cost_memory.csv']
    components = ['reference_GiB', 'index_GiB', 'sampling_GiB', 'integral_GiB', 'other_resident_GiB']
    require(len({(r['case'], r['Nvar']) for r in costs}) == len(costs), 'Repeated cost/memory rows')
    require({r['case'] for r in costs} == {'fep_triplet', 'fep_quintet'}, 'Cost/memory spin states')
    for state in ('triplet', 'quintet'):
        selected = sorted((r for r in costs if r['case'] == 'fep_'+state), key=lambda r: int(r['Nvar']))
        require([int(r['Nvar']) for r in selected] == [n*1000000 for n in (1, 3, 10, 30, 100)], 'Cost/memory support sequence')
        for row in selected:
            n = int(row['Nvar'])
            require(float(row['PT_seconds']) > 0, 'Nonpositive PT2 time')
            require(0 <= float(row['reported_SE_microHa']) <= float(row['target_SE_microHa']), 'Cost/memory SE target')
            saved = next(r for r in f5 if r['case'] == row['case'] and int(r['Nvar']) == n)
            close(float(row['reported_SE_microHa'])*1e-6, saved['reported_SE_Ha'], 1e-14, 'Cost/memory and energy SE')
            target = settings['science']['target_SE_Ha']['1M_and_3M' if n <= 3000000 else 'larger_supports']
            close(float(row['target_SE_microHa'])*1e-6, target, 1e-14, 'Cost/memory requested SE')
            require(all(float(row[key]) >= 0 for key in components), 'Negative memory component')
            close(math.fsum(float(row[key]) for key in components), row['resident_GiB'], 1e-10, 'Memory decomposition')
            for stage, bucket_key in [('control', 'control_buckets'), ('stochastic', 'tail_buckets')]:
                buckets = int(row[bucket_key])
                require(buckets > 0 and not buckets & (buckets-1), 'Sketch bucket count')
                array_gib = settings['fep_cost_memory'][stage+'_bytes_per_bucket']*buckets/2**30
                require(float(row[stage+'_peak_GiB']) >= float(row['resident_GiB'])+array_gib, 'Stage memory below array sizes')
        require(all(int(a['control_buckets']) >= int(b['control_buckets']) for a, b in zip(selected, selected[1:])), 'Control bucket sequence')
    for n in (1000000, 3000000, 10000000, 30000000, 100000000):
        triplet, quintet = [next(r for r in costs if r['case'] == 'fep_'+state and int(r['Nvar']) == n)
                           for state in ('triplet', 'quintet')]
        for key in [*components, 'resident_GiB', 'control_peak_GiB', 'stochastic_peak_GiB']:
            close(triplet[key], quintet[key], 5e-5, 'Spin-state memory agreement')

    steps = [100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 16000, 20000, 25000, 30000]
    benchmarks = {r['system']: r for r in tables['nnqs_benchmarks.csv']}
    for system, benchmark in benchmarks.items():
        selected = [r for r in f7 if r['system'] == system]
        require(sorted(int(r['update']) for r in selected) == steps, 'NNQS checkpoint sequence')
        for row in selected:
            close(row['literature_reference_Ha'], benchmark['benchmark_energy_Ha'], 1e-11, 'NNQS benchmark')
    hcn = tables['hcn_components.csv']
    for row in hcn:
        close((float(row['energy_Ha'])-float(benchmarks['hcn_ccpvdz']['benchmark_energy_Ha']))*1000,
              row['error_mHa'], 1e-8, 'HCN component error')
    final = next(r for r in f7 if r['system'] == 'hcn_ccpvdz' and int(r['update']) == 30000)
    close(final['reference_energy_Ha'], hcn[0]['energy_Ha'], 1e-11, 'HCN reference')
    close(final['PT2_energy_Ha'], hcn[-1]['energy_Ha'], 1e-11, 'HCN full correction')
    require(all(not r['reported_SE_microHa'] and float(r['seconds']) > 0 for r in tables['trimci.csv']), 'TrimCI timing records')
    for row in f3:
        saved = next(r for r in tables['validation.csv'] if r['system'] == row['system'] and r['method'] == row['method'])
        close(row['reported_SE_microHa'], saved['reported_SE_microHa'], .0051, 'Validation/performance SE')
    require(settings['nnqs']['parameter_updates'] == steps[-1], 'Fixed training budget')
    summary = variance_summary()
    close(summary['scatter'][0]['empirical_SD_Ha']*1e6, 1.916, .0005, 'Displayed complete-run SD')
    close(summary['scatter'][0]['RMS_reported_uncertainty_Ha']*1e6, 1.924, .0005, 'Displayed RMS SE')
    close(summary['log_variance_log_B_slope'], -.989, .0005, 'Displayed collision slope')
    sketch_accuracy = tables['sketch_accuracy.csv']
    require([int(row['Nref']) for row in sketch_accuracy] == [29715, 100200, 298352], 'Sketch accuracy supports')
    require(all(int(row['GPUs']) == 8 and float(row['reported_sketch_SE_microHa']) > 0
                and float(row['absolute_difference_from_exact_microHa']) >= 0 for row in sketch_accuracy),
            'Sketch accuracy records')
    print(f'PASS: {len(DATA_FILES)} data tables, {sum(DATA_FILES.values())} rows; checksums, fits, gaps and benchmark arithmetic verified.')


def write_tables(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, rows):
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    for source, target in [('validation.csv', 'table1_validation.csv'), ('sampling.csv', 'tableS1_sampling.csv'),
                           ('sketch_accuracy.csv', 'tableS3_sketch_accuracy.csv'),
                           ('nnqs_benchmarks.csv', 'tableS4_nnqs_benchmarks.csv'), ('hcn_components.csv', 'tableS5_hcn_components.csv')]:
        save(target, read(source))
    performance = {(r['system'], r['method']): r for r in read('figure3.csv') + read('trimci.csv')}
    rows = []
    for system in ('cr2', 'benzene', 'pyrazine', 'pyrimidine'):
        gpu, arrow, trimci = [performance[system, m] for m in ('gpu', 'arrow', 'trimci')]
        rows.append(dict(system=system, gpu_seconds=gpu['seconds'], arrow_seconds=arrow['seconds'],
            trimci_seconds=trimci['seconds'], gpu_reported_SE_microHa=gpu['reported_SE_microHa'],
            arrow_reported_SE_microHa=arrow['reported_SE_microHa'], trimci_reported_SE_microHa='',
            arrow_over_gpu_speedup=float(arrow['seconds'])/float(gpu['seconds'])))
    save('tableS2_performance.csv', rows)
    save('derived_spin_gaps.csv', gaps())
    save('derived_extrapolation.csv', [dict(system=s, state=state, intercept_Ha=fit([r for r in read('figure5.csv') if r['case']==s+'_'+state])[0],
         slope=fit([r for r in read('figure5.csv') if r['case']==s+'_'+state])[1])
         for s in ('feo', 'fep') for state in ('triplet', 'quintet')])
    (output/'derived_variance_summary.json').write_text(
        json.dumps(variance_summary(), indent=2)+'\n', encoding='utf-8')
    print('Numerical tables written to', output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Also regenerate paper tables into this directory')
    args = parser.parse_args()
    verify()
    if args.output:
        write_tables(args.output)
