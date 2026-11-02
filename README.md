# GPU PT2 data deposit

Numerical source data and figure-reproduction scripts for
**Fully GPU-accelerated semistochastic Epstein–Nesbet perturbation theory**, by Lizhong Fu,
Honghui Shang and Jinlong Yang. Manuscript data snapshot: 2026-11-03.

This repository provides the numerical source data and analysis scripts for
Figures 3–7, Figures S1–S2, Table 1 and Tables S1–S5. The iron–oxo sequence extends
to 60 million reference determinants and Fe-porphyrin to 100 million.
State-specific extrapolations use all reference sizes in each sequence.

## Run

From the repository root, verify the data and regenerate the numerical tables
using Python 3.10 or newer:

```bash
python verify.py --output output
```

Install the plotting dependencies and regenerate the figures:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The default destination is `output`. Use `--output PATH` to choose
another directory and `--formats png` or `--formats pdf,svg` to choose formats.
`python reproduce.py --check-data` verifies data integrity and numerical
consistency. The plotting script uses Arial when available and DejaVu Sans
as a fallback. The reference plotting environment is Python 3.10 with the
package versions in `requirements.txt`.

## Data dictionary

| File in `data/` | Content and units |
| --- | --- |
| `validation.csv` | Table 1: total corrected energies in Ha and reported SE in microHa, at manuscript precision; active spaces and reference counts. |
| `figure3.csv` | Complete GPU/Arrow PT2 times in seconds and returned SE in microHa. |
| `trimci.csv` | TrimCI PT2 times in seconds for Table S2. |
| `sampling.csv` | Table S1: fine thresholds in Ha, draws per sampled iteration, and iteration counts. Common settings are in `parameters.json`. |
| `figure4.csv` | Strong scaling and Figure S2: GPUs, nodes, five repeats, median/min/max seconds and peak allocated GiB per GPU. |
| `figure5.csv` | SCI reference energies, external PT2 corrections and reported PT2 SE, all in Ha. `Nvar` is the determinant count; `feo` denotes the iron–oxo complex. |
| `figure5_extrapolation.csv` | All-point, unweighted fits of `E_SCI+PT2 = intercept + slope*abs(E2)` in Ha. Gap is quintet minus triplet in mHa. Fits are checked against the points. |
| `figure5_literature.csv` | Published spin gaps in mHa, with the corresponding model definitions and source links. |
| `figure5_geometries.csv` | Atom indices, elements and coordinates in angstrom, used for molecular insets. Source identifiers are in `plot_settings.json`. |
| `fep_cost_memory.csv` | Figure 6: complete PT2 times in seconds, allocated tensor memory and stage peaks in GiB, target/returned SE in microHa, and sketch bucket counts for both Fe-porphyrin states at 1–100 million determinants. |
| `figure7.csv` | Figure 7: three NNQS trajectories, 13 paired checkpoints each: parameter updates, retained-support reference energy, fully corrected energy and literature benchmark, in Ha. |
| `nnqs_benchmarks.csv` | Table S4: final unique support counts, reference energies in Ha and literature uncertainties in microHa. |
| `hcn_components.csv` | Table S5: HCN reference/external/raw-internal/Olsen cumulative energies in Ha and errors in mHa, on the same support and coefficients. |
| `sketch_accuracy.csv` | Table S3: reported sketch SE and absolute energy difference from explicit accumulation, in microHa, at manuscript precision. |
| `variance_complete_runs.csv` | Figure S1a: 64 complete PT2 results, control and sampled-correction energies in Ha, reported variances in Ha squared, counts and source-report identities. |
| `variance_conditional_hashes.csv` | Figure S1b: 512 paired sampled-correction estimates in Ha, crossing eight source/batch draws with sixteen shared hash maps at each of four bucket counts. |
| `variance_components.csv` | Figure S1b: crossed source/batch, hash and interaction variance estimates in Ha squared. Signed moment estimates are retained. |

`parameters.json` records scientific and timing settings; `plot_settings.json`
records presentation settings. `SHA256SUMS` covers all 17 CSVs and both JSONs.
Reported SEs describe statistical uncertainty. Empty CSV cells indicate missing
values. NNQS reference energies are evaluated on the retained support; the
training trajectories use 30,000 fixed parameter updates. Energy differences
are expressed as quintet minus triplet, with 1 mHa = 0.001 Ha.

### Variance analysis

`verify.py` recomputes sample SDs with the n-1 divisor, root-mean-square
reported uncertainties, and the crossed source/hash variance components from
the numerical observations. The complete-run table satisfies
`E2_Ha = control_Ha + sampled_mean_Ha` and
`reported_SE_Ha^2 = control_variance_Ha2 + tail_variance_Ha2`.
The control variance column is the reported conservative guard; the sampled
variance refers to the mean of six iterations.

For an 8-by-16 matrix of correction estimates at each bucket count, the
source/batch estimate is `(MS_draw - MS_interaction)/16`. Conditional collision
variance is the mean of the eight within-row sample variances. The logarithmic
slope is fitted to all four collision-variance points. `parameters.json`
records the calculation settings. `python verify.py --output output` also
writes the complete derived variance summary; `reproduce.py` generates
Figures S1 and S2 together with the main numerical figures.

### Figure 6 columns

`case` identifies `fep_triplet` or `fep_quintet`; `Nvar` is the reference
determinant count. `PT_seconds` is the complete PT2 time, and
`target_SE_microHa` / `reported_SE_microHa` are the requested and returned
statistical SEs. `control_buckets` and `tail_buckets` are the sketch bucket
counts per denominator-sign sector.

`resident_GiB` is the measured tensor allocation before the PT2 stages.
`reference_GiB`, `index_GiB`, `sampling_GiB` and `integral_GiB` are array-size
contributions from determinant keys and coefficients, the membership index,
source sampling and Hamiltonian tables. `other_resident_GiB` is the remaining
allocation. These components sum to the measured total.
`control_peak_GiB` and `stochastic_peak_GiB` are the measured total peak tensor
allocations during the two stages, including the reference and tables.
Figure 6b,c show the triplet; the CSV contains both states. Memory is reported
in GiB (2^30 bytes) and excludes allocator reserve and CUDA-context allocations.

## Data availability and citation

The numerical source data, analysis parameters and figure-reproduction scripts
are available at [fulz024/2026-gpu-pt2-data-deposit](https://github.com/fulz024/2026-gpu-pt2-data-deposit).
Please cite the accompanying manuscript and this data deposit using the
metadata in [CITATION.cff](CITATION.cff).

Create a versioned ZIP archive with `python tools/make_release.py`. Each archive
contains the source data, analysis scripts and a SHA-256 file manifest.

Code and documentation use the [BSD 3-Clause License](LICENSE). Published
comparison values and molecular coordinates retain the source attributions
listed in the data; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
