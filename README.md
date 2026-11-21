# HARP-CI Replication Package

This repository contains the replication materials for the HARP-CI study. It includes the processed case-level data, validation results, analysis notebooks, and the scripts used to retrieve CI logs and perform hardware-backed validation.

## Repository contents

```text
data/
  ci_recover_78_with_validation.csv   Case-level recovery results with validation columns
  validation_per_pick.csv             One row per candidate pick and validation verdict
  validation_per_test.csv             One row per test, with three repeated runs per side

pipeline_code/
  fetch_ci_data.ipynb                 Collect PyTorch GitHub Actions/ROCm metadata
  pipeline.ipynb                      Main recovery and agent-analysis pipeline
  results_analysis_from_validation.ipynb
                                      Post-validation statistical and consistency analysis

validation_code/
  fetch_ci_log.py                     Download and parse CI logs
  build_g90a.py                       Build gfx90a PyTorch wheels
  run_g950.py                         Build and test on gfx950/MI350X
  run_g950_test.py                    Corrected MI350X test runner
  run_g950_rerun.py                   Corrected rerun utility for selected MI350X units
  run_g90a_dev.py                     gfx90a/MI210 validation runner via SSH and Kubernetes
  oci_extract.py                      Extract runtime files from a public OCI image
  build_package.py                    Assemble the validation package from raw results
```

The `data/` directory contains the packaged outputs used for analysis. It does not contain the full raw CI-log archive, PyTorch source checkout, built wheels, Docker images, or cluster results.

## Quick start: reproduce the included analysis

1. Create a Python 3.10+ environment.
2. Install the notebook dependencies listed in [`requirements-notebooks.txt`](requirements-notebooks.txt).
3. Start Jupyter from the repository root:

   ```bash
   jupyter lab
   ```

4. Open `pipeline_code/results_analysis_from_validation.ipynb`.
5. Set the notebook working directory to the repository root, or update `BASE` in the first code cell if your notebook uses another working directory.
6. Run the notebook. It reads the three CSV files in `data/` and performs the validation-summary and GitHub comparison analyses.

The two earlier pipeline notebooks are research-workflow notebooks rather than turnkey scripts. They expect additional intermediate CSV/JSONL files and API credentials; see the configuration cells at the top of each notebook before running them.

## CI-log retrieval

`validation_code/fetch_ci_log.py` uses the PyTorch public S3 log mirror first and can fall back to the GitHub Actions API.

Examples:

```bash
python validation_code/fetch_ci_log.py 102313187406 85487999320 \
  --out ci_logs --parse

python validation_code/fetch_ci_log.py \
  --csv data/ci_recover_78_with_validation.csv \
  --col job_id --out ci_logs --parse
```

For the GitHub fallback, export either `GH_TOKEN` or `GITHUB_TOKEN`. The script can still retrieve logs from the public S3 mirror without a token when the mirror has them.

## Hardware validation

The validation runners are not intended to run on an ordinary laptop. They require, depending on the runner:

- AMD ROCm hardware and a compatible Linux host (MI210/gfx90a or MI350X/gfx950; the package also records gfx942 results).
- Docker with GPU access, including `/dev/kfd` and `/dev/dri`.
- A PyTorch source checkout and the commit objects referenced by the validation units.
- Access to the corresponding public PyTorch CI Docker images in GHCR.
- For `run_g90a_dev.py`, SSH access to the configured build host and a working `kubectl` context.
- The unit JSON files and intermediate build/result directories referenced by the scripts. These are not included in this compact package.

Before running these scripts, replace environment-specific placeholders such as `<build-host>`, `<workdir>`, and `<outdir>` and review the path constants near the top of each file. `run_g950.py` uses `~/apr` by default and creates/uses `src/pytorch`, `wheels`, `results`, and related directories there.

The validation convention is three repeated runs for the parent commit and three for the candidate commit. A test is classified as:

- `F2P`: failed at the parent and passed at the candidate;
- `F2F`: failed at both;
- `P2P`: passed at both;
- `FLAKY`: mixed outcomes across repeats;
- `OTHER`: missing, skipped, or otherwise not classifiable.

## Python dependencies

The standard-library utilities (`fetch_ci_log.py`, `oci_extract.py`, and most of `build_package.py`) do not require third-party Python packages. The notebooks require the packages in [`requirements-notebooks.txt`](requirements-notebooks.txt). The GPU runners use tools installed in the PyTorch CI images and therefore should not be treated as a normal local `pip` workflow.

The notebooks also reference project- or environment-specific modules, including `openai_codex` and `tools.text_chunker`. Those modules are not included in this repository and may require the original research environment or local replacement code. The notebook cells contain blank credential placeholders; supply credentials through environment variables or local configuration and do not commit secrets.

## Reproducibility and provenance

The included CSV files are snapshots of the analysis outputs. Results that depend on GitHub APIs, CI-log retention, model/API responses, PyTorch commit history, Docker image availability, or GPU hardware may change over time. Keep API tokens private, record the commit/image identifiers used for a rerun, and preserve the generated logs and summaries alongside any new results.

## License and citation

No license or citation metadata was present in this package at the time this README was written. Add the paper's preferred citation and any upstream PyTorch/third-party license notices before public redistribution.
