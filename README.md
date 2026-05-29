# SISAP 2026 Challenge: Working example on Julia

This repository is a working example for the SISAP 2026 Indexing Challenge <https://sisap-challenges.github.io/>, working with Julia and GitHub Actions, as specified in the task descriptions. It is based on the previous year example.

## Quick start (Docker)

Build the image and run search on all three spot-check datasets:

```bash
docker build -t sisap26/julia-baseline .
bash run_search.sh
```

Results are written to `results/task-{1,2,3}-spot-check/results.h5`.

## Manual setup (without Docker)

Requires Julia v1.10.10 (also works with v1.11 and v1.12 series).
Download Julia from <https://julialang.org/downloads/>.

### Fetch the datasets

```bash
bash prepare-data.sh
```

### Instantiate the project

```bash
JULIA_PROJECT=. JULIA_NUM_THREADS=8 julia -e 'using Pkg; Pkg.instantiate()'
```

### Run

```bash
JULIA_PROJECT=. JULIA_NUM_THREADS=8 julia -L sisap2026.jl -e 'main_task1(); main_task2(); main_task3()'
```

### Evaluation

```bash
JULIA_PROJECT=. julia -L eval.jl -e 'eval_task1()'
JULIA_PROJECT=. julia -L eval.jl -e 'eval_task2()'
JULIA_PROJECT=. julia -L eval.jl -e 'eval_task3()'
```

Two result files will be created: `result-task1.csv`, `result-task2.csv`, `result-task3.csv`.

## TIRA submission

The container entrypoint is `search.jl`.  The TIRA command is:

```
julia /app/search.jl --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir
```

To submit a code submission via `tira-cli`:

```bash
tira-cli code-submission \
  --path . \
  --command 'julia /app/search.jl --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
  --task sisap-2026 \
  --dataset task-1-spot-check-20260528-training
```

## How to create your own system

Fork this repository and adjust the search algorithms in `task1.jl`, `task2.jl`, and `task3.jl`.  The CI workflow (`.github/workflows/ci.yml`) builds the Docker image and tests it on all three spot-check datasets automatically on every push.

See also: <https://github.com/sisap-challenges/sisap2026-julia-example/actions>
