#!/usr/bin/env julia
# search.jl – entrypoint for the SISAP 2026 Julia baseline
#
# Usage (mirrors the reference Python baseline):
#   julia --project search.jl \
#       --input          /path/to/dataset.h5 \
#       --task-description /path/to/config.json \
#       --output         /path/to/output/dir/
#
# --input            path to the HDF5 data file
# --task-description path to the config.json describing the task
# --output           output directory; results are written to <output>/results.h5
#
# config.json must follow the flat format used by the Python baseline:
#   { "task": "task1", "filename": "foo.h5", "k": 15, "dataset_name": "...", ... }

using JSON

include("sisap2026.jl")

# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #

function parse_args(args)
    input_file   = nothing
    task_desc    = nothing
    output_path  = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--input" && i + 1 <= length(args)
            input_file = args[i + 1];  i += 2
        elseif args[i] == "--task-description" && i + 1 <= length(args)
            task_desc = args[i + 1];   i += 2
        elseif args[i] == "--output" && i + 1 <= length(args)
            output_path = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    input_file, task_desc, output_path
end

input_file, task_desc, output_path = parse_args(ARGS)

if input_file === nothing || task_desc === nothing || output_path === nothing
    println(stderr, "Usage: julia search.jl --input <dataset.h5> --task-description <config.json> --output <output-dir>")
    println(stderr, "")
    println(stderr, "  --input            path to the HDF5 data file")
    println(stderr, "  --task-description path to the config.json describing the task")
    println(stderr, "  --output           output directory (results.h5 is written inside it)")
    exit(1)
end

# --------------------------------------------------------------------------- #
# Load config.json and detect task                                             #
# --------------------------------------------------------------------------- #

cfg = JSON.parsefile(task_desc)

if !haskey(cfg, "task")
    println(stderr, "config.json must contain a \"task\" key (e.g. \"task\": \"task1\")")
    exit(1)
end

task = cfg["task"]
k    = Int(cfg["k"])

output_file = joinpath(output_path, "results.h5")

@info "Detected task=$task  dataset=$(cfg["dataset_name"])  k=$k"
@info "Input file: $input_file"
@info "Output file: $output_file"

# Ensure the output directory exists
mkpath(abspath(output_path))

# --------------------------------------------------------------------------- #
# Dispatch – each task function writes a single file to output_file           #
# --------------------------------------------------------------------------- #

if task == "task1"
    main_task1(; file=input_file, k, output_file, dataset=cfg["dataset_name"], task)
elseif task == "task2"
    main_task2(; file=input_file, k, output_file, dataset=cfg["dataset_name"], task)
elseif task == "task3"
    main_task3(input_file; k, output_file, dataset=cfg["dataset_name"], task)
else
    println(stderr, "Unknown task: $task  (expected task1, task2, or task3)")
    exit(1)
end
