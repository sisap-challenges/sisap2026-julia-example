using LinearAlgebra, HDF5, JLD2, Glob, Random, Printf, Dates, StatsBase, Statistics, MultivariateStats, Accessors
using SimilaritySearch, SimilaritySearch.Dist, SimilaritySearch.ScalarQuant

function save_results(knns_::Matrix, meta, resfile::AbstractString)
    knns = convert(Matrix{Int32}, knns_)
    dists = convert(Matrix{Float32}, knns_)

    mkpath(dirname(resfile))
    h5open(resfile, "w") do f
        f["knns"] = knns
        f["dists"] = dists
        A = attributes(f)
        A["algo"] = meta["algo"]

        A["buildtime"] = meta["buildtime"]
        A["optimtime"] = get(meta, "optimtime", 0.0)
        A["querytime"] = meta["querytime"]
        A["params"] = meta["params"]
        A["searchparams"] = meta["searchparams"]
        A["size"] = meta["size"]
    end
end

function pretty_params(params)
    params = replace(params, r"\s+"ism => " ")
    params = replace(params, "Float32" => "")
    params = replace(params, "minrecall" => "")
    params = replace(params, ":" => "")
    params = replace(params, "f0" => "")
    params = replace(params, r"\s+"ism => " ")
    params = replace(params, " tol " => " ")
    replace(params, r"\s+$"ism => "")
end

include("task1.jl")
include("task2.jl")
include("task3.jl")