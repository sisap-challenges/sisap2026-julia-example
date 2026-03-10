using SimilaritySearch, InvertedFiles, SparseArrays, HDF5, LinearAlgebra, StatsBase
using SimilaritySearch.Special.Sparse

function load_sparse_matrix(filename, group_path)
    h5open(filename, "r") do file
        g = file[group_path]

        # 1. Extract components from the HDF5 group
        # Typically stored as 'data', 'indices', and 'indptr'
        nzval   = Float32.(read(g["data"]))
        colval  = Int32.(read(g["indices"]))
        rowptr  = Int32.(read(g["indptr"]))

        m, n = read(attributes(g)["shape"])
        colval .+= 1 # 1-based indexing
        rowptr .+= 1
        # for i in eachindex(nzval)
        #     nzval[i] = nzval[i] > 0.5f0 ? nzval[i] : 0f0
        # end
        X = SparseMatrixCSC(n, m, rowptr, colval, nzval)
        @show filename size(X) quantile(nzval, 0:0.1:1)
        db = Sparse.SparseDatabase(X)

        db
    end
end

function main_task3(filename="data/fiqa-dev.h5"; k::Int=30,logbase=1.1f0, minrecall=0.99f0)
    mkpath("data")

    db = load_sparse_matrix(filename, "train")

    S = SearchGraph(; dist=Sparse.NormCosine(), db)
    
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(; logbase),
        hyperparameters_callback=OptimizeParameters(MinRecall(minrecall)),
        #hints_callback=RandomHints(; logbase=1.05f0)
    )
    buildtime = @elapsed index!(S, ctx)

    meta = Dict()
    algo = "ABS"
    meta["algo"] = algo
    meta["buildtime"] = buildtime
    meta["params"] = pretty_params(string("b=$logbase r=$minrecall"))
    meta["size"] = length(db)
    meta["optimtime"] = 0.0

    queries = load_sparse_matrix(filename, "otest/queries")
    knns = zeros(IdDist, k, length(queries))
    #S.algo[] = BeamSearch(; Δ=1.4, bsize=8, maxvisits=10^4)
    #empty!(S.hints)
    #for i in 1:length(S)
    #    rand() < 0.5 && push!(S.hints, i)
    #end
    B = S.algo[]
    for step in -7:7
        Δ = B.Δ * 1.05^step
        S.algo[] = BeamSearch(; Δ, B.bsize, B.maxvisits)
        querytime = @elapsed knns = searchbatch(S, ctx, queries, k)
        meta["querytime"] = querytime
        meta["searchparams"] = string(S.algo[])
        meta["totaltime"] = buildtime + querytime 
        @info meta
        outdir = "results/task3-$algo"
        resfile = joinpath(outdir, algo * " " * meta["params"] * " " * meta["searchparams"] * ".h5")
        save_results(knns, meta, resfile)
    end
end
