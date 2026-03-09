using SimilaritySearch, InvertedFiles, SparseArrays, HDF5, LinearAlgebra, StatsBase
using InvertedFiles: sparseiterator

function load_sparse_matrix(filename, group_path)
    h5open(filename, "r") do file
        g = file[group_path]

        # 1. Extract components from the HDF5 group
        # Typically stored as 'data', 'indices', and 'indptr'
        nzval   = Float16.(read(g["data"]))
        colval  = Int32.(read(g["indices"]))
        rowptr  = Int32.(read(g["indptr"]))

        m, n = read(attributes(g)["shape"])
        colval .+= 1 # 1-based indexing
        rowptr .+= 1

        # create a CSC Matrix via transposing data
        return SparseMatrixCSC(n, m, rowptr, colval, nzval)
    end
end

function InvertedFiles.sparseiterator(X::SparseMatrixCSC, i)
    r = nzrange(X, i)
    v = nonzeros(X)
    @info r, view(v, r)
    exit(0)
    zip(r, view(v, r))
end

function InvertedFiles.sparseiterator(vec::SubArray{<:AbstractFloat, 1, <:SparseMatrixCSC})  # to efficiently support views
    _, i = vec.indices
    InvertedFiles.sparseiterator(vec.parent, i)
end


function main_task3(filename="data/fiqa-dev.h5"; k::Int=30, t::Int=2)
    mkpath("data")

    X = load_sparse_matrix(filename, "train")
    @show size(X)

    invfile = WeightedInvertedFile(size(X, 1))
    ctx = InvertedFileContext()
    buildtime = @elapsed append_items!(invfile, ctx, MatrixDatabase(X))
    @info "Posting lists quantiles"
    let quant = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
        qval = quantile(neighbors_length.(Ref(invfile.adj), 1:length(invfile)), quant)
        for (a, b) in zip(quant, qval)
            @info a => b
        end
    end
    
    meta = Dict()
    algo = "InvertedFile"
    meta["algo"] = algo
    meta["buildtime"] = buildtime
    meta["params"] = "plain data"
    meta["size"] = size(X, 2)
    meta["optimtime"] = 0.0

    queries = MatrixDatabase(load_sparse_matrix(filename, "otest/queries"))
    knns = zeros(IdDist, k, length(queries))

    ##querytime = @elapsed Threads.@threads :static for i in 1:length(queries)
    ##    res = knnqueue(KnnSorted, view(knns, :, i))
    ##    InvertedFiles.search_invfile(invfile, ctx, queries[i], res, t) do p
    ##        N = neighbors(p.idx.adj, p.tokenID)
    ##        N !== nothing
    ##        #if N === nothing || length(N) > 10000 #|| p.weight < 1.0
    ##        #    false
    ##        #else
    ##        #    true
    ##        #end
    ##    end
    ##end
    querytime = @elapsed knns = searchbatch(invfile, ctx, queries, k)

    meta["querytime"] = querytime
    meta["searchparams"] = ""
    meta["totaltime"] = buildtime + querytime 
    @info meta
    outdir = "results/task3-$algo"
    resfile = joinpath(outdir, algo * " " * meta["params"] * ".h5")
    save_results(knns, meta, resfile)
end
