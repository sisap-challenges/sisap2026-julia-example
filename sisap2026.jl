using LinearAlgebra, HDF5, JLD2, Glob, JSON, ProgressMeter, Random, Printf, Dates, StatsBase, Statistics, MultivariateStats, Accessors
using SimilaritySearch, SimilaritySearch.Dist, SimilaritySearch.ScalarQuant, SimilaritySearch.Projections

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

function run_task1(dist, db;
    name,
    optim=MinRecall(0.9),
    benchmark,
    optimsearch=MinRecall(0.90),
    neighborhood=Neighborhood(; filter=SatNeighborhood(), neardup=0.05f0, logbase=1.3f0),
)
    params = let tname = Base.typename(typeof(db)).name
        pretty_params(string("$tname ", optim, "; ", neighborhood))
    end

    ctx = SearchGraphContext(;
        hyperparameters_callback=OptimizeParameters(optim),
        hints_callback=RandomHints(logbase=1.1),
        logger=InformativeLog(prompt=name * "> "),
        neighborhood,
    )

    @info "indexing $params"
    G = SearchGraph(; db, dist)
    buildtime = @elapsed G = index!(G, ctx)

    meta = Dict()
    meta["algo"] = "ABS"
    meta["buildtime"] = buildtime
    #meta["params"] = pretty_params(string("PCA $(benchmark.maxoutdim)+$(benchmark.idim) i8; ", optim, "; ", neighborhood))
    meta["params"] = params
    meta["size"] = length(db)
    @info meta
    meta["optimtime"] = @elapsed bestlist = optimize_index!(G, ctx, optimsearch, ksearch=benchmark.k + 1)
    meta["querytime"] = @elapsed knns = allknn(G, ctx, benchmark.k)
    meta["searchparams"] = "task1 algorithm"

    knns, meta
end

function main_task1(;
    file="data/benchmark-dev-wikipedia-bge-m3-small.h5",
    k::Int=15 + 1,
)
    dist, X, name = jldopen(file) do f
        @time "Loading $file/train" X = f["train"]
        Dist.CastF32.NormCosine(), StrideMatrixDatabase(X), "f16-cos"
        #ScalarQuant.SQu8SqL2(), ScalarQuant.SQu8(X), "SQu8SqL2"
        #ScalarQuant.SQu8SqL2(), ScalarQuant.SQu2(X), "SQu2SqL2"
        #=let pca = fit(PCA, Float32.(X[:, 1:300_000]); maxoutdim=256)
            X = predict(pca, X)
            ScalarQuant.SQu8SqL2(), ScalarQuant.SQu8(X), "PCA256-SQu8-SqL2"
        end=#
        #ScalarQuant.SQu8NormCosine(), ScalarQuant.SQu8(X), "SQu8Cos"
        #=let rp = Projections.qr(1024, 256)
            X = Projections.transform(rp, X)
            ScalarQuant.SQu8SqL2(), ScalarQuant.SQu8(X), "RP256-SQu8-SqL2"
        end=#
    end

    outdir = "results/task1-$name"
    resfile_ = replace(basename(file), ".h5" => "")
    #indexfile = joinpath(outdir, "index-$(name).jl2")
    benchmark = (; file, k)
    totaltime = @elapsed knns, meta = run_task1(dist, X; benchmark, name="$name $file")
    meta["totaltime"] = totaltime
    @info meta
    resfile = joinpath(outdir, resfile_, "ABS " * meta["params"] * ".h5")
    save_results(knns, meta, resfile)
end

function run_task2(dist, db, queries;
    name,
    optim=MinRecall(0.99),
    benchmark,
    outdir,
    optimsearch=MinRecall(0.90),
    neighborhood=Neighborhood(; filter=SatNeighborhood(), neardup=0.05f0, logbase=1.1f0),
)
    params = let tname = Base.typename(typeof(db)).name
        pretty_params(string("$tname ", optim, "; ", neighborhood))
    end

    ctx = SearchGraphContext(;
        hyperparameters_callback=OptimizeParameters(optim),
        hints_callback=RandomHints(logbase=1.1),
        logger=InformativeLog(prompt=name * "> "),
        neighborhood,
    )

    @info "indexing $params"
    G = SearchGraph(; db, dist)
    buildtime = @elapsed G = index!(G, ctx)

    meta = Dict()
    meta["algo"] = "ABS"
    meta["buildtime"] = buildtime
    meta["params"] = params
    meta["size"] = length(db)
    meta["optimtime"] = optimtime = @elapsed bestlist = optimize_index!(G, ctx, optimsearch, ksearch=benchmark.k + 1)

    beam = G.algo[]
    step = 1.05f0
    for i in -4:5
        Δ = beam.Δ * step^i
        G.algo[] = BeamSearch(; beam.bsize, Δ, beam.maxvisits)
        querytime = @elapsed knns = searchbatch(G, ctx, queries, benchmark.k)
        meta["querytime"] = querytime
        meta["searchparams"] = "$optimsearch $(G.algo[])"
        meta["totaltime"] = buildtime + querytime + optimtime
        @info meta
        resfile = joinpath(outdir, "ABS " * meta["params"] * ".h5")
        save_results(knns, meta, resfile)
    end
end

function run_task2_exhaustive(dist, db, queries;
    name,
    benchmark,
    outdir
)
    S = ExhaustiveSearch(; dist, db)
    buildtime = 0.0
    params = name
    meta = Dict()
    meta["algo"] = "ExhaustiveSearch"
    meta["buildtime"] = buildtime
    meta["params"] = params
    meta["size"] = length(db)
    meta["optimtime"] = optimtime = 0.0

    ctx = GenericContext()
    querytime = @elapsed knns = searchbatch(S, ctx, queries, benchmark.k)
    meta["querytime"] = querytime
    meta["searchparams"] = ""
    meta["totaltime"] = buildtime + querytime + optimtime
    @info meta
    resfile = joinpath(outdir, meta["algo"] * " " * meta["params"] * ".h5")
    save_results(knns, meta, resfile)
end

function main_task2(;
    file="data/llama-dev.h5",
    k::Int=30,
)
    dist, db, queries, name = jldopen(file) do f
        @time "Loading $file/train" X = f["train"]
        @time "Loading $file/queries" Q = f["test/queries"]
        @show size(X) size(Q) typeof(X)
        #=let pca = fit(PCA, X)
            X = predict(pca, X)
            Q = predict(pca, Q)
            Dist.SqL2(), StrideMatrixDatabase(X), StrideMatrixDatabase(Q), "EX PCA f32 SqL2"
        end=#
        #=let rp = Projections.qr(128, 96)
            Dist.NormCosine(), StrideMatrixDatabase(Projections.transform(rp, X)), StrideMatrixDatabase(Projections.transform(rp, Q)), "EX RP f32-96 dot"
        end=#
        #Dist.CastF32.NormCosine(), StrideMatrixDatabase(sq_global_u8(X, minmax=[-16.0, 19.0])), StrideMatrixDatabase(sq_global_u8(Q, minmax=[-16.0, 19.0])), "EX SQGu8-dot"

        #Dist.NormCosine(), StrideMatrixDatabase(X), StrideMatrixDatabase(Q), "EX f32-dot"
        #Dist.CastF32.NormCosine(), StrideMatrixDatabase(Float16.(X)), StrideMatrixDatabase(Float16.(Q)), "EX f16-dot"
        #ScalarQuant.SQu8NormCosine(), ScalarQuant.SQu8(X), ScalarQuant.SQu8(Q), "EX u8-dot"
        #ScalarQuant.SQu4SqL2(), ScalarQuant.SQu4(X), ScalarQuant.SQu4(Q), "EX u4-dot"
        #Dist.CastF32.SqL2(), StrideMatrixDatabase(Float16.(X)), StrideMatrixDatabase(Float16.(Q)), "ABS f16-L2"
        Dist.CastF32.NormCosine(), StrideMatrixDatabase(Float16.(X)), StrideMatrixDatabase(Float16.(Q)), "f16-Cos"
    end

    outdir = "results/task2-$name"
    benchmark = (; file, k)
    run_task2_exhaustive(dist, db, queries; benchmark, name, outdir)
    #run_task2(dist, db, queries; benchmark, name="$name $file", outdir)
end
