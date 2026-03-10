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
