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

    outdir = "results/task2"
    benchmark = (; file, k)
    run_task2_exhaustive(dist, db, queries; benchmark, name, outdir)
    #run_task2(dist, db, queries; benchmark, name="$name $file", outdir)
end
