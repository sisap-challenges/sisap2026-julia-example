using JLD2, SimilaritySearch, DataFrames, CSV, Glob, UnicodePlots, Printf, Statistics, StatsBase, Markdown

function read_gold(filename; group, k=30, starts=1)
    G = jldopen(filename) do f
        display(md"## structure of the h5 file")
        display(f)
        knns = f["$group/knns"][1:k, :]
        dists = f["$group/dists"][1:k, :]

        if starts == 0
            knns .+= 1
        end
        (; knns, dists)
    end

    G
end

function distquantiles(name, dists; qlist=0.0:0.1:1.0, klist=[1, 3, 10, 30])
    qlist = 0.0:0.1:1.0
    D = DataFrame(name=[], min=[], q10=[], q20=[], q30=[], q40=[], q50=[], q60=[], q70=[], q80=[], q90=[], max=[])
    for k in klist
        push!(D, ("$(k)nn", quantile(dists[k, :], qlist)...))
    end
    display(md"""## Quantiles for different nearest neighbors""")
    display(name)
    display(D)
end

function remove_self_loop!(knns, dists)
    for i in axes(knns, 2)
        knns_ = @view knns[:, i]
        p = findfirst(x -> x == i, knns_)

        if p !== nothing
            dists_ = @view dists[:, i]
            knns_[p:end-1] .= knns_[p+1:end]
            dists_[p:end-1] .= dists_[p+1:end]
        end
    end

end

function eval_task1(;
    goldfile="data/benchmark-dev-wikipedia-bge-m3-small.h5",
    reslist=glob("results/task1*/*/*.h5"),
    outfile="results-task1.csv",
    k=15
)

    group = "/allknn"
    G = read_gold(goldfile; k=k + 1, group)
    remove_self_loop!(G.knns, G.dists)
    distquantiles(md"gold standard: $goldfile", G.dists, klist=[1, 5, 10, 15])

    # Some statistics about the gold standard 

    display(md"## Result analysis of your _task1_ results")
    D = []

    for resfile in reslist
        jldopen(resfile) do f
            display(md"""
            ### result file structure
            file: $resfile

            $f
            """)
            #A = attrs(f)
            A = JLD2.load_attributes(f, "")
            knns, dists = f["knns"][1:k, :], f["dists"][1:k, :]
            remove_self_loop!(knns, dists)
            recall = macrorecall((@view G.knns[1:k, :]), (@view knns[1:k, :]))

            distquantiles(md"", dists, klist=[1, 5, 10, 15])
            buildtime = A["buildtime"]
            optimtime = A["optimtime"]
            querytime = A["querytime"]
            task = "allknn" # A["task"]
            dataset = goldfile #A["db"]
            totaltime = get(A, "totaltime", buildtime + optimtime + querytime)
            R = (; algo=A["algo"], recall, totaltime, buildtime, optimtime, querytime, size=A["size"], resfile, dataset, task)
            push!(D, R)
        end
    end

    D = DataFrame(D)

    display(D)
    CSV.write(outfile, D)
end

function eval_task2(;
    goldfile="data/llama-dev.h5",
    reslist=glob("results/task2*/*.h5"),
    outfile="results-task2.csv",
    k=30
)

    group = "/test"
    G = read_gold(goldfile; k, group, starts=0)
    distquantiles(md"gold standard: $goldfile", G.dists, klist=[1, 10, 20, 30])

    # Some statistics about the gold standard 

    display(md"## Result analysis ")
    D = []

    for resfile in reslist
        jldopen(resfile) do f
            display(md"""
            ### result file structure
            file: $resfile

            $f
            """)
            #A = attrs(f)
            A = JLD2.load_attributes(f, "")
            knns, dists = f["knns"][1:k, :], f["dists"][1:k, :]
            @show G.knns[1:k, 1:8]
            @show knns[1:k, 1:8]

            recall = macrorecall((@view G.knns[1:k, :]), (@view knns[1:k, :]))

            distquantiles(md"", dists, klist=[1, 10, 20, 30])
            buildtime = A["buildtime"]
            optimtime = A["optimtime"]
            querytime = A["querytime"]
            task = "task2" # A["task"]
            dataset = goldfile #A["db"]
            totaltime = get(A, "totaltime", buildtime + optimtime + querytime)
            R = (; algo=A["algo"], recall, totaltime, buildtime, optimtime, querytime, size=A["size"], resfile, dataset, task)
            push!(D, R)
        end
    end

    D = DataFrame(D)

    display(D)
    CSV.write(outfile, D)
end

