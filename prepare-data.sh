
if [ ! -d "data" ]
then
    mkdir data
fi

#for f in benchmark-dev-wikipedia-bge-m3.h5 llama-dev.h5 nq.h5  ## full datasets
for f in benchmark-dev-wikipedia-bge-m3-small.h5 llama-dev.h5 fiqa-dev.h5
do
    out="data/${f}"
    
    if [ ! -f $out ]
    then
        curl -L "https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/${f}?download=true" -o $out
    fi
done
