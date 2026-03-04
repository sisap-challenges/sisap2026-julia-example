mkdir data

curl -L "https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/benchmark-dev-wikipedia-bge-m3-small.h5?download=true" -o data/benchmark-dev-wikipedia-bge-m3-small.h5
curl -L "https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/llama-dev.h5?download=true" -o data/llama-dev.h5

