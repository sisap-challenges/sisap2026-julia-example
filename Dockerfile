# Define base image/operating system
FROM julia:1.10.10

WORKDIR /sisap2026

# Copy files and directory structure to working directory
COPY . . 

RUN JULIA_PROJECT=. julia -t8 -Cnative -O3 -e 'using Pkg; Pkg.instantiate(); '
RUN JULIA_PROJECT=. julia -t8 -Cnative -O3 sisap2026.jl

#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
