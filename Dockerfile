# Define base image/operating system
FROM julia:1.10.10

WORKDIR /app

ENV JULIA_DEPOT_PATH=/usr/local/julia-depot

# Copy files and directory structure to working directory
COPY . .

RUN JULIA_PROJECT=. julia -t8 -Cnative -O3 -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    && chmod -R a+rX /usr/local/julia-depot
RUN JULIA_PROJECT=. julia -t8 -Cnative -O3 sisap2026.jl

ENTRYPOINT ["julia", "--project=/app", "/app/search.jl"]
