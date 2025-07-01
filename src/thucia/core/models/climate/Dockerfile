FROM rocker/tidyverse:4.4.2

RUN apt-get update && apt-get install -y \
    libudunits2-dev \
    libnetcdf-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libproj-dev \
    proj-data \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN install2.r --error \
    doParallel \
    argparse \
    quantmod \
    foreach \
    logger \
    tidync \
    ncdf4 \
    VGAM

RUN R -e 'install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)'

RUN install2.r --error \
    sf \
    spdep

COPY . /app
WORKDIR /app
