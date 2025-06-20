FROM rocker/tidyverse:4.4.2

# Install R packages
RUN install2.r --error \
    argparse \
    logger \
    VGAM

RUN R -e 'install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)'

COPY . /app
WORKDIR /app
