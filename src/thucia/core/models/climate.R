library(argparse)
library(logger)
library(INLA)
library(VGAM)

# library(dlnm)
# library(dplyr)
# library(splines)
# library(tsModel)
# library(quantmod)
# library(data.table)

parser <- ArgumentParser()
parser$add_argument("--date", "-d", help = "Date")
parser$add_argument("--output", "-o", help = "Output filename")
xargs <- parser$parse_args()

target_date <- xargs$date
filename_i <- xargs$output

run_province_model_func <- function(data, formula = default_pois_formula) {
  setkeyv(data, c("TIME", "PROVINCE"))
  model <- inla(
    formula = formula,
    data = data, family = "zeroinflatedpoisson0", offset = log(POP_OFFSET),
    verbose = FALSE,
    control.inla = list(
      strategy = "adaptive",
      cmin = 0
    ),
    control.family = list(link = "log"),
    control.compute = list(
      waic = TRUE, dic = TRUE,
      cpo = TRUE, config = TRUE,
      return.marginals = TRUE
    ),
    control.fixed = list(
      correlation.matrix = TRUE,
      prec.intercept = 1, prec = 1
    ),
    control.predictor = list(link = 1, compute = TRUE)
  )
  model <- inla.rerun(model)
  return(model)
}

prior.prec <- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))
forecast_formula <- CASES ~ 1 +
  f(MONTH,
    replicate = PROV_IND, model = "rw1", cyclic = TRUE,
    constr = TRUE, scale.model = TRUE, hyper = prior.prec
  ) +
  f(YEAR, replicate = PROV_IND, model = "iid") +
  f(PROV_IND,
    model = "bym2",
    hyper = prior.prec,
    scale.model = TRUE,
    graph = file.path(
      province.inla.data.in.dir,
      "nbr_piura_tumbes_lambayeque.graph"
    )
  ) +
  SQ_RSI_DIR_LAG + SEASON +
  ns(MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG, df = 4) +
  tmin_roll_2_basis + prec_roll_2_basis + icen_basis + spi_basis

log_info("Run province model func")
tmp_climate_cv_fit <- run_province_model_func(
  data = rt_forecast_dt,
  formula = forecast_formula
)
idx.pred <- seq(i * num_provinces + 1, i * num_provinces + num_provinces)
xx <- inla.posterior.sample(samples, tmp_climate_cv_fit)
xx.s <- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx)
ypred <- matrix(NA, num_provinces, samples)
dirpred <- matrix(NA, num_provinces, samples)
for (s.idx in 1:samples) { # sample ID
  xx.sample <- xx.s[, s.idx]
  ypred[, s.idx] <- rzipois(
    num_provinces,
    lambda = exp(xx.sample[-1]),
    pstr0 = xx.sample[1])
  dirpred[, s.idx] <- ypred[, s.idx] / pop_offsets
}
