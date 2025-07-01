library(data.table)
library(lubridate)
library(quantmod)
library(argparse)
library(splines)
library(tidync)
library(logger)
library(INLA)
library(VGAM)

library(sf)
library(spdep)

#
# This code performs forecasting onto target_date
# This requires all predictors in the model to be present and available
# If target_date is not specified, all dates are predicted (note that at present the
# model is trained on all available data so these are not strictly forecasts)
#
# Rows can be included with Cases=NA, which can attempt to be predicted by the model
#


# Neighbours graph file
adjacency_graph_file <- file.path(
  "nbr_PER.graph"  # current working folder (i.e. temporary file)
)

parser <- ArgumentParser()
parser$add_argument("--target_date", "-t", help = "Target date")
parser$add_argument("--input", "-i", help = "Input case data filename")
parser$add_argument("--output", "-o", help = "Output filename")
parser$add_argument("--samples", "-s", type = "integer", default = 1000,
                    help = "Number of samples for posterior distribution")
parser$add_argument("--admin1", help = "GID-1 list, e.g. 'PER.14_1,PER.21_1,PER.25_1'")
xargs <- parser$parse_args()

target_date <- xargs$target_date
cases_filename <- xargs$input
output_filename <- xargs$output
samples <- xargs$samples
adm1 <- xargs$admin1

# target_date = '2021-01-01'
# cases_filename = '/data/cases_with_climate.nc'
# output_filename = '/data/climate_cases_samples.csv'
# samples = 1000
# adm1 = 'PER.14_1,PER.21_1,PER.25_1'

adm1 <- unlist(strsplit(adm1, ","))  # comma-separated list to list

df <- as.data.table(hyper_tibble(tidync(cases_filename)))
# "ADM1"      "ADM2"      "GID_1"     "GID_2"     "Cases"     "tmax"
# "tmin"      "prec"      "SPI6"      "TotalONI"  "AnomONI"   "pop_count"
# "DIR"       "Date"

setDT(df)

# Filter by adm1
if (!is.null(adm1)) {
  df <- df[GID_1 %in% adm1]
}
# df[is.na(Cases), Cases := 0]  # NA cases are predictions to be made

# Derive MONTH, YEAR, PROVINCE, and POP
df[, Date := as.Date(Date)]
df[, CASES := Cases]
df[, `:=`(
  MONTH = as.integer(format(Date, "%m")),
  YEAR = as.integer(format(Date, "%Y")),
  PROVINCE = GID_2,
  POP = pop_count
)]
df[, c("GID_1", "GID_2", "pop_count", "Date", "Cases") := NULL]  # drop cols

# Derive TIME, an index of month-year (1-140)
df[, Date := ceiling_date(as.Date(paste(YEAR, MONTH, 1, sep = "-")), "month") - days(1)]
# df[, Date := as.Date(paste(YEAR, MONTH, 1, sep = "-"))]
date_lookup <- unique(df[, .(Date)])[order(Date)][, TIME := .I]
df <- merge(df, date_lookup, by = "Date", all.x = TRUE)

# Add additional required columns
summer_months <- c(12, seq(1, 4))
df[, `:=`(
  PROV_IND = as.integer(factor(PROVINCE)),
  POP_OFFSET = POP / 1e5,
  DIR = CASES / POP * 1e5,
  SEASON = as.integer(MONTH %in% summer_months)
)]

df[, HISTORICAL_DIR := shift(DIR, 12, type = "lag"), by = "PROVINCE"]
df[, HISTORICAL_DIR := fifelse(is.na(HISTORICAL_DIR), 0, HISTORICAL_DIR)]
df[, DIFF_WITH_HISTORICAL_DIR := DIR - HISTORICAL_DIR,
  by = "PROVINCE"
]

# Ensure input is a data.table and ordered correctly
tmp <- copy(df)
setorder(tmp, PROVINCE, TIME)
# Compute 12-month forward DIR and fill in missing differences
tmp[, `:=`(
  FUTURE_DIR = shift(DIR, type = "lead", n = 12),
  DIFF_WITH_HISTORICAL_DIR = fifelse(
    is.na(DIFF_WITH_HISTORICAL_DIR),
    DIR - shift(DIR, type = "lead", n = 12),
    DIFF_WITH_HISTORICAL_DIR
  )
), by = PROVINCE]

# Compute 1-month lag of the difference
tmp[,
  MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG := shift(DIFF_WITH_HISTORICAL_DIR, 1),
  by = PROVINCE
]
tmp[, MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG := fifelse(
  is.na(MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG),
  0,
  MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG
)]
# Merge the lagged difference back to the original data
tmp_dedup <- unique(tmp[, .(PROVINCE, TIME, DIFF_WITH_HISTORICAL_DIR, MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG)])
dupes <- tmp_dedup[, .N, by = .(PROVINCE, TIME)][N > 1]
if (nrow(dupes) > 0) {
  print("❗ Duplicates in tmp_dedup:")
  print(dupes)
  tmp_dedup <- tmp_dedup[, .SD[1], by = .(PROVINCE, TIME)]  # keep first row only
}
df <- merge(df, tmp_dedup, by = c("PROVINCE", "TIME"))

# SQ_RSI_DIR_LAG
safe_rsi <- function(x, n = 3) {
  if (length(x) < n || sum(!is.na(x)) < n) {
    return(rep(NA_real_, length(x)))
  }

  result <- tryCatch(
    RSI(x, n = n),
    error = function(e) {
      warning("❌ RSI failed: ", conditionMessage(e))
      rep(NA_real_, length(x))
    }
  )

  result
}
df[, RSI_DIR := safe_rsi(DIR, n = 3), by = PROVINCE]
df[, RSI_DIR := fifelse(is.na(RSI_DIR), 0, RSI_DIR)]  # Replace NA RSI with 0
# df[, RSI_DIR := RSI(DIR, n = 3), by = PROVINCE]
df[, RSI_DIR_LAG := shift(RSI_DIR, 1), by = PROVINCE]
df[, RSI_DIR_LAG := fifelse(is.na(RSI_DIR_LAG), 0, RSI_DIR_LAG)]  # Replace NA RSI with 0
df[, SQ_RSI_DIR_LAG := RSI_DIR_LAG^2]

# --- Helper function ---
add_ns_basis <- function(dt, varname, df_val = 2, prefix = NULL) {
  stopifnot(is.data.table(dt))
  x <- dt[[varname]]
  basis <- as.data.table(ns(x, df = df_val))
  if (is.null(prefix)) prefix <- varname
  setnames(basis, paste0(prefix, ".", seq_len(ncol(basis))))
  return(cbind(dt, basis))
}

# 1–2 month lags of TotalONI as proxy for E_INDEX
df[, TotalONI_lag1 := shift(TotalONI, 1, type = "lag"), by = PROVINCE]
df[, TotalONI_lag2 := shift(TotalONI, 2, type = "lag"), by = PROVINCE]

# Apply splines to lags
df <- add_ns_basis(df, "TotalONI_lag1", df_val = 2, prefix = "icen_basis_lag1")
df <- add_ns_basis(df, "TotalONI_lag2", df_val = 2, prefix = "icen_basis_lag2")

# --- Apply to each climatic variable ---
df[, tmin_roll_2 := zoo::rollmean(tmin, k = 2, fill = NA, align = "right"), by = PROVINCE]
df[, prec_roll_2 := zoo::rollmean(prec, k = 2, fill = NA, align = "right"), by = PROVINCE]

# Add lags 1–4 and basis functions for each
for (k in 1:4) {
  df[[paste0("tmin_roll_2_lag", k)]] <- shift(df$tmin_roll_2, k, type = "lag")
  df <- add_ns_basis(df, paste0("tmin_roll_2_lag", k), df_val = 2, prefix = paste0("tmin_roll_2_basis_lag", k))

  df[[paste0("prec_roll_2_lag", k)]] <- shift(df$prec_roll_2, k, type = "lag")
  df <- add_ns_basis(df, paste0("prec_roll_2_lag", k), df_val = 2, prefix = paste0("prec_roll_2_basis_lag", k))
}

for (k in 1:2) {
  df[[paste0("SPI6_lag", k)]] <- shift(df$SPI6, k, type = "lag")
  df <- add_ns_basis(df, paste0("SPI6_lag", k), df_val = 2, prefix = paste0("spi_basis_lag", k))
}

for (k in 1:4) {
  df[[paste0("TotalONI_lag", k)]] <- shift(df$TotalONI, k, type = "lag")
  df <- add_ns_basis(df, paste0("TotalONI_lag", k), df_val = 2, prefix = paste0("icen_basis_lag", k))
}

# Copy df to cases_dt
cases_dt <- copy(df)

# Drop rows with NA predictors, only if CASES is known
predictor_cols <- setdiff(names(cases_dt), "CASES")
# cases_dt <- cases_dt[complete.cases(.SD), .SDcols = predictor_cols]
# if (anyNA(cases_dt[!is.na(CASES)])) {
#   stop("❌ Missing values in predictors — INLA will fail.")
# }
# cases_dt <- cases_dt[complete.cases(cases_dt)]

# Create neighbours object
log_info("Creating adjacency graph: {adjacency_graph_file}")
provinces_sf <- st_read("/data/gadm41_PER.gpkg", layer="ADM_ADM_2")
provinces_sf <- subset(provinces_sf, provinces_sf$GID_1 %in% adm1)
nb <- poly2nb(st_make_valid(provinces_sf), queen = TRUE)
nb2INLA(file = adjacency_graph_file, nb)
province_names_ordered <- provinces_sf$GID_2

# Ensure PROVINCE is a factor with levels in the correct order
cases_dt[, PROV_IND := match(PROVINCE, province_names_ordered)]
stopifnot(!any(is.na(cases_dt$PROV_IND)))
stopifnot(all(province_names_ordered[cases_dt$PROV_IND] == cases_dt$PROVINCE))

# Build a complete date-province grid
date_seq <- sort(unique(df$Date))
prov_seq <- sort(unique(df$PROVINCE))
full_grid <- CJ(PROVINCE = prov_seq, Date = date_seq)

# See if anything is missing
missing_combos <- full_grid[!df, on = .(PROVINCE, Date)]
if (nrow(missing_combos) > 0) {
  print("⚠️ Missing PROVINCE x DATE combinations:")
  print(missing_combos)
}

# ---- Determine prediction index ----
num_provinces <- length(unique(cases_dt$PROVINCE))
if (is.null(target_date)) {
  # No date provided: forecast the next unseen month

  # Get the first date with Cases=NA
  i <- min(cases_dt[is.na(CASES), TIME]) - 1

  # i <- max(date_lookup$TIME) + 1
  forecast_date <- date_lookup[TIME == i + 1, Date]
} else {
  target_date <- as.Date(target_date)

  # If target_date exists, get its index
  if (target_date %in% date_lookup$Date) {
    i <- date_lookup[Date == target_date, TIME]
  } else {
    # Extend date_lookup to include target_date
    all_dates <- sort(unique(c(date_lookup$Date, target_date)))
    date_lookup <- data.table(Date = all_dates)[, TIME := .I]
    i <- date_lookup[Date == target_date, TIME]
  }

  forecast_date <- target_date
}
if (i %in% cases_dt$TIME) {
  pop_offsets <- c(cases_dt[TIME == i, POP_OFFSET])
} else {
  warning("⚠️ No population data for forecast TIME = ", i, "; using latest available instead.")
  pop_offsets <- c(cases_dt[TIME == max(cases_dt$TIME), POP_OFFSET])
}
log_info("Forecasting cases for month: {forecast_date}")

# Check and add rows for the forecast time if missing ------ NEEDS TO BE OVERHAULED ----
if (!(i %in% cases_dt$TIME)) {
  stop("❌ No forecast predictors for row for TIME = ", i, " (", forecast_date, ")")
}

# --------------------------------------------------------------------------------------

run_province_model_func <- function(data, formula) {
  setkeyv(data, c("TIME", "PROVINCE"))
  model <- inla(
    formula = formula,
    data = data,
    family = "zeroinflatedpoisson0",
    offset = log(data$POP_OFFSET),
    verbose = TRUE,
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
    graph = adjacency_graph_file
  ) +
  SQ_RSI_DIR_LAG + SEASON +
  ns(MODIFIED_DIFF_WITH_HISTORICAL_DIR_LAG, df = 4) +
  tmin_roll_2_basis_lag1.1 + tmin_roll_2_basis_lag1.2 +
  tmin_roll_2_basis_lag2.1 + tmin_roll_2_basis_lag2.2 +
  tmin_roll_2_basis_lag3.1 + tmin_roll_2_basis_lag3.2 +
  tmin_roll_2_basis_lag4.1 + tmin_roll_2_basis_lag4.2 +

  prec_roll_2_basis_lag1.1 + prec_roll_2_basis_lag1.2 +
  prec_roll_2_basis_lag2.1 + prec_roll_2_basis_lag2.2 +
  prec_roll_2_basis_lag3.1 + prec_roll_2_basis_lag3.2 +
  prec_roll_2_basis_lag4.1 + prec_roll_2_basis_lag4.2 +

  spi_basis_lag1.1 + spi_basis_lag1.2 +
  spi_basis_lag2.1 + spi_basis_lag2.2 +

  icen_basis_lag1.1 + icen_basis_lag1.2 +
  icen_basis_lag2.1 + icen_basis_lag2.2 +
  icen_basis_lag3.1 + icen_basis_lag3.2 +
  icen_basis_lag4.1 + icen_basis_lag4.2

log_info("Run province model func")
inla_province_model <- run_province_model_func(
  data = cases_dt,
  formula = forecast_formula
)

# ---- Sample from the posterior ----
log_info("Drawing samples...")

# Draw posterior samples
xx <- inla.posterior.sample(samples, inla_province_model)

# Get predictor row indices
latent_names <- rownames(xx[[1]]$latent)
predictor_rows <- grep("^Predictor:", latent_names)

df_samples <- NULL
for (i in date_lookup[, TIME]) {
    log_info("Processing TIME = {i} ({forecast_date})...")
    forecast_date <- date_lookup[TIME == i, Date]

    # Get predictor row indices
    start_idx <- (i - 1) * num_provinces + 1
    end_idx <- start_idx + num_provinces - 1
    idx.pred <- predictor_rows[start_idx:end_idx]

    samples <- length(xx)
    num_preds <- length(idx.pred)
    results <- matrix(NA_real_, nrow = num_preds + 1, ncol = samples)  # +1 for theta1

    extract_fun <- function(sample) {
      theta1 <- sample$hyperpar[1]
      preds <- sample$latent[idx.pred]
      c(theta1, preds)
    }

    for (s in seq_len(samples)) {
      results[, s] <- extract_fun(xx[[s]])
    }

    xx.s <- results

    # ---- Simulate case counts and incidence rates ----
    ypred <- matrix(NA, num_provinces, samples)
    for (s.idx in 1:samples) {
      theta1 <- xx.s[1, s.idx]
      lambda <- exp(xx.s[-1, s.idx])
      ypred[, s.idx] <- rzipois(num_provinces, lambda = lambda, pstr0 = theta1)
    }

    # Output
    # GID_2, Date, sample, prediction, Cases
    log_info("Preparing output...")
    df_date <- data.table(
        GID_2 = rep(province_names_ordered, each = samples),
        Date = rep(forecast_date, num_provinces * samples),
        sample = rep(1:samples, times = num_provinces),
        prediction = as.vector(t(ypred)),  # (all samples together, per region)
        Cases = rep(cases_dt[TIME == i, CASES], each = samples)
    )

    df_samples <- rbind(df_samples, df_date)
}

print(df_samples)

# ---- Save predictions as NetCDF ----
log_info("Saving predictions...")
fwrite(df_samples, output_filename, row.names = FALSE)
