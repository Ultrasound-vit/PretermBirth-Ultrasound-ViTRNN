# Statistical analysis for preterm birth prediction models
#
# This publication-oriented script was consolidated from the original analysis
# notes. It contains no patient-level data, institutional identifiers, or
# machine-specific paths. Models are fitted on the training cohort only and
# then evaluated without refitting on the internal and external cohorts.

required_packages <- c("readr", "readxl", "dplyr", "broom", "pROC", "ggplot2", "writexl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Install the required R packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(pROC)
})

set.seed(1234)

# -----------------------------------------------------------------------------
# User-configurable paths
# -----------------------------------------------------------------------------

# Set DATA_DIR and OUTPUT_DIR as environment variables, or replace the relative
# defaults below. Do not commit patient data to a public repository.
data_dir <- Sys.getenv("DATA_DIR", unset = "data")
output_dir <- Sys.getenv("OUTPUT_DIR", unset = "outputs/statistical_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

training_file <- file.path(data_dir, "training_cohort.csv")
internal_file <- file.path(data_dir, "internal_validation_cohort.csv")
external_file <- file.path(data_dir, "external_validation_cohort.csv")

# Canonical analysis variables. Change these only if the manuscript model uses
# a different prespecified predictor set.
outcome <- "ending"                  # 0 = term birth, 1 = preterm birth
clinical_predictors <- c(
  "cervical_length",
  "hemoglobin",
  "ART",
  "number_of_fetuses"
)
imaging_predictor <- "img_score"

# -----------------------------------------------------------------------------
# Data import and validation
# -----------------------------------------------------------------------------

read_cohort <- function(path) {
  if (!file.exists(path)) {
    stop("Input file not found: ", path)
  }

  extension <- tolower(tools::file_ext(path))
  if (extension == "csv") {
    return(as.data.frame(readr::read_csv(path, show_col_types = FALSE)))
  }
  if (extension %in% c("xlsx", "xls")) {
    return(as.data.frame(readxl::read_excel(path)))
  }
  stop("Unsupported input format: ", extension, ". Use CSV, XLSX, or XLS.")
}

standardize_columns <- function(data) {
  aliases <- list(
    cervical_length = c("cervical_length", "Cervical_length", "length"),
    hemoglobin = c("hemoglobin", "Hb", "HGB2"),
    number_of_fetuses = c("number_of_fetuses", "Number_of_fetuses"),
    ART = c("ART"),
    img_score = c("img_score"),
    ending = c("ending")
  )

  for (canonical_name in names(aliases)) {
    candidates <- intersect(aliases[[canonical_name]], names(data))
    if (!(canonical_name %in% names(data)) && length(candidates) > 0) {
      names(data)[names(data) == candidates[1]] <- canonical_name
    }
  }
  data
}

validate_cohort <- function(data, cohort_name, required_columns) {
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0) {
    stop(
      cohort_name,
      " is missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }

  observed_outcomes <- unique(stats::na.omit(data[[outcome]]))
  if (!all(observed_outcomes %in% c(0, 1))) {
    stop(cohort_name, " outcome must be coded as 0 or 1.")
  }
}

required_columns <- unique(c(outcome, clinical_predictors, imaging_predictor))
training_data <- standardize_columns(read_cohort(training_file))
internal_data <- standardize_columns(read_cohort(internal_file))
external_data <- standardize_columns(read_cohort(external_file))

validate_cohort(training_data, "Training cohort", required_columns)
validate_cohort(internal_data, "Internal validation cohort", required_columns)
validate_cohort(external_data, "External validation cohort", required_columns)

# Complete-case analysis follows the original analysis strategy. Report the
# number excluded from each cohort in the manuscript or study flow diagram.
complete_cases <- function(data, variables, cohort_name) {
  retained <- stats::complete.cases(data[, variables, drop = FALSE])
  message(
    cohort_name, ": retained ", sum(retained), " of ", nrow(data),
    " participants after complete-case filtering."
  )
  data[retained, , drop = FALSE]
}

training_data <- complete_cases(training_data, required_columns, "Training cohort")
internal_data <- complete_cases(internal_data, required_columns, "Internal validation cohort")
external_data <- complete_cases(external_data, required_columns, "External validation cohort")

# -----------------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------------

make_formula <- function(predictors) {
  stats::reformulate(predictors, response = outcome)
}

model_formulas <- list(
  Clinical = make_formula(clinical_predictors),
  Imaging = make_formula(imaging_predictor),
  Multimodal = make_formula(c(clinical_predictors, imaging_predictor))
)

# All models are fitted once using the training cohort.
models <- lapply(
  model_formulas,
  function(formula) stats::glm(formula, data = training_data, family = binomial())
)

coefficient_tables <- lapply(names(models), function(model_name) {
  broom::tidy(models[[model_name]], conf.int = TRUE, exponentiate = TRUE) |>
    filter(term != "(Intercept)") |>
    mutate(Model = model_name, .before = 1)
})
coefficient_table <- bind_rows(coefficient_tables)
writexl::write_xlsx(
  coefficient_table,
  file.path(output_dir, "model_coefficients.xlsx")
)

# -----------------------------------------------------------------------------
# Discrimination and classification metrics
# -----------------------------------------------------------------------------

safe_divide <- function(numerator, denominator) {
  if (denominator == 0) return(NA_real_)
  numerator / denominator
}

classification_metrics <- function(observed, probability, threshold) {
  predicted <- as.integer(probability >= threshold)
  true_positive <- sum(predicted == 1 & observed == 1)
  true_negative <- sum(predicted == 0 & observed == 0)
  false_positive <- sum(predicted == 1 & observed == 0)
  false_negative <- sum(predicted == 0 & observed == 1)

  data.frame(
    Threshold = threshold,
    Sensitivity = safe_divide(true_positive, true_positive + false_negative),
    Specificity = safe_divide(true_negative, true_negative + false_positive),
    Accuracy = safe_divide(true_positive + true_negative, length(observed)),
    Precision = safe_divide(true_positive, true_positive + false_positive),
    F1 = safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative),
    TN = true_negative,
    FP = false_positive,
    FN = false_negative,
    TP = true_positive
  )
}

evaluate_model <- function(model, model_name, data, cohort_name, threshold) {
  probability <- stats::predict(model, newdata = data, type = "response")
  observed <- data[[outcome]]
  roc_object <- pROC::roc(
    response = observed,
    predictor = probability,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  auc_ci <- as.numeric(pROC::ci.auc(roc_object, method = "delong"))

  metrics <- classification_metrics(observed, probability, threshold) |>
    mutate(
      Model = model_name,
      Cohort = cohort_name,
      N = length(observed),
      AUC = as.numeric(pROC::auc(roc_object)),
      AUC_Lower_95CI = auc_ci[1],
      AUC_Upper_95CI = auc_ci[3],
      Brier_score = mean((probability - observed)^2),
      .before = 1
    )

  list(roc = roc_object, probability = probability, metrics = metrics)
}

# Select each model's threshold using the training cohort only, then keep that
# threshold fixed for both validation cohorts.
training_thresholds <- vapply(names(models), function(model_name) {
  probability <- stats::predict(models[[model_name]], training_data, type = "response")
  roc_object <- pROC::roc(
    training_data[[outcome]], probability,
    levels = c(0, 1), direction = "<", quiet = TRUE
  )
  as.numeric(
    pROC::coords(
      roc_object, x = "best", best.method = "youden",
      ret = "threshold", transpose = FALSE
    )[1, 1]
  )
}, numeric(1))

cohorts <- list(
  Training = training_data,
  Internal_validation = internal_data,
  External_validation = external_data
)

evaluation <- list()
for (cohort_name in names(cohorts)) {
  evaluation[[cohort_name]] <- list()
  for (model_name in names(models)) {
    evaluation[[cohort_name]][[model_name]] <- evaluate_model(
      model = models[[model_name]],
      model_name = model_name,
      data = cohorts[[cohort_name]],
      cohort_name = cohort_name,
      threshold = training_thresholds[[model_name]]
    )
  }
}

performance_table <- bind_rows(lapply(evaluation, function(cohort_results) {
  bind_rows(lapply(cohort_results, `[[`, "metrics"))
}))
writexl::write_xlsx(
  performance_table,
  file.path(output_dir, "model_performance.xlsx")
)

# Paired DeLong tests compare AUCs within each validation cohort.
delong_comparisons <- function(results, cohort_name) {
  comparisons <- list(
    c("Multimodal", "Clinical"),
    c("Multimodal", "Imaging"),
    c("Clinical", "Imaging")
  )
  bind_rows(lapply(comparisons, function(pair) {
    test <- pROC::roc.test(
      results[[pair[1]]]$roc,
      results[[pair[2]]]$roc,
      paired = TRUE,
      method = "delong"
    )
    data.frame(
      Cohort = cohort_name,
      Model_1 = pair[1],
      Model_2 = pair[2],
      P_value = as.numeric(test$p.value)
    )
  }))
}

delong_table <- bind_rows(
  delong_comparisons(evaluation$Internal_validation, "Internal validation"),
  delong_comparisons(evaluation$External_validation, "External validation")
)
writexl::write_xlsx(delong_table, file.path(output_dir, "delong_tests.xlsx"))

# -----------------------------------------------------------------------------
# ROC plots
# -----------------------------------------------------------------------------

plot_roc_curves <- function(results, cohort_title, output_path) {
  roc_data <- bind_rows(lapply(names(results), function(model_name) {
    roc_object <- results[[model_name]]$roc
    data.frame(
      False_positive_rate = 1 - roc_object$specificities,
      Sensitivity = roc_object$sensitivities,
      Model = paste0(
        model_name,
        " (AUC = ", sprintf("%.3f", as.numeric(pROC::auc(roc_object))), ")"
      )
    )
  }))

  figure <- ggplot(roc_data, aes(False_positive_rate, Sensitivity, color = Model)) +
    geom_line(linewidth = 0.9) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    coord_equal() +
    labs(
      title = cohort_title,
      x = "1 - Specificity",
      y = "Sensitivity",
      color = "Model"
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom"
    )

  ggsave(output_path, figure, width = 6.5, height = 6.0, dpi = 300)
}

plot_roc_curves(
  evaluation$Internal_validation,
  "Internal validation cohort",
  file.path(output_dir, "roc_internal_validation.png")
)
plot_roc_curves(
  evaluation$External_validation,
  "External validation cohort",
  file.path(output_dir, "roc_external_validation.png")
)

# -----------------------------------------------------------------------------
# Calibration assessment
# -----------------------------------------------------------------------------

calibration_statistics <- function(observed, probability, model_name, cohort_name) {
  epsilon <- 1e-6
  bounded_probability <- pmin(pmax(probability, epsilon), 1 - epsilon)
  linear_predictor <- qlogis(bounded_probability)
  calibration_model <- stats::glm(observed ~ linear_predictor, family = binomial())

  data.frame(
    Model = model_name,
    Cohort = cohort_name,
    Calibration_intercept = unname(stats::coef(calibration_model)[1]),
    Calibration_slope = unname(stats::coef(calibration_model)[2]),
    Brier_score = mean((probability - observed)^2)
  )
}

calibration_table <- bind_rows(lapply(names(evaluation), function(cohort_name) {
  bind_rows(lapply(names(evaluation[[cohort_name]]), function(model_name) {
    result <- evaluation[[cohort_name]][[model_name]]
    calibration_statistics(
      cohorts[[cohort_name]][[outcome]],
      result$probability,
      model_name,
      cohort_name
    )
  }))
}))
writexl::write_xlsx(
  calibration_table,
  file.path(output_dir, "calibration_statistics.xlsx")
)

plot_calibration <- function(observed, probability, cohort_title, output_path) {
  calibration_data <- data.frame(observed = observed, probability = probability)
  figure <- ggplot(calibration_data, aes(probability, observed)) +
    geom_smooth(
      method = "loess", formula = y ~ x,
      se = TRUE, color = "#E69F00", fill = "#F6D7A7"
    ) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = cohort_title,
      x = "Predicted probability",
      y = "Observed outcome"
    ) +
    theme_classic(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))

  ggsave(output_path, figure, width = 5.5, height = 5.5, dpi = 300)
}

plot_calibration(
  internal_data[[outcome]],
  evaluation$Internal_validation$Multimodal$probability,
  "Multimodal model in the internal validation cohort",
  file.path(output_dir, "calibration_internal_validation.png")
)
plot_calibration(
  external_data[[outcome]],
  evaluation$External_validation$Multimodal$probability,
  "Multimodal model in the external validation cohort",
  file.path(output_dir, "calibration_external_validation.png")
)

message("Statistical analysis completed. Results were saved to: ", output_dir)
