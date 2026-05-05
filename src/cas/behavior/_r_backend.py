"""Internal R backend for the behavioral hazard pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap

import pandas as pd


BACKEND_ID = "r_behavior_backend"
BACKEND_NOTE = (
    "Behavioral hazard models are fit through the R backend, using either fixed-effect glm "
    "or glmmTMB mixed-effects models depending on config."
)


def _write_table_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _run_rscript(script_text: str, args: list[str], *, verbose: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="cas_behavior_r_") as tmpdir:
        script_path = Path(tmpdir) / "run_behavior.R"
        script_path.write_text(script_text, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path), *args],
            capture_output=not verbose,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr or ""
            stdout = completed.stdout or ""
            message = stderr.strip() or stdout.strip() or f"Rscript failed with exit code {completed.returncode}"
            raise RuntimeError(message)


def run_r_lag_selection(
    *,
    fpp_table: pd.DataFrame,
    candidate_lags_ms: list[int],
    model_backend: str,
    lag_selection_criterion: str,
    score_path: Path,
    selected_path: Path,
    lag_sensitivity_path: Path | None = None,
    verbose: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix="cas_behavior_lag_") as tmpdir:
        tmp_root = Path(tmpdir)
        fpp_csv = _write_table_csv(fpp_table, tmp_root / "fpp.csv")
        spec_json = tmp_root / "spec.json"
        spec_json.write_text(
            json.dumps(
                {
                    "candidate_lags_ms": [int(v) for v in candidate_lags_ms],
                    "model_backend": str(model_backend),
                    "lag_selection_criterion": str(lag_selection_criterion),
                    "score_path": str(score_path),
                    "selected_path": str(selected_path),
                    "lag_sensitivity_path": str(lag_sensitivity_path) if lag_sensitivity_path else None,
                    "verbose": bool(verbose),
                }
            ),
            encoding="utf-8",
        )
        _run_rscript(_lag_selection_script(), [str(fpp_csv), str(spec_json)], verbose=verbose)


def run_r_model_bundle(
    *,
    fpp_table: pd.DataFrame,
    spp_table: pd.DataFrame,
    pooled_table: pd.DataFrame,
    selected_lag_ms: int,
    candidate_lags_ms: list[int],
    model_backend: str,
    lag_selection_criterion: str,
    model_specs: list[dict[str, object]],
    comparison_specs: list[dict[str, object]],
    model_metrics_path: Path,
    coefficient_path: Path,
    comparison_path: Path,
    convergence_path: Path,
    lag_sensitivity_path: Path | None,
    figure_prediction_path: Path | None = None,
    timing_heatmap_path: Path | None = None,
    three_way_path: Path | None = None,
    verbose: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix="cas_behavior_models_") as tmpdir:
        tmp_root = Path(tmpdir)
        fpp_csv = _write_table_csv(fpp_table, tmp_root / "fpp.csv")
        spp_csv = _write_table_csv(spp_table, tmp_root / "spp.csv")
        pooled_csv = _write_table_csv(pooled_table, tmp_root / "pooled.csv")
        spec_json = tmp_root / "spec.json"
        spec_json.write_text(
            json.dumps(
                {
                    "selected_lag_ms": int(selected_lag_ms),
                    "candidate_lags_ms": [int(v) for v in candidate_lags_ms],
                    "model_backend": str(model_backend),
                    "lag_selection_criterion": str(lag_selection_criterion),
                    "model_specs": model_specs,
                    "comparison_specs": comparison_specs,
                    "model_metrics_path": str(model_metrics_path),
                    "coefficient_path": str(coefficient_path),
                    "comparison_path": str(comparison_path),
                    "convergence_path": str(convergence_path),
                    "lag_sensitivity_path": str(lag_sensitivity_path) if lag_sensitivity_path else None,
                    "figure_prediction_path": str(figure_prediction_path) if figure_prediction_path else None,
                    "timing_heatmap_path": str(timing_heatmap_path) if timing_heatmap_path else None,
                    "three_way_path": str(three_way_path) if three_way_path else None,
                    "verbose": bool(verbose),
                }
            ),
            encoding="utf-8",
        )
        _run_rscript(_model_bundle_script(), [str(fpp_csv), str(spp_csv), str(pooled_csv), str(spec_json)], verbose=verbose)


def _common_r_helpers() -> str:
    return r"""
      log_msg <- function(...) {
        if (isTRUE(spec$verbose)) {
          cat(paste0(..., "\n"))
          flush.console()
        }
      }
      progress_bar <- function(current, total, width = 24) {
        if (!isTRUE(spec$verbose)) return("")
        total <- max(1L, as.integer(total))
        current <- max(0L, min(as.integer(current), total))
        filled <- as.integer(round(width * current / total))
        empty <- max(0L, width - filled)
        paste0("[", paste0(rep("=", filled), collapse = ""), paste0(rep(".", empty), collapse = ""), "]")
      }
      backend_runtime_id <- if (identical(spec$model_backend, "glmm")) "r_glmmtmb_binomial_mixed" else "r_glm_binomial_fixed"
      covariance_type <- if (identical(spec$model_backend, "glm")) "model_based" else "mixed_model"
      random_effects_label <- if (identical(spec$model_backend, "glmm")) "(1 | dyad_id) + (1 | subject)" else "none"
      criterion_label <- if (identical(spec$lag_selection_criterion, "log_likelihood")) "log_likelihood" else "bic"

      add_random_effects <- function(formula_text) {
        paste0(formula_text, " + (1 | dyad_id) + (1 | subject)")
      }
      fit_model <- function(formula_fixed, formula_full, data) {
        warning_messages <- character()
        started_at <- proc.time()[["elapsed"]]
        fit <- withCallingHandlers(
          tryCatch(
            {
              if (identical(spec$model_backend, "glmm")) {
                glmmTMB::glmmTMB(
                  formula = as.formula(formula_full),
                  data = data,
                  family = stats::binomial(link = "logit"),
                  REML = FALSE
                )
              } else {
                stats::glm(
                  formula = stats::as.formula(formula_fixed),
                  data = data,
                  family = stats::binomial(link = "logit")
                )
              }
            },
            error = function(e) e
          ),
          warning = function(w) {
            warning_messages <<- c(warning_messages, conditionMessage(w))
            invokeRestart("muffleWarning")
          }
        )
        elapsed_s <- as.numeric(proc.time()[["elapsed"]] - started_at)
        list(fit = fit, warnings = unique(warning_messages), elapsed_s = elapsed_s)
      }
      extract_metrics <- function(fit) {
        loglik_obj <- logLik(fit)
        list(
          n = stats::nobs(fit),
          k = as.integer(attr(loglik_obj, "df")),
          logLik = as.numeric(loglik_obj),
          AIC = stats::AIC(fit),
          BIC = stats::BIC(fit)
        )
      }
      is_converged <- function(fit) {
        if (inherits(fit, "glmmTMB")) {
          return(isTRUE(fit$sdr$pdHess))
        }
        if (!is.null(fit$converged)) {
          return(isTRUE(fit$converged))
        }
        TRUE
      }
      is_boundary <- function(fit) {
        if (inherits(fit, "glmmTMB")) {
          return(!isTRUE(fit$sdr$pdHess))
        }
        FALSE
      }
      fit_coefficient_table <- function(fit, model_id, anchor_subset, selected_lag_ms, formula_fixed, formula_full, warnings_text) {
        if (inherits(fit, "glmmTMB")) {
          coef_table <- as.data.frame(summary(fit)$coefficients$cond)
          coef_table$term <- rownames(coef_table)
          names(coef_table) <- c("estimate", "std_error", "z_value", "p_value", "term")
        } else {
          coef_table <- as.data.frame(summary(fit)$coefficients)
          coef_table$term <- rownames(coef_table)
          names(coef_table) <- c("estimate", "std_error", "z_value", "p_value", "term")
        }
        coef_table$ci_low <- coef_table$estimate - 1.96 * coef_table$std_error
        coef_table$ci_high <- coef_table$estimate + 1.96 * coef_table$std_error
        coef_table$model_id <- model_id
        coef_table$anchor_subset <- anchor_subset
        coef_table$model_backend <- spec$model_backend
        coef_table$lag_selection_criterion <- criterion_label
        coef_table$selected_lag_ms <- as.integer(selected_lag_ms)
        coef_table$formula_fixed <- formula_fixed
        coef_table$formula_full <- formula_full
        coef_table$random_effects <- random_effects_label
        coef_table$covariance_type <- covariance_type
        coef_table$warnings <- warnings_text
        coef_table[, c("anchor_subset", "model_id", "term", "estimate", "std_error", "ci_low", "ci_high", "p_value", "z_value", "model_backend", "lag_selection_criterion", "selected_lag_ms", "formula_fixed", "formula_full", "random_effects", "covariance_type", "warnings")]
      }
      first_level <- function(x) {
        if (is.factor(x)) return(levels(x)[1])
        unique_values <- unique(x)
        unique_values[[1]]
      }
    """


def _lag_selection_script() -> str:
    return textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{
          library(jsonlite)
          library(glmmTMB)
        }})

        args <- commandArgs(trailingOnly = TRUE)
        fpp_csv <- args[[1]]
        spec_path <- args[[2]]
        spec <- jsonlite::fromJSON(spec_path, simplifyVector = TRUE)
        { _common_r_helpers() }

        fpp <- read.csv(fpp_csv, stringsAsFactors = FALSE)
        fpp$subject <- as.factor(fpp$subject)
        fpp$dyad_id <- as.factor(fpp$dyad_id)

        lag_rows <- list()
        lag_sensitivity_rows <- list()
        total_lags <- length(spec$candidate_lags_ms)
        for (i in seq_along(spec$candidate_lags_ms)) {{
          lag_ms <- as.integer(spec$candidate_lags_ms[[i]])
          rate_col <- sprintf("z_information_rate_lag_%s", lag_ms)
          prop_col <- sprintf("z_prop_expected_cum_info_lag_%s", lag_ms)
          formula_fixed <- sprintf(
            "event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + %s + %s",
            rate_col,
            prop_col
          )
          formula_full <- add_random_effects(formula_fixed)
          log_msg("[behavior hazard][R] ", progress_bar(i, total_lags), " Shared M_3 lag ", i, "/", total_lags, " (", lag_ms, " ms)")
          fit_result <- fit_model(formula_fixed, formula_full, fpp)
          fit <- fit_result$fit
          if (inherits(fit, "error")) {{
            lag_rows[[length(lag_rows) + 1]] <- data.frame(
              lag_ms = lag_ms,
              model_id = "M_3",
              anchor_subset = "fpp",
              model_backend = spec$model_backend,
              lag_selection_criterion = criterion_label,
              n = nrow(fpp),
              k = NA_integer_,
              logLik = NA_real_,
              AIC = NA_real_,
              BIC = NA_real_,
              formula_fixed = formula_fixed,
              formula_full = formula_full,
              converged = FALSE,
              warnings = conditionMessage(fit),
              stringsAsFactors = FALSE
            )
            next
          }}
          metrics <- extract_metrics(fit)
          warning_text <- paste(unique(fit_result$warnings), collapse = " | ")
          lag_rows[[length(lag_rows) + 1]] <- data.frame(
            lag_ms = lag_ms,
            model_id = "M_3",
            anchor_subset = "fpp",
            model_backend = spec$model_backend,
            lag_selection_criterion = criterion_label,
            n = as.integer(metrics$n),
            k = as.integer(metrics$k),
            logLik = as.numeric(metrics$logLik),
            AIC = as.numeric(metrics$AIC),
            BIC = as.numeric(metrics$BIC),
            formula_fixed = formula_fixed,
            formula_full = formula_full,
            converged = is_converged(fit),
            warnings = warning_text,
            stringsAsFactors = FALSE
          )
          coef_table <- fit_coefficient_table(fit, "M_3", "fpp", lag_ms, formula_fixed, formula_full, warning_text)
          for (term_name in c(rate_col, prop_col)) {{
            row <- coef_table[coef_table$term == term_name, , drop = FALSE]
            if (!nrow(row)) next
            lag_sensitivity_rows[[length(lag_sensitivity_rows) + 1]] <- data.frame(
              candidate_lag_ms = lag_ms,
              term = term_name,
              predictor = ifelse(grepl("information_rate", term_name), "information_rate", "prop_expected_cum_info"),
              estimate = row$estimate[[1]],
              ci_low = row$ci_low[[1]],
              ci_high = row$ci_high[[1]],
              backend = backend_runtime_id,
              stringsAsFactors = FALSE
            )
          }}
          log_msg("[behavior hazard][R] ", progress_bar(i, total_lags), " done in ", sprintf("%.1f", fit_result$elapsed_s), "s")
        }}

        scores <- do.call(rbind, lag_rows)
        finite_bic <- is.finite(scores$BIC)
        finite_ll <- is.finite(scores$logLik)
        min_bic <- if (any(finite_bic)) min(scores$BIC[finite_bic]) else NA_real_
        max_ll <- if (any(finite_ll)) max(scores$logLik[finite_ll]) else NA_real_
        scores$delta_BIC <- scores$BIC - min_bic
        scores$delta_logLik <- scores$logLik - max_ll
        scores$rank_by_BIC <- rank(scores$BIC, ties.method = "first", na.last = "keep")
        scores$rank_by_logLik <- rank(-scores$logLik, ties.method = "first", na.last = "keep")
        if (identical(criterion_label, "bic")) {{
          selected_idx <- which.min(scores$BIC)
        }} else {{
          selected_idx <- which.max(scores$logLik)
        }}
        selected_idx <- selected_idx[[1]]
        scores$selected <- FALSE
        scores$selected[selected_idx] <- TRUE
        selected_lag_ms <- as.integer(scores$lag_ms[selected_idx])

        payload <- list(
          selected_lag_ms = selected_lag_ms,
          selector_model_id = "M_3",
          anchor_subset = "fpp",
          model_backend = spec$model_backend,
          lag_selection_criterion = criterion_label,
          formula_fixed = as.character(scores$formula_fixed[selected_idx]),
          formula_full = as.character(scores$formula_full[selected_idx]),
          random_effects = random_effects_label,
          covariance_type = covariance_type
        )
        dir.create(dirname(spec$score_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(scores[, c("lag_ms", "model_id", "anchor_subset", "model_backend", "lag_selection_criterion", "n", "k", "logLik", "AIC", "BIC", "delta_logLik", "delta_BIC", "rank_by_logLik", "rank_by_BIC", "selected", "formula_fixed", "formula_full", "converged", "warnings")], spec$score_path, row.names = FALSE)
        dir.create(dirname(spec$selected_path), recursive = TRUE, showWarnings = FALSE)
        writeLines(jsonlite::toJSON(payload, auto_unbox = TRUE, pretty = TRUE), spec$selected_path)
        if (!is.null(spec$lag_sensitivity_path)) {{
          lag_sensitivity <- if (length(lag_sensitivity_rows)) do.call(rbind, lag_sensitivity_rows) else data.frame(candidate_lag_ms=numeric(), term=character(), predictor=character(), estimate=numeric(), ci_low=numeric(), ci_high=numeric(), backend=character(), stringsAsFactors = FALSE)
          dir.create(dirname(spec$lag_sensitivity_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(lag_sensitivity, spec$lag_sensitivity_path, row.names = FALSE)
        }}
      """
    )


def _model_bundle_script() -> str:
    return textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{
          library(jsonlite)
          library(glmmTMB)
        }})

        args <- commandArgs(trailingOnly = TRUE)
        fpp_csv <- args[[1]]
        spp_csv <- args[[2]]
        pooled_csv <- args[[3]]
        spec_path <- args[[4]]
        spec <- jsonlite::fromJSON(spec_path, simplifyVector = FALSE)
        { _common_r_helpers() }

        fpp <- read.csv(fpp_csv, stringsAsFactors = FALSE)
        spp <- read.csv(spp_csv, stringsAsFactors = FALSE)
        pooled <- read.csv(pooled_csv, stringsAsFactors = FALSE)
        for (table_name in c("fpp", "spp", "pooled")) {{
          table <- get(table_name)
          table$subject <- as.factor(table$subject)
          table$dyad_id <- as.factor(table$dyad_id)
          if ("anchor_type" %in% names(table)) {{
            table$anchor_type <- factor(table$anchor_type, levels = c("SPP", "FPP"))
          }}
          assign(table_name, table)
        }}

        dataset_lookup <- list(FPP = fpp, SPP = spp, pooled_fpp_spp = pooled)
        fit_lookup <- list()
        model_rows <- list()
        coefficient_rows <- list()
        warning_rows <- list()

        total_models <- length(spec$model_specs)
        for (i in seq_along(spec$model_specs)) {{
          model_spec <- spec$model_specs[[i]]
          model_id <- model_spec$model_id
          anchor_subset <- model_spec$anchor_subset
          dataset_name <- model_spec$dataset
          formula_fixed <- model_spec$formula_fixed
          formula_full <- model_spec$formula_full
          data <- dataset_lookup[[dataset_name]]
          log_msg("[behavior hazard][R] [model ", i, "/", total_models, "] start ", model_id, " on ", anchor_subset, " (rows=", nrow(data), ", events=", sum(data$event, na.rm = TRUE), ")")
          fit_result <- fit_model(formula_fixed, formula_full, data)
          fit <- fit_result$fit
          if (inherits(fit, "error")) {{
            warning_rows[[length(warning_rows) + 1]] <- data.frame(model_id=model_id, anchor_subset=anchor_subset, warning=conditionMessage(fit), stringsAsFactors = FALSE)
            model_rows[[length(model_rows) + 1]] <- data.frame(
              anchor_subset = anchor_subset,
              model_id = model_id,
              model_backend = spec$model_backend,
              lag_selection_criterion = criterion_label,
              selected_lag_ms = as.integer(spec$selected_lag_ms),
              formula_fixed = formula_fixed,
              formula_full = formula_full,
              random_effects = random_effects_label,
              covariance_type = covariance_type,
              n = nrow(data),
              k = NA_integer_,
              logLik = NA_real_,
              AIC = NA_real_,
              BIC = NA_real_,
              converged = FALSE,
              warnings = conditionMessage(fit),
              stringsAsFactors = FALSE
            )
            next
          }}
          metrics <- extract_metrics(fit)
          warnings_text <- paste(unique(fit_result$warnings), collapse = " | ")
          fit_lookup[[paste(anchor_subset, model_id, sep="::")]] <- fit
          model_rows[[length(model_rows) + 1]] <- data.frame(
            anchor_subset = anchor_subset,
            model_id = model_id,
            model_backend = spec$model_backend,
            lag_selection_criterion = criterion_label,
            selected_lag_ms = as.integer(spec$selected_lag_ms),
            formula_fixed = formula_fixed,
            formula_full = formula_full,
            random_effects = random_effects_label,
            covariance_type = covariance_type,
            n = as.integer(metrics$n),
            k = as.integer(metrics$k),
            logLik = as.numeric(metrics$logLik),
            AIC = as.numeric(metrics$AIC),
            BIC = as.numeric(metrics$BIC),
            converged = is_converged(fit),
            warnings = warnings_text,
            stringsAsFactors = FALSE
          )
          coefficient_rows[[length(coefficient_rows) + 1]] <- fit_coefficient_table(fit, model_id, anchor_subset, spec$selected_lag_ms, formula_fixed, formula_full, warnings_text)
          log_msg("[behavior hazard][R] [model ", i, "/", total_models, "] done ", model_id, " in ", sprintf("%.1f", fit_result$elapsed_s), "s")
        }}

        model_metrics <- if (length(model_rows)) do.call(rbind, model_rows) else data.frame()
        coefficients <- if (length(coefficient_rows)) do.call(rbind, coefficient_rows) else data.frame()
        warnings_table <- if (length(warning_rows)) do.call(rbind, warning_rows) else data.frame(model_id=character(), anchor_subset=character(), warning=character(), stringsAsFactors = FALSE)

        comparison_rows <- list()
        for (comparison_spec in spec$comparison_specs) {{
          parent_anchor <- comparison_spec$anchor_subset
          child_anchor <- comparison_spec$anchor_subset
          parent_id <- comparison_spec$parent_model_id
          child_id <- comparison_spec$child_model_id
          parent_key <- paste(parent_anchor, parent_id, sep="::")
          child_key <- paste(child_anchor, child_id, sep="::")
          parent_fit <- fit_lookup[[parent_key]]
          child_fit <- fit_lookup[[child_key]]
          if (is.null(parent_fit) || is.null(child_fit)) next
          parent_metrics <- extract_metrics(parent_fit)
          child_metrics <- extract_metrics(child_fit)
          valid_nested <- identical(as.integer(parent_metrics$n), as.integer(child_metrics$n))
          df_diff <- as.integer(child_metrics$k - parent_metrics$k)
          lrt_stat <- 2.0 * (as.numeric(child_metrics$logLik) - as.numeric(parent_metrics$logLik))
          p_value <- if (isTRUE(valid_nested) && is.finite(df_diff) && df_diff > 0) stats::pchisq(lrt_stat, df = df_diff, lower.tail = FALSE) else NA_real_
          comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
            comparison_id = comparison_spec$comparison_id,
            anchor_subset = comparison_spec$anchor_subset,
            model_backend = spec$model_backend,
            lag_selection_criterion = criterion_label,
            selected_lag_ms = as.integer(spec$selected_lag_ms),
            parent_model_id = parent_id,
            child_model_id = child_id,
            parent_n = as.integer(parent_metrics$n),
            child_n = as.integer(child_metrics$n),
            parent_k = as.integer(parent_metrics$k),
            child_k = as.integer(child_metrics$k),
            parent_logLik = as.numeric(parent_metrics$logLik),
            child_logLik = as.numeric(child_metrics$logLik),
            parent_AIC = as.numeric(parent_metrics$AIC),
            child_AIC = as.numeric(child_metrics$AIC),
            parent_BIC = as.numeric(parent_metrics$BIC),
            child_BIC = as.numeric(child_metrics$BIC),
            delta_logLik = as.numeric(child_metrics$logLik - parent_metrics$logLik),
            delta_AIC = as.numeric(child_metrics$AIC - parent_metrics$AIC),
            delta_BIC = as.numeric(child_metrics$BIC - parent_metrics$BIC),
            df_diff = df_diff,
            lrt_statistic = lrt_stat,
            p_value = p_value,
            valid_nested_comparison = isTRUE(valid_nested),
            warnings = if (isTRUE(valid_nested)) "" else "Row masks differed across models.",
            stringsAsFactors = FALSE
          )
        }}
        comparisons <- if (length(comparison_rows)) do.call(rbind, comparison_rows) else data.frame()

        predict_frame <- function(fit, newdata) {{
          if (inherits(fit, "glmmTMB")) {{
            pred <- predict(fit, newdata = newdata, type = "link", se.fit = FALSE, re.form = NA, allow.new.levels = TRUE)
            eta <- as.numeric(pred)
            se <- rep(NA_real_, length(eta))
          }} else {{
            pred <- predict(fit, newdata = newdata, type = "link", se.fit = TRUE)
            eta <- as.numeric(pred$fit)
            se <- as.numeric(pred$se.fit)
          }}
          out <- newdata
          out$predicted_hazard <- stats::plogis(eta)
          out$ci_low <- ifelse(is.finite(se), stats::plogis(eta - 1.96 * se), NA_real_)
          out$ci_high <- ifelse(is.finite(se), stats::plogis(eta + 1.96 * se), NA_real_)
          out
        }}

        selected_lag_ms <- as.integer(spec$selected_lag_ms)
        rate_col <- sprintf("z_information_rate_lag_%s", selected_lag_ms)
        prop_col <- sprintf("z_prop_expected_cum_info_lag_%s", selected_lag_ms)
        timing_columns <- c("z_time_from_partner_onset_s", "z_time_from_partner_offset_s", "z_time_from_partner_offset_s_squared")

        if (!is.null(spec$figure_prediction_path) && !is.null(fit_lookup[["fpp::M_3"]]) && !is.null(fit_lookup[["pooled::M_pooled_anchor_interaction"]])) {{
          log_msg("[behavior hazard][R] Building Figure 2 prediction table")
          figure_rows <- list()
          common_columns <- c("figure", "panel", "anchor_type", "predictor", "x_value_z", "x_value_original", "predicted_hazard", "ci_low", "ci_high")
          rate_z <- seq(stats::quantile(fpp[[rate_col]], 0.05, na.rm = TRUE), stats::quantile(fpp[[rate_col]], 0.95, na.rm = TRUE), length.out = 40)
          prop_z <- seq(stats::quantile(fpp[[prop_col]], 0.05, na.rm = TRUE), stats::quantile(fpp[[prop_col]], 0.95, na.rm = TRUE), length.out = 40)
          rate_raw <- seq(stats::quantile(fpp$information_rate, 0.05, na.rm = TRUE), stats::quantile(fpp$information_rate, 0.95, na.rm = TRUE), length.out = 40)
          prop_raw <- seq(stats::quantile(fpp$prop_expected_cum_info, 0.05, na.rm = TRUE), stats::quantile(fpp$prop_expected_cum_info, 0.95, na.rm = TRUE), length.out = 40)
          base_vals <- lapply(c(timing_columns, rate_col, prop_col), function(name) stats::median(as.numeric(fpp[[name]]), na.rm = TRUE))
          names(base_vals) <- c(timing_columns, rate_col, prop_col)

          panel_a <- data.frame(
            z_time_from_partner_onset_s = rep(base_vals$z_time_from_partner_onset_s, length(rate_z)),
            z_time_from_partner_offset_s = rep(base_vals$z_time_from_partner_offset_s, length(rate_z)),
            z_time_from_partner_offset_s_squared = rep(base_vals$z_time_from_partner_offset_s_squared, length(rate_z)),
            stringsAsFactors = FALSE
          )
          panel_a[[rate_col]] <- rate_z
          panel_a[[prop_col]] <- rep(0.0, length(rate_z))
          pred_a <- predict_frame(fit_lookup[["fpp::M_3"]], panel_a)
          pred_a$figure <- "fig02_primary_information_effects"
          pred_a$panel <- "A"
          pred_a$anchor_type <- "FPP"
          pred_a$predictor <- "information_rate"
          pred_a$x_value_z <- rate_z
          pred_a$x_value_original <- rate_raw
          figure_rows[[length(figure_rows) + 1]] <- pred_a[, common_columns, drop = FALSE]

          panel_b <- panel_a[seq_along(prop_z), timing_columns, drop = FALSE]
          panel_b[[rate_col]] <- rep(0.0, length(prop_z))
          panel_b[[prop_col]] <- prop_z
          pred_b <- predict_frame(fit_lookup[["fpp::M_3"]], panel_b)
          pred_b$figure <- "fig02_primary_information_effects"
          pred_b$panel <- "B"
          pred_b$anchor_type <- "FPP"
          pred_b$predictor <- "prop_expected_cum_info"
          pred_b$x_value_z <- prop_z
          pred_b$x_value_original <- prop_raw
          figure_rows[[length(figure_rows) + 1]] <- pred_b[, common_columns, drop = FALSE]

          pooled_base <- data.frame(
            z_time_from_partner_onset_s = rep(stats::median(as.numeric(pooled$z_time_from_partner_onset_s), na.rm = TRUE), length(rate_z)),
            z_time_from_partner_offset_s = rep(stats::median(as.numeric(pooled$z_time_from_partner_offset_s), na.rm = TRUE), length(rate_z)),
            z_time_from_partner_offset_s_squared = rep(stats::median(as.numeric(pooled$z_time_from_partner_offset_s_squared), na.rm = TRUE), length(rate_z)),
            stringsAsFactors = FALSE
          )
          pooled_base[[prop_col]] <- rep(0.0, length(rate_z))
          for (anchor_label in c("SPP", "FPP")) {{
            panel_c <- pooled_base
            panel_c$anchor_type <- factor(rep(anchor_label, length(rate_z)), levels = levels(pooled$anchor_type))
            panel_c[[rate_col]] <- rate_z
            pred_c <- predict_frame(fit_lookup[["pooled::M_pooled_anchor_interaction"]], panel_c)
            pred_c$figure <- "fig02_primary_information_effects"
            pred_c$panel <- "C"
            pred_c$anchor_type <- anchor_label
            pred_c$predictor <- "information_rate"
            pred_c$x_value_z <- rate_z
            pred_c$x_value_original <- rate_raw
            figure_rows[[length(figure_rows) + 1]] <- pred_c[, common_columns, drop = FALSE]
          }}
          figure_predictions <- do.call(rbind, figure_rows)
          dir.create(dirname(spec$figure_prediction_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(figure_predictions, spec$figure_prediction_path, row.names = FALSE)
        }}

        if (!is.null(spec$timing_heatmap_path) && !is.null(fit_lookup[["fpp::M_4"]])) {{
          log_msg("[behavior hazard][R] Building Figure 3 timing interaction prediction table")
          rate_levels_z <- seq(stats::quantile(fpp[[rate_col]], 0.05, na.rm = TRUE), stats::quantile(fpp[[rate_col]], 0.95, na.rm = TRUE), length.out = 11)
          rate_levels_raw <- seq(stats::quantile(fpp$information_rate, 0.05, na.rm = TRUE), stats::quantile(fpp$information_rate, 0.95, na.rm = TRUE), length.out = 11)
          onset_values_z <- seq(stats::quantile(fpp$z_time_from_partner_onset_s, 0.05, na.rm = TRUE), stats::quantile(fpp$z_time_from_partner_onset_s, 0.95, na.rm = TRUE), length.out = 60)
          offset_values_z <- seq(stats::quantile(fpp$z_time_from_partner_offset_s, 0.05, na.rm = TRUE), stats::quantile(fpp$z_time_from_partner_offset_s, 0.95, na.rm = TRUE), length.out = 60)
          rows <- list()
          for (j in seq_along(rate_levels_z)) {{
            rz <- rate_levels_z[[j]]
            rraw <- rate_levels_raw[[j]]
            onset_grid <- data.frame(
              z_time_from_partner_onset_s = onset_values_z,
              z_time_from_partner_offset_s = rep(0.0, length(onset_values_z)),
              z_time_from_partner_offset_s_squared = rep(0.0, length(onset_values_z)),
              stringsAsFactors = FALSE
            )
            onset_grid[[rate_col]] <- rep(rz, length(onset_values_z))
            onset_grid[[prop_col]] <- rep(0.0, length(onset_values_z))
            pred_onset <- predict_frame(fit_lookup[["fpp::M_4"]], onset_grid)
            pred_onset$panel <- "A"
            pred_onset$timing_reference <- "partner_onset"
            pred_onset$time_value_z <- onset_values_z
            pred_onset$time_value_s <- onset_values_z
            pred_onset$information_rate_z <- rz
            pred_onset$information_rate_original <- rraw
            rows[[length(rows) + 1]] <- pred_onset[, c("panel", "timing_reference", "time_value_s", "time_value_z", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high"), drop = FALSE]

            offset_grid <- data.frame(
              z_time_from_partner_onset_s = rep(0.0, length(offset_values_z)),
              z_time_from_partner_offset_s = offset_values_z,
              z_time_from_partner_offset_s_squared = offset_values_z ^ 2,
              stringsAsFactors = FALSE
            )
            offset_grid[[rate_col]] <- rep(rz, length(offset_values_z))
            offset_grid[[prop_col]] <- rep(0.0, length(offset_values_z))
            pred_offset <- predict_frame(fit_lookup[["fpp::M_4"]], offset_grid)
            pred_offset$panel <- "B"
            pred_offset$timing_reference <- "partner_offset"
            pred_offset$time_value_z <- offset_values_z
            pred_offset$time_value_s <- offset_values_z
            pred_offset$information_rate_z <- rz
            pred_offset$information_rate_original <- rraw
            rows[[length(rows) + 1]] <- pred_offset[, c("panel", "timing_reference", "time_value_s", "time_value_z", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high"), drop = FALSE]
          }}
          timing_predictions <- do.call(rbind, rows)
          dir.create(dirname(spec$timing_heatmap_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(timing_predictions, spec$timing_heatmap_path, row.names = FALSE)
        }}

        if (!is.null(spec$three_way_path)) {{
          dir.create(dirname(spec$three_way_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(data.frame(anchor_type=character(), timing_reference=character(), time_value_s=numeric(), information_rate_z=numeric(), information_rate_original=numeric(), predicted_hazard=numeric(), ci_low=numeric(), ci_high=numeric(), stringsAsFactors = FALSE), spec$three_way_path, row.names = FALSE)
        }}

        dir.create(dirname(spec$model_metrics_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(model_metrics, spec$model_metrics_path, row.names = FALSE)
        dir.create(dirname(spec$coefficient_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(coefficients, spec$coefficient_path, row.names = FALSE)
        dir.create(dirname(spec$comparison_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(comparisons, spec$comparison_path, row.names = FALSE)
        dir.create(dirname(spec$convergence_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(warnings_table, spec$convergence_path, row.names = FALSE)
        if (!is.null(spec$lag_sensitivity_path)) {{
          dir.create(dirname(spec$lag_sensitivity_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(data.frame(candidate_lag_ms=numeric(), term=character(), predictor=character(), estimate=numeric(), ci_low=numeric(), ci_high=numeric(), backend=character(), stringsAsFactors = FALSE), spec$lag_sensitivity_path, row.names = FALSE)
        }}
      """
    )
