"""Internal R GLMM backend for the behavioral hazard pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap

import pandas as pd


BACKEND_ID = "r_glmmtmb_binomial_mixed"
BACKEND_NOTE = (
    "Behavioral hazard models are fit through glmmTMB in R with mixed-effects random intercepts, "
    "using the restored timing baseline and within-anchor standardization behavior."
)


def _write_table_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _run_rscript(script_text: str, args: list[str], *, verbose: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="cas_behavior_r_") as tmpdir:
        script_path = Path(tmpdir) / "run_behavior_glmm.R"
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
    pooled_table: pd.DataFrame,
    candidate_lags_ms: list[int],
    score_path: Path,
    selected_path: Path,
    family_summary_path: Path,
    family_rankings_path: Path,
    model_diagnostics_path: Path,
    selector_comparison_path: Path,
    lag_sensitivity_path: Path | None = None,
    verbose: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix="cas_behavior_lag_") as tmpdir:
        tmp_root = Path(tmpdir)
        fpp_csv = _write_table_csv(fpp_table, tmp_root / "fpp.csv")
        pooled_csv = _write_table_csv(pooled_table, tmp_root / "pooled.csv")
        spec_json = tmp_root / "spec.json"
        spec_json.write_text(
            json.dumps(
                {
                    "candidate_lags_ms": [int(v) for v in candidate_lags_ms],
                    "score_path": str(score_path),
                    "selected_path": str(selected_path),
                    "family_summary_path": str(family_summary_path),
                    "family_rankings_path": str(family_rankings_path),
                    "model_diagnostics_path": str(model_diagnostics_path),
                    "selector_comparison_path": str(selector_comparison_path),
                    "lag_sensitivity_path": str(lag_sensitivity_path) if lag_sensitivity_path else None,
                    "verbose": bool(verbose),
                }
            ),
            encoding="utf-8",
        )
        _run_rscript(_lag_selection_script(), [str(fpp_csv), str(pooled_csv), str(spec_json)], verbose=verbose)


def run_r_model_bundle(
    *,
    fpp_table: pd.DataFrame,
    pooled_table: pd.DataFrame,
    selected_lags: dict[str, int],
    candidate_lags_ms: list[int],
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
        pooled_csv = _write_table_csv(pooled_table, tmp_root / "pooled.csv")
        spec_json = tmp_root / "spec.json"
        spec_json.write_text(
            json.dumps(
                {
                    "selected_lags": {str(key): int(value) for key, value in selected_lags.items()},
                    "candidate_lags_ms": [int(v) for v in candidate_lags_ms],
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
        _run_rscript(_model_bundle_script(), [str(fpp_csv), str(pooled_csv), str(spec_json)], verbose=verbose)


def _lag_selection_script() -> str:
    return textwrap.dedent(
        r"""
        suppressPackageStartupMessages({
          library(glmmTMB)
          library(jsonlite)
        })

        args <- commandArgs(trailingOnly = TRUE)
        fpp_csv <- args[[1]]
        pooled_csv <- args[[2]]
        spec_path <- args[[3]]
        spec <- jsonlite::fromJSON(spec_path, simplifyVector = TRUE)
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
        fpp <- read.csv(fpp_csv, stringsAsFactors = FALSE)
        pooled <- read.csv(pooled_csv, stringsAsFactors = FALSE)
        for (table_name in c("fpp", "pooled")) {
          table <- get(table_name)
          table$subject <- as.factor(table$subject)
          table$dyad_id <- as.factor(table$dyad_id)
          table$.__row_id__ <- seq_len(nrow(table))
          if ("anchor_type" %in% names(table)) {
            if (all(c("SPP", "FPP") %in% unique(table$anchor_type))) {
              table$anchor_type <- factor(table$anchor_type, levels = c("SPP", "FPP"))
            } else {
              table$anchor_type <- as.factor(table$anchor_type)
            }
          }
          assign(table_name, table)
        }

        fit_model <- function(formula_text, data) {
          warning_messages <- character()
          started_at <- proc.time()[["elapsed"]]
          fit <- withCallingHandlers(
            tryCatch(
              glmmTMB::glmmTMB(
                formula = as.formula(formula_text),
                data = data,
                family = stats::binomial(link = "logit"),
                REML = FALSE
              ),
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

        model_diagnostic_rows <- list()
        family_rows <- list()
        lag_sensitivity_rows <- list()

        fit_selector <- function(family_label, lag_ms, selector_model_id, reduced_model_id, dataset_name, data, reduced_formula, selector_formula, extra_model_specs, lag_index, total_lags) {
          log_msg(
            "[behavior hazard][R] ",
            progress_bar(lag_index, total_lags),
            " Family ", family_label,
            " lag ", lag_index, "/", total_lags,
            " (", lag_ms, " ms): preparing data for ", selector_model_id
          )
          needed_columns <- unique(c(
            "event",
            "z_time_from_partner_onset_s",
            "z_time_from_partner_offset_s",
            "z_time_from_partner_offset_s_squared",
            "dyad_id",
            "subject",
            if (dataset_name == "pooled_fpp_spp") "anchor_type" else NULL,
            sprintf("z_information_rate_lag_%s", lag_ms),
            sprintf("z_prop_expected_cum_info_lag_%s", lag_ms)
          ))
          working <- data[stats::complete.cases(data[, needed_columns, drop = FALSE]), , drop = FALSE]
          row_sig <- sprintf(
            "n=%d;sum=%0.0f;sumsq=%0.0f",
            nrow(working),
            sum(working$.__row_id__),
            sum(working$.__row_id__ * working$.__row_id__)
          )
          log_msg(
            "[behavior hazard][R] ",
            progress_bar(lag_index, total_lags),
            " Family ", family_label,
            " lag ", lag_ms, " ms uses n=", nrow(working),
            " rows and events=", sum(working$event, na.rm = TRUE)
          )
          log_msg(
            "[behavior hazard][R] ",
            progress_bar(lag_index, total_lags),
            " Family ", family_label,
            " lag ", lag_ms, " ms: fitting reduced model ", reduced_model_id
          )
          reduced_result <- fit_model(reduced_formula, working)
          reduced_fit <- reduced_result$fit
          reduced_ll <- if (inherits(reduced_fit, "error")) NA_real_ else as.numeric(logLik(reduced_fit))
          reduced_bic <- if (inherits(reduced_fit, "error")) NA_real_ else BIC(reduced_fit)
          reduced_k <- if (inherits(reduced_fit, "error")) NA_integer_ else as.integer(attr(logLik(reduced_fit), "df"))
          if (inherits(reduced_fit, "error")) {
            log_msg(
              "[behavior hazard][R] ",
              progress_bar(lag_index, total_lags),
              " Family ", family_label,
              " lag ", lag_ms, " ms: reduced model failed after ",
              sprintf("%.1f", reduced_result$elapsed_s), "s: ",
              conditionMessage(reduced_fit)
            )
          } else {
            log_msg(
              "[behavior hazard][R] ",
              progress_bar(lag_index, total_lags),
              " Family ", family_label,
              " lag ", lag_ms, " ms: reduced model done in ",
              sprintf("%.1f", reduced_result$elapsed_s), "s (logLik=",
              sprintf("%.3f", reduced_ll), ", BIC=", sprintf("%.3f", reduced_bic), ")"
            )
          }
          model_diagnostic_rows[[length(model_diagnostic_rows) + 1]] <<- data.frame(
            family = family_label,
            model_id = reduced_model_id,
            lag_ms = as.integer(lag_ms),
            n = nrow(working),
            k = reduced_k,
            logLik = reduced_ll,
            BIC = reduced_bic,
            delta_logLik = 0.0,
            converged = if (inherits(reduced_fit, "error")) FALSE else TRUE,
            singular_or_boundary = if (inherits(reduced_fit, "error")) TRUE else !isTRUE(reduced_fit$sdr$pdHess),
            warnings = if (inherits(reduced_fit, "error")) conditionMessage(reduced_fit) else paste(unique(reduced_result$warnings), collapse = " | "),
            row_mask_signature = row_sig,
            stringsAsFactors = FALSE
          )

          selector_candidates <- list()
          for (spec_row in extra_model_specs) {
            log_msg(
              "[behavior hazard][R] ",
              progress_bar(lag_index, total_lags),
              " Family ", family_label,
              " lag ", lag_ms, " ms: fitting candidate ", spec_row$model_id
            )
            candidate_result <- fit_model(spec_row$formula, working)
            candidate_fit <- candidate_result$fit
            candidate_ll <- if (inherits(candidate_fit, "error")) NA_real_ else as.numeric(logLik(candidate_fit))
            candidate_bic <- if (inherits(candidate_fit, "error")) NA_real_ else BIC(candidate_fit)
            candidate_k <- if (inherits(candidate_fit, "error")) NA_integer_ else as.integer(attr(logLik(candidate_fit), "df"))
            delta_ll <- candidate_ll - reduced_ll
            if (inherits(candidate_fit, "error")) {
              log_msg(
                "[behavior hazard][R] ",
                progress_bar(lag_index, total_lags),
                " Family ", family_label,
                " lag ", lag_ms, " ms: candidate ", spec_row$model_id,
                " failed after ", sprintf("%.1f", candidate_result$elapsed_s), "s: ",
                conditionMessage(candidate_fit)
              )
            } else {
              log_msg(
                "[behavior hazard][R] ",
                progress_bar(lag_index, total_lags),
                " Family ", family_label,
                " lag ", lag_ms, " ms: candidate ", spec_row$model_id,
                " done in ", sprintf("%.1f", candidate_result$elapsed_s),
                "s (deltaLL=", sprintf("%.3f", delta_ll), ", BIC=", sprintf("%.3f", candidate_bic), ")"
              )
            }
            model_diagnostic_rows[[length(model_diagnostic_rows) + 1]] <<- data.frame(
              family = family_label,
              model_id = spec_row$model_id,
              lag_ms = as.integer(lag_ms),
              n = nrow(working),
              k = candidate_k,
              logLik = candidate_ll,
              BIC = candidate_bic,
              delta_logLik = delta_ll,
              converged = if (inherits(candidate_fit, "error")) FALSE else TRUE,
              singular_or_boundary = if (inherits(candidate_fit, "error")) TRUE else !isTRUE(candidate_fit$sdr$pdHess),
              warnings = if (inherits(candidate_fit, "error")) conditionMessage(candidate_fit) else paste(unique(candidate_result$warnings), collapse = " | "),
              row_mask_signature = row_sig,
              stringsAsFactors = FALSE
            )
            selector_candidates[[length(selector_candidates) + 1]] <- data.frame(
              family = family_label,
              selector_model_id = spec_row$model_id,
              lag_ms = as.integer(lag_ms),
              n = nrow(working),
              k = candidate_k,
              logLik = candidate_ll,
              BIC = candidate_bic,
              delta_logLik = delta_ll,
              row_mask_signature = row_sig,
              stringsAsFactors = FALSE
            )
            if (!is.null(spec$lag_sensitivity_path) && family_label == "A" && spec_row$model_id == "A3_joint_information" && !inherits(candidate_fit, "error")) {
              coef_table <- as.data.frame(summary(candidate_fit)$coefficients$cond)
              coef_table$term <- rownames(coef_table)
              names(coef_table) <- c("estimate", "std_error", "z_value", "p_value", "term")
              for (term_name in c(
                sprintf("z_information_rate_lag_%s", lag_ms),
                sprintf("z_prop_expected_cum_info_lag_%s", lag_ms)
              )) {
                row <- coef_table[coef_table$term == term_name, , drop = FALSE]
                if (nrow(row) == 0) next
                lag_sensitivity_rows[[length(lag_sensitivity_rows) + 1]] <<- data.frame(
                  candidate_lag_ms = as.integer(lag_ms),
                  term = term_name,
                  predictor = ifelse(grepl("information_rate", term_name), "information_rate", "prop_expected_cum_info"),
                  estimate = row$estimate[[1]],
                  ci_low = row$estimate[[1]] - 1.96 * row$std_error[[1]],
                  ci_high = row$estimate[[1]] + 1.96 * row$std_error[[1]],
                  backend = "r_glmmtmb_binomial_mixed",
                  stringsAsFactors = FALSE
                )
              }
            }
          }
          selector_table <- do.call(rbind, selector_candidates)
          finite_delta <- is.finite(selector_table$delta_logLik)
          if (any(finite_delta)) {
            selector_idx <- which.max(selector_table$delta_logLik)
          } else {
            selector_idx <- 1L
          }
          selector_row <- selector_table[selector_idx, , drop = FALSE]
          log_msg(
            "[behavior hazard][R] ",
            progress_bar(lag_index, total_lags),
            " Family ", family_label,
            " lag ", lag_ms, " ms: best selector candidate for this lag is ",
            selector_row$selector_model_id[[1]],
            " (deltaLL=", sprintf("%.3f", selector_row$delta_logLik[[1]]), ")"
          )
          family_rows[[length(family_rows) + 1]] <<- selector_row
        }

        total_lags <- length(spec$candidate_lags_ms)
        lag_index <- 0L
        for (lag_ms in spec$candidate_lags_ms) {
          lag_index <- lag_index + 1L
          fit_selector(
            family_label = "A",
            lag_ms = lag_ms,
            selector_model_id = "A3_joint_information",
            reduced_model_id = "A0_timing",
            dataset_name = "FPP",
            data = fpp,
            reduced_formula = "event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + (1 | dyad_id) + (1 | subject)",
            selector_formula = "",
            extra_model_specs = list(
              list(model_id = "A1_information_rate", formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms)),
              list(model_id = "A2_expected_cum_info", formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_prop_expected_cum_info_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms)),
              list(model_id = "A3_joint_information", formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + z_prop_expected_cum_info_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms))
            ),
            lag_index = lag_index,
            total_lags = total_lags
          )
          fit_selector(
            family_label = "B",
            lag_ms = lag_ms,
            selector_model_id = "B2_anchor_x_information",
            reduced_model_id = "B1_shared_information",
            dataset_name = "pooled_fpp_spp",
            data = pooled,
            reduced_formula = sprintf("event ~ anchor_type + z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + z_prop_expected_cum_info_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms),
            selector_formula = "",
            extra_model_specs = list(
              list(model_id = "B2_anchor_x_information", formula = sprintf("event ~ anchor_type + z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + anchor_type * z_information_rate_lag_%s + anchor_type * z_prop_expected_cum_info_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms))
            ),
            lag_index = lag_index,
            total_lags = total_lags
          )
          fit_selector(
            family_label = "C",
            lag_ms = lag_ms,
            selector_model_id = "C1_onset_x_rate",
            reduced_model_id = "A3_joint_information",
            dataset_name = "FPP",
            data = fpp,
            reduced_formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + z_prop_expected_cum_info_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms),
            selector_formula = "",
            extra_model_specs = list(
              list(model_id = "C1_onset_x_rate", formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + z_prop_expected_cum_info_lag_%s + z_time_from_partner_onset_s:z_information_rate_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms, lag_ms)),
              list(model_id = "C2_offset_x_rate", formula = sprintf("event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + z_information_rate_lag_%s + z_prop_expected_cum_info_lag_%s + z_time_from_partner_offset_s:z_information_rate_lag_%s + (1 | dyad_id) + (1 | subject)", lag_ms, lag_ms, lag_ms))
            ),
            lag_index = lag_index,
            total_lags = total_lags
          )
        }

        model_diag <- do.call(rbind, model_diagnostic_rows)
        family_rankings <- do.call(rbind, family_rows)
        family_rankings$rank_by_BIC <- ave(family_rankings$BIC, family_rankings$family, FUN = function(x) rank(x, ties.method = "first"))
        family_rankings$rank_by_delta_logLik <- ave(-family_rankings$delta_logLik, family_rankings$family, FUN = function(x) rank(x, ties.method = "first"))
        family_rankings$criterion <- "delta_log_likelihood"

        build_reason <- function(rows) {
          reasons <- character()
          if (length(unique(rows$n)) > 1) reasons <- c(reasons, "varying n across lags")
          if (length(unique(rows$k)) > 1) reasons <- c(reasons, "varying k across lags")
          if (length(unique(rows$row_mask_signature)) > 1) reasons <- c(reasons, "different row masks")
          if (any(!rows$converged) || any(rows$singular_or_boundary)) reasons <- c(reasons, "convergence/boundary differences")
          if (length(reasons) == 0) reasons <- c("criterion-only disagreement")
          paste(unique(reasons), collapse = " | ")
        }

        family_summary_rows <- list()
        selector_rows <- list()
        for (family_label in c("A", "B", "C")) {
          rows <- family_rankings[family_rankings$family == family_label, , drop = FALSE]
          best_delta <- rows[order(-rows$delta_logLik, rows$lag_ms), , drop = FALSE][1, , drop = FALSE]
          best_bic <- rows[order(rows$BIC, rows$lag_ms), , drop = FALSE][1, , drop = FALSE]
          agree <- identical(as.integer(best_delta$lag_ms), as.integer(best_bic$lag_ms))
          family_summary_rows[[length(family_summary_rows) + 1]] <- data.frame(
            family = family_label,
            lag_selection_model_or_score = as.character(best_delta$selector_model_id),
            criterion = "delta_log_likelihood",
            best_lag_ms = as.integer(best_delta$lag_ms),
            best_lag_by_bic_ms = as.integer(best_bic$lag_ms),
            bic_delta_ll_agree = agree,
            disagreement_reason = if (agree) "" else build_reason(model_diag[model_diag$family == family_label, , drop = FALSE]),
            n = as.integer(best_delta$n),
            logLik = as.numeric(best_delta$logLik),
            BIC = as.numeric(best_delta$BIC),
            delta_logLik = as.numeric(best_delta$delta_logLik),
            stringsAsFactors = FALSE
          )
          selector_rows[[length(selector_rows) + 1]] <- data.frame(
            selector = paste0(family_label, "_family"),
            backend = "current R mixed backend",
            model_family_or_model = paste0(family_label, "-series"),
            criterion = "delta_log_likelihood",
            best_lag_ms = as.integer(best_delta$lag_ms),
            stringsAsFactors = FALSE
          )
        }
        family_summary <- do.call(rbind, family_summary_rows)

        old_style_rows <- family_rankings[family_rankings$family == "A", , drop = FALSE]
        old_style_best <- old_style_rows[order(old_style_rows$BIC, old_style_rows$lag_ms), , drop = FALSE][1, , drop = FALSE]
        selector_rows[[length(selector_rows) + 1]] <- data.frame(
          selector = "old_style_omnibus",
          backend = "current R mixed backend",
          model_family_or_model = "old full-information equivalent",
          criterion = "BIC",
          best_lag_ms = as.integer(old_style_best$lag_ms),
          stringsAsFactors = FALSE
        )
        selector_table <- do.call(rbind, selector_rows)

        a_rows <- family_rankings[family_rankings$family == "A", , drop = FALSE]
        a_selected <- a_rows[order(-a_rows$delta_logLik, a_rows$lag_ms), , drop = FALSE][1, , drop = FALSE]
        candidate_scores <- data.frame(
          candidate_lag_ms = as.integer(a_rows$lag_ms),
          log_likelihood_baseline = as.numeric(a_rows$logLik - a_rows$delta_logLik),
          log_likelihood_joint_information = as.numeric(a_rows$logLik),
          delta_log_likelihood = as.numeric(a_rows$delta_logLik),
          selected = as.integer(a_rows$lag_ms) == as.integer(a_selected$lag_ms),
          stringsAsFactors = FALSE
        )

        payload <- list(
          selected_lag_ms = as.integer(a_selected$lag_ms),
          selected_lag_family = "A",
          best_lag_A_ms = as.integer(family_summary[family_summary$family == "A", "best_lag_ms"]),
          best_lag_B_ms = as.integer(family_summary[family_summary$family == "B", "best_lag_ms"]),
          best_lag_C_ms = as.integer(family_summary[family_summary$family == "C", "best_lag_ms"]),
          family_lags = list(
            A = as.integer(family_summary[family_summary$family == "A", "best_lag_ms"]),
            B = as.integer(family_summary[family_summary$family == "B", "best_lag_ms"]),
            C = as.integer(family_summary[family_summary$family == "C", "best_lag_ms"])
          ),
          comparison_metric = "delta_log_likelihood",
          baseline_model_id = "A0_timing",
          selected_model_id = "A3_joint_information",
          global_old_style_omnibus_lag = list(
            selector = "old_style_omnibus",
            backend = "current R mixed backend",
            model_family_or_model = "old full-information equivalent",
            criterion = "BIC",
            best_lag_ms = as.integer(old_style_best$lag_ms)
          )
        )

        dir.create(dirname(spec$score_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(candidate_scores, spec$score_path, row.names = FALSE)
        dir.create(dirname(spec$selected_path), recursive = TRUE, showWarnings = FALSE)
        writeLines(jsonlite::toJSON(payload, auto_unbox = TRUE, pretty = TRUE), spec$selected_path)
        dir.create(dirname(spec$family_summary_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(family_summary, spec$family_summary_path, row.names = FALSE)
        dir.create(dirname(spec$family_rankings_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(family_rankings[, c("family", "lag_ms", "n", "k", "logLik", "BIC", "delta_logLik", "rank_by_BIC", "rank_by_delta_logLik", "selector_model_id", "row_mask_signature")], spec$family_rankings_path, row.names = FALSE)
        dir.create(dirname(spec$model_diagnostics_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(model_diag[, c("family", "model_id", "lag_ms", "n", "k", "logLik", "BIC", "delta_logLik", "converged", "singular_or_boundary", "warnings", "row_mask_signature")], spec$model_diagnostics_path, row.names = FALSE)
        dir.create(dirname(spec$selector_comparison_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(selector_table, spec$selector_comparison_path, row.names = FALSE)
        if (!is.null(spec$lag_sensitivity_path)) {
          lag_sensitivity <- if (length(lag_sensitivity_rows)) do.call(rbind, lag_sensitivity_rows) else data.frame(
            candidate_lag_ms = numeric(),
            term = character(),
            predictor = character(),
            estimate = numeric(),
            ci_low = numeric(),
            ci_high = numeric(),
            backend = character(),
            stringsAsFactors = FALSE
          )
          dir.create(dirname(spec$lag_sensitivity_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(lag_sensitivity, spec$lag_sensitivity_path, row.names = FALSE)
        }
        log_msg("[behavior hazard][R] Selected family-wise lags: A=", payload$best_lag_A_ms, " ms; B=", payload$best_lag_B_ms, " ms; C=", payload$best_lag_C_ms, " ms")
        """
    )


def _model_bundle_script() -> str:
    return textwrap.dedent(
        r"""
        suppressPackageStartupMessages({
          library(glmmTMB)
          library(jsonlite)
        })

        args <- commandArgs(trailingOnly = TRUE)
        fpp_csv <- args[[1]]
        pooled_csv <- args[[2]]
        spec_path <- args[[3]]
        spec <- jsonlite::fromJSON(spec_path, simplifyVector = FALSE)
        verbose_flag <- isTRUE(spec$verbose)
        log_msg <- function(...) {
          if (verbose_flag) {
            cat(paste0(..., "\n"))
            flush.console()
          }
        }

        fpp <- read.csv(fpp_csv, stringsAsFactors = FALSE)
        pooled <- read.csv(pooled_csv, stringsAsFactors = FALSE)
        log_msg("[behavior hazard][R] Loaded FPP rows: ", nrow(fpp))
        log_msg("[behavior hazard][R] Loaded pooled rows: ", nrow(pooled))
        for (table_name in c("fpp", "pooled")) {
          table <- get(table_name)
          table$subject <- as.factor(table$subject)
          table$dyad_id <- as.factor(table$dyad_id)
          if ("anchor_type" %in% names(table)) {
            if (all(c("SPP", "FPP") %in% unique(table$anchor_type))) {
              table$anchor_type <- factor(table$anchor_type, levels = c("SPP", "FPP"))
            } else {
              table$anchor_type <- as.factor(table$anchor_type)
            }
          }
          assign(table_name, table)
        }

        backend_id <- "r_glmmtmb_binomial_mixed"
        backend_note <- "Behavioral hazard models are fit through glmmTMB in R with mixed-effects random intercepts, using the restored timing baseline and within-anchor standardization behavior."

        fit_model <- function(formula_text, data) {
          warning_messages <- character()
          started_at <- proc.time()[["elapsed"]]
          fit <- withCallingHandlers(
            tryCatch(
              glmmTMB::glmmTMB(
                formula = as.formula(formula_text),
                data = data,
                family = stats::binomial(link = "logit"),
                REML = FALSE
              ),
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

        coefficient_rows <- list()
        model_rows <- list()
        fit_lookup <- list()
        warning_rows <- list()

        total_models <- length(spec$model_specs)
        model_index <- 0L
        for (model_spec in spec$model_specs) {
          model_index <- model_index + 1L
          dataset_name <- model_spec$dataset
          model_id <- model_spec$model_id
          public_model_id <- if (!is.null(model_spec$public_model_id)) model_spec$public_model_id else model_id
          family_label <- if (!is.null(model_spec$family)) model_spec$family else ""
          lag_ms_for_model <- if (!is.null(model_spec$lag_ms)) as.integer(model_spec$lag_ms) else NA_integer_
          group_name <- model_spec$group
          formula_text <- model_spec$formula
          data <- if (identical(dataset_name, "FPP")) fpp else pooled
          log_msg(
            "[behavior hazard][R] [model ", model_index, "/", total_models, "] start ",
            model_id, " on ", dataset_name, " (rows=", nrow(data), ", events=", sum(data$event, na.rm = TRUE), ")"
          )
          fit_result <- fit_model(formula_text, data)
          fit <- fit_result$fit
          if (inherits(fit, "error")) {
            log_msg(
              "[behavior hazard][R] [model ", model_index, "/", total_models, "] failed ",
              model_id, " after ", sprintf("%.1f", fit_result$elapsed_s), "s: ", conditionMessage(fit)
            )
            warning_rows[[length(warning_rows) + 1]] <- data.frame(
              dataset = dataset_name,
              model_id = model_id,
              warning = conditionMessage(fit),
              stringsAsFactors = FALSE
            )
            model_rows[[length(model_rows) + 1]] <- data.frame(
              group = group_name,
              dataset = dataset_name,
              model_id = model_id,
              public_model_id = public_model_id,
              family = family_label,
              lag_ms = lag_ms_for_model,
              formula = formula_text,
              n_rows = nrow(data),
              n_events = sum(data$event, na.rm = TRUE),
              log_likelihood = NA_real_,
              aic = NA_real_,
              bic = NA_real_,
              converged = FALSE,
              backend = backend_id,
              notes = backend_note,
              warnings = conditionMessage(fit),
              n_parameters = NA_integer_,
              stringsAsFactors = FALSE
            )
            next
          }
          log_msg(
            "[behavior hazard][R] [model ", model_index, "/", total_models, "] done ",
            model_id, " in ", sprintf("%.1f", fit_result$elapsed_s), "s"
          )
          fit_lookup[[model_id]] <- fit
          ll <- as.numeric(logLik(fit))
          aic <- AIC(fit)
          bic <- BIC(fit)
          n_parameters <- length(stats::coef(fit)$cond)
          converged <- isTRUE(fit$sdr$pdHess)
          warning_text <- paste(unique(fit_result$warnings), collapse = " | ")
          model_rows[[length(model_rows) + 1]] <- data.frame(
            group = group_name,
            dataset = dataset_name,
            model_id = model_id,
            public_model_id = public_model_id,
            family = family_label,
            lag_ms = lag_ms_for_model,
            formula = formula_text,
            n_rows = nrow(data),
            n_events = sum(data$event, na.rm = TRUE),
            log_likelihood = ll,
            aic = aic,
            bic = bic,
            converged = converged,
            backend = backend_id,
            notes = backend_note,
            warnings = warning_text,
            n_parameters = n_parameters,
            stringsAsFactors = FALSE
          )
          if (!converged) {
            warning_rows[[length(warning_rows) + 1]] <- data.frame(
              dataset = dataset_name,
              model_id = model_id,
              warning = "glmmTMB reported a non-positive-definite Hessian.",
              stringsAsFactors = FALSE
            )
          }
          if (length(fit_result$warnings) > 0) {
            for (message in unique(fit_result$warnings)) {
              warning_rows[[length(warning_rows) + 1]] <- data.frame(
                dataset = dataset_name,
                model_id = model_id,
                warning = message,
                stringsAsFactors = FALSE
              )
            }
          }
          coef_table <- as.data.frame(summary(fit)$coefficients$cond)
          coef_table$term <- rownames(coef_table)
          names(coef_table) <- c("estimate", "std_error", "z_value", "p_value", "term")
          coef_table$ci_low <- coef_table$estimate - 1.96 * coef_table$std_error
          coef_table$ci_high <- coef_table$estimate + 1.96 * coef_table$std_error
          coef_table$model_id <- model_id
          coef_table$public_model_id <- public_model_id
          coef_table$family <- family_label
          coef_table$dataset <- dataset_name
          coef_table$backend <- backend_id
          coef_table$selected_lag_ms <- lag_ms_for_model
          coef_table$notes <- backend_note
          coefficient_rows[[length(coefficient_rows) + 1]] <- coef_table[, c(
            "dataset", "family", "model_id", "public_model_id", "term", "estimate", "std_error", "z_value", "p_value",
            "ci_low", "ci_high", "backend", "selected_lag_ms", "notes"
          )]
        }

        model_metrics <- if (length(model_rows)) do.call(rbind, model_rows) else data.frame()
        coefficients <- if (length(coefficient_rows)) do.call(rbind, coefficient_rows) else data.frame()
        warnings_table <- if (length(warning_rows)) do.call(rbind, warning_rows) else data.frame(
          dataset = character(), model_id = character(), warning = character(), stringsAsFactors = FALSE
        )

        comparison_rows <- list()
        total_comparisons <- length(spec$comparison_specs)
        comparison_index <- 0L
        for (comparison_spec in spec$comparison_specs) {
          comparison_index <- comparison_index + 1L
          reduced_id <- comparison_spec$reduced
          full_id <- comparison_spec$full
          log_msg(
            "[behavior hazard][R] [comparison ", comparison_index, "/", total_comparisons, "] ",
            reduced_id, " vs ", full_id
          )
          if (is.null(fit_lookup[[reduced_id]]) || is.null(fit_lookup[[full_id]])) {
            next
          }
          reduced_fit <- fit_lookup[[reduced_id]]
          full_fit <- fit_lookup[[full_id]]
          ll_reduced <- as.numeric(logLik(reduced_fit))
          ll_full <- as.numeric(logLik(full_fit))
          npar_reduced <- attr(logLik(reduced_fit), "df")
          npar_full <- attr(logLik(full_fit), "df")
          df_diff <- as.numeric(npar_full - npar_reduced)
          lr_stat <- 2 * (ll_full - ll_reduced)
          p_value <- if (is.finite(df_diff) && df_diff > 0) stats::pchisq(lr_stat, df = df_diff, lower.tail = FALSE) else NA_real_
          comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
            dataset = comparison_spec$dataset,
            family = if (!is.null(comparison_spec$family)) comparison_spec$family else "",
            comparison_id = sprintf(
              "%s__vs__%s",
              if (!is.null(comparison_spec$public_reduced)) comparison_spec$public_reduced else reduced_id,
              if (!is.null(comparison_spec$public_full)) comparison_spec$public_full else full_id
            ),
            model_reduced = if (!is.null(comparison_spec$public_reduced)) comparison_spec$public_reduced else reduced_id,
            model_full = if (!is.null(comparison_spec$public_full)) comparison_spec$public_full else full_id,
            log_likelihood_reduced = ll_reduced,
            log_likelihood_full = ll_full,
            delta_log_likelihood = ll_full - ll_reduced,
            statistic = lr_stat,
            p_value = p_value,
            notes = backend_note,
            backend = backend_id,
            selected_lag_ms = if (!is.null(comparison_spec$lag_ms)) as.integer(comparison_spec$lag_ms) else NA_integer_,
            stringsAsFactors = FALSE
          )
        }
        comparisons <- if (length(comparison_rows)) do.call(rbind, comparison_rows) else data.frame()

        fit_for_predictions <- function(model_id) fit_lookup[[model_id]]
        base_row <- function(data, columns) {
          out <- list()
          for (column_name in columns) {
            out[[column_name]] <- stats::median(as.numeric(data[[column_name]]), na.rm = TRUE)
          }
          out
        }
        predict_frame <- function(fit, newdata) {
          pred <- predict(fit, newdata = newdata, type = "link", se.fit = TRUE, re.form = NA, allow.new.levels = TRUE)
          if (is.list(pred)) {
            eta <- as.numeric(pred$fit)
            se <- as.numeric(pred$se.fit)
          } else {
            eta <- as.numeric(pred)
            se <- rep(NA_real_, length(eta))
          }
          out <- newdata
          out$predicted_hazard <- stats::plogis(eta)
          out$ci_low <- ifelse(is.finite(se), stats::plogis(eta - 1.96 * se), NA_real_)
          out$ci_high <- ifelse(is.finite(se), stats::plogis(eta + 1.96 * se), NA_real_)
          out
        }

        first_level <- function(x) {
          if (is.factor(x)) {
            return(levels(x)[1])
          }
          unique_values <- unique(x)
          return(unique_values[[1]])
        }

        lag_ms_a <- as.integer(spec$selected_lags$A)
        lag_ms_b <- as.integer(spec$selected_lags$B)
        lag_ms_c <- as.integer(spec$selected_lags$C)

        conditional_median_curve <- function(data, x_col, y_col, x_values) {
          curve <- stats::aggregate(
            data[[y_col]],
            by = list(x = data[[x_col]]),
            FUN = stats::median,
            na.rm = TRUE
          )
          curve <- curve[stats::complete.cases(curve), , drop = FALSE]
          if (!nrow(curve)) {
            return(rep(0, length(x_values)))
          }
          curve <- curve[order(curve$x), , drop = FALSE]
          if (nrow(curve) == 1L) {
            return(rep(curve$x.1[[1]], length(x_values)))
          }
          stats::approx(
            x = curve$x,
            y = curve$x.1,
            xout = x_values,
            rule = 2
          )$y
        }

        if (!is.null(spec$figure_prediction_path) && !is.null(fit_lookup[["A3_joint_information"]]) && !is.null(fit_lookup[["B2_anchor_x_information"]])) {
          figure2_started <- proc.time()[["elapsed"]]
          log_msg("[behavior hazard][R] Building Figure 2 prediction table")
          rate_col <- sprintf("z_information_rate_lag_%s", lag_ms_a)
          prop_col <- sprintf("z_prop_expected_cum_info_lag_%s", lag_ms_a)
          rate_col_b <- sprintf("z_information_rate_lag_%s", lag_ms_b)
          prop_col_b <- sprintf("z_prop_expected_cum_info_lag_%s", lag_ms_b)
          timing_cols <- c("z_time_from_partner_onset_s", "z_time_from_partner_offset_s", "z_time_from_partner_offset_s_squared")
          fpp_base <- base_row(fpp, timing_cols)
          rate_raw <- seq(min(fpp$information_rate, na.rm = TRUE), max(fpp$information_rate, na.rm = TRUE), length.out = 40)
          rate_z <- seq(min(fpp[[rate_col]], na.rm = TRUE), max(fpp[[rate_col]], na.rm = TRUE), length.out = 40)
          prop_raw <- seq(min(fpp$prop_expected_cum_info, na.rm = TRUE), max(fpp$prop_expected_cum_info, na.rm = TRUE), length.out = 40)
          prop_z <- seq(min(fpp[[prop_col]], na.rm = TRUE), max(fpp[[prop_col]], na.rm = TRUE), length.out = 40)
          figure_rows <- list()
          panel_a <- data.frame(
            dyad_id = factor(rep(first_level(fpp$dyad_id), length(rate_raw)), levels = levels(fpp$dyad_id)),
            subject = factor(rep(first_level(fpp$subject), length(rate_raw)), levels = levels(fpp$subject)),
            information_rate = rate_raw,
            prop_expected_cum_info = rep(stats::median(prop_raw, na.rm = TRUE), length(rate_raw)),
            x_value_z = rate_z,
            x_value_original = rate_raw,
            stringsAsFactors = FALSE
          )
          for (name in names(fpp_base)) {
            panel_a[[name]] <- rep(fpp_base[[name]], nrow(panel_a))
          }
          panel_a[[rate_col]] <- rate_z
          panel_a[[prop_col]] <- rep(0, nrow(panel_a))
          pred_a <- predict_frame(fit_lookup[["A3_joint_information"]], panel_a)
          pred_a$figure <- "fig02_primary_information_effects"
          pred_a$panel <- "A"
          pred_a$anchor_type <- "FPP"
          pred_a$predictor <- "information_rate"
          figure_rows[[length(figure_rows) + 1]] <- pred_a

          panel_b <- data.frame(
            dyad_id = factor(rep(first_level(fpp$dyad_id), length(prop_raw)), levels = levels(fpp$dyad_id)),
            subject = factor(rep(first_level(fpp$subject), length(prop_raw)), levels = levels(fpp$subject)),
            information_rate = rep(stats::median(rate_raw, na.rm = TRUE), length(prop_raw)),
            prop_expected_cum_info = prop_raw,
            x_value_z = prop_z,
            x_value_original = prop_raw,
            stringsAsFactors = FALSE
          )
          for (name in names(fpp_base)) {
            panel_b[[name]] <- rep(fpp_base[[name]], nrow(panel_b))
          }
          panel_b[[rate_col]] <- rep(0, nrow(panel_b))
          panel_b[[prop_col]] <- prop_z
          pred_b <- predict_frame(fit_lookup[["A3_joint_information"]], panel_b)
          pred_b$figure <- "fig02_primary_information_effects"
          pred_b$panel <- "B"
          pred_b$anchor_type <- "FPP"
          pred_b$predictor <- "prop_expected_cum_info"
          figure_rows[[length(figure_rows) + 1]] <- pred_b

          pooled_base <- base_row(pooled, timing_cols)
          pooled_base[[prop_col_b]] <- stats::median(as.numeric(pooled[[prop_col_b]]), na.rm = TRUE)
          for (anchor in c("SPP", "FPP")) {
            panel_c <- data.frame(
              dyad_id = factor(rep(first_level(pooled$dyad_id), length(rate_raw)), levels = levels(pooled$dyad_id)),
              subject = factor(rep(first_level(pooled$subject), length(rate_raw)), levels = levels(pooled$subject)),
              anchor_type = factor(rep(anchor, length(rate_raw)), levels = levels(pooled$anchor_type)),
              information_rate = rate_raw,
              prop_expected_cum_info = rep(stats::median(prop_raw, na.rm = TRUE), length(rate_raw)),
              x_value_z = rate_z,
              x_value_original = rate_raw,
              stringsAsFactors = FALSE
            )
            for (name in names(pooled_base)) {
              panel_c[[name]] <- rep(pooled_base[[name]], nrow(panel_c))
            }
            panel_c[[rate_col_b]] <- rate_z
            panel_c[[prop_col_b]] <- rep(pooled_base[[prop_col_b]], nrow(panel_c))
            pred_c <- predict_frame(fit_lookup[["B2_anchor_x_information"]], panel_c)
            pred_c$figure <- "fig02_primary_information_effects"
            pred_c$panel <- "C"
            pred_c$anchor_type <- anchor
            pred_c$predictor <- "information_rate"
            figure_rows[[length(figure_rows) + 1]] <- pred_c
          }
          figure_predictions <- do.call(rbind, figure_rows)
          dir.create(dirname(spec$figure_prediction_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(figure_predictions[, c("figure", "panel", "anchor_type", "predictor", "x_value_z", "x_value_original", "predicted_hazard", "ci_low", "ci_high")], spec$figure_prediction_path, row.names = FALSE)
          log_msg("[behavior hazard][R] Figure 2 prediction table done in ", sprintf("%.1f", proc.time()[["elapsed"]] - figure2_started), "s")
        }

        if (!is.null(spec$timing_heatmap_path) && !is.null(fit_lookup[["C1_onset_x_rate"]]) && !is.null(fit_lookup[["C2_offset_x_rate"]])) {
          figure3_started <- proc.time()[["elapsed"]]
          log_msg("[behavior hazard][R] Building Figure 3 timing heatmap table")
          rate_col <- sprintf("z_information_rate_lag_%s", lag_ms_c)
          prop_col <- sprintf("z_prop_expected_cum_info_lag_%s", lag_ms_c)
          onset_values <- seq(min(fpp$time_from_partner_onset_s, na.rm = TRUE), max(fpp$time_from_partner_onset_s, na.rm = TRUE), length.out = 25)
          offset_values <- seq(min(fpp$time_from_partner_offset_s, na.rm = TRUE), max(fpp$time_from_partner_offset_s, na.rm = TRUE), length.out = 25)
          rate_raw <- seq(min(fpp$information_rate, na.rm = TRUE), max(fpp$information_rate, na.rm = TRUE), length.out = 25)
          rate_z <- seq(min(fpp[[rate_col]], na.rm = TRUE), max(fpp[[rate_col]], na.rm = TRUE), length.out = 25)
          onset_mean <- mean(fpp$time_from_partner_onset_s, na.rm = TRUE)
          onset_sd <- stats::sd(fpp$time_from_partner_onset_s, na.rm = TRUE)
          offset_mean <- mean(fpp$time_from_partner_offset_s, na.rm = TRUE)
          offset_sd <- stats::sd(fpp$time_from_partner_offset_s, na.rm = TRUE)
          onset_sd <- ifelse(is.finite(onset_sd) && onset_sd > 0, onset_sd, 1)
          offset_sd <- ifelse(is.finite(offset_sd) && offset_sd > 0, offset_sd, 1)
          conditioned_offsets <- conditional_median_curve(fpp, "time_from_partner_onset_s", "time_from_partner_offset_s", onset_values)
          conditioned_onsets <- conditional_median_curve(fpp, "time_from_partner_offset_s", "time_from_partner_onset_s", offset_values)
          prop_median <- stats::median(as.numeric(fpp[[prop_col]]), na.rm = TRUE)
          heat_rows <- list()
          onset_grid <- expand.grid(
            time_value_s = onset_values,
            rate_idx = seq_along(rate_raw),
            KEEP.OUT.ATTRS = FALSE,
            stringsAsFactors = FALSE
          )
          onset_grid$dyad_id <- factor(rep(first_level(fpp$dyad_id), nrow(onset_grid)), levels = levels(fpp$dyad_id))
          onset_grid$subject <- factor(rep(first_level(fpp$subject), nrow(onset_grid)), levels = levels(fpp$subject))
          onset_grid$z_time_from_partner_onset_s <- (onset_grid$time_value_s - onset_mean) / onset_sd
          onset_grid$time_from_partner_onset_s <- onset_grid$time_value_s
          onset_grid$time_from_partner_offset_s <- conditioned_offsets[match(onset_grid$time_value_s, onset_values)]
          onset_grid$z_time_from_partner_offset_s <- (onset_grid$time_from_partner_offset_s - offset_mean) / offset_sd
          onset_grid$z_time_from_partner_offset_s_squared <- onset_grid$z_time_from_partner_offset_s ^ 2
          onset_grid$information_rate_z <- rate_z[onset_grid$rate_idx]
          onset_grid$information_rate_original <- rate_raw[onset_grid$rate_idx]
          onset_grid[[rate_col]] <- onset_grid$information_rate_z
          onset_grid[[prop_col]] <- rep(prop_median, nrow(onset_grid))
          pred_onset <- predict_frame(fit_lookup[["C1_onset_x_rate"]], onset_grid)
          pred_onset$panel <- "A"
          pred_onset$timing_reference <- "partner_onset"
          heat_rows[[length(heat_rows) + 1]] <- pred_onset

          offset_grid <- expand.grid(
            time_value_s = offset_values,
            rate_idx = seq_along(rate_raw),
            KEEP.OUT.ATTRS = FALSE,
            stringsAsFactors = FALSE
          )
          offset_grid$dyad_id <- factor(rep(first_level(fpp$dyad_id), nrow(offset_grid)), levels = levels(fpp$dyad_id))
          offset_grid$subject <- factor(rep(first_level(fpp$subject), nrow(offset_grid)), levels = levels(fpp$subject))
          offset_grid$time_from_partner_onset_s <- conditioned_onsets[match(offset_grid$time_value_s, offset_values)]
          offset_grid$z_time_from_partner_onset_s <- (offset_grid$time_from_partner_onset_s - onset_mean) / onset_sd
          offset_grid$time_from_partner_offset_s <- offset_grid$time_value_s
          offset_grid$z_time_from_partner_offset_s <- (offset_grid$time_value_s - offset_mean) / offset_sd
          offset_grid$z_time_from_partner_offset_s_squared <- offset_grid$z_time_from_partner_offset_s ^ 2
          offset_grid$information_rate_z <- rate_z[offset_grid$rate_idx]
          offset_grid$information_rate_original <- rate_raw[offset_grid$rate_idx]
          offset_grid[[rate_col]] <- offset_grid$information_rate_z
          offset_grid[[prop_col]] <- rep(prop_median, nrow(offset_grid))
          pred_offset <- predict_frame(fit_lookup[["C2_offset_x_rate"]], offset_grid)
          pred_offset$panel <- "B"
          pred_offset$timing_reference <- "partner_offset"
          heat_rows[[length(heat_rows) + 1]] <- pred_offset

          heatmap_predictions <- do.call(rbind, heat_rows)
          dir.create(dirname(spec$timing_heatmap_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(heatmap_predictions[, c("panel", "timing_reference", "time_value_s", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high")], spec$timing_heatmap_path, row.names = FALSE)
          log_msg("[behavior hazard][R] Figure 3 timing heatmap table done in ", sprintf("%.1f", proc.time()[["elapsed"]] - figure3_started), "s")
        }

        if (!is.null(spec$three_way_path)) {
          log_msg("[behavior hazard][R] Writing exploratory three-way table placeholder")
          empty_three_way <- data.frame(
            anchor_type = character(),
            timing_reference = character(),
            time_value_s = numeric(),
            information_rate_z = numeric(),
            information_rate_original = numeric(),
            predicted_hazard = numeric(),
            ci_low = numeric(),
            ci_high = numeric(),
            stringsAsFactors = FALSE
          )
          dir.create(dirname(spec$three_way_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(empty_three_way, spec$three_way_path, row.names = FALSE)
        }

        lag_sensitivity <- data.frame()
        if (!is.null(spec$lag_sensitivity_path)) {
          lag_rows <- list()
          total_lag_refits <- length(spec$candidate_lags_ms)
          lag_refit_index <- 0L
          for (lag_ms in spec$candidate_lags_ms) {
            lag_refit_index <- lag_refit_index + 1L
            log_msg("[behavior hazard][R] [lag sensitivity ", lag_refit_index, "/", total_lag_refits, "] refit: ", lag_ms, " ms")
            rate_col <- sprintf("z_information_rate_lag_%s", lag_ms)
            prop_col <- sprintf("z_prop_expected_cum_info_lag_%s", lag_ms)
            formula_text <- sprintf(
              "event ~ z_time_from_partner_onset_s + z_time_from_partner_offset_s + z_time_from_partner_offset_s_squared + %s + %s + (1 | dyad_id) + (1 | subject)",
              rate_col,
              prop_col
            )
            fit_result <- fit_model(formula_text, fpp)
            fit <- fit_result$fit
            if (inherits(fit, "error")) next
            log_msg(
              "[behavior hazard][R] [lag sensitivity ", lag_refit_index, "/", total_lag_refits, "] done: ",
              lag_ms, " ms in ", sprintf("%.1f", fit_result$elapsed_s), "s"
            )
            coef_table <- as.data.frame(summary(fit)$coefficients$cond)
            coef_table$term <- rownames(coef_table)
            names(coef_table) <- c("estimate", "std_error", "z_value", "p_value", "term")
            for (term_name in c(rate_col, prop_col)) {
              row <- coef_table[coef_table$term == term_name, , drop = FALSE]
              if (nrow(row) == 0) next
              lag_rows[[length(lag_rows) + 1]] <- data.frame(
                candidate_lag_ms = as.integer(lag_ms),
                term = term_name,
                predictor = ifelse(grepl("information_rate", term_name), "information_rate", "prop_expected_cum_info"),
                estimate = row$estimate[[1]],
                ci_low = row$estimate[[1]] - 1.96 * row$std_error[[1]],
                ci_high = row$estimate[[1]] + 1.96 * row$std_error[[1]],
                backend = backend_id,
                stringsAsFactors = FALSE
              )
            }
          }
          lag_sensitivity <- if (length(lag_rows)) do.call(rbind, lag_rows) else data.frame()
        }

        log_msg("[behavior hazard][R] Writing model metrics, coefficients, comparisons, diagnostics, and prediction tables")
        dir.create(dirname(spec$model_metrics_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(model_metrics, spec$model_metrics_path, row.names = FALSE)
        dir.create(dirname(spec$coefficient_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(coefficients, spec$coefficient_path, row.names = FALSE)
        dir.create(dirname(spec$comparison_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(comparisons, spec$comparison_path, row.names = FALSE)
        dir.create(dirname(spec$convergence_path), recursive = TRUE, showWarnings = FALSE)
        write.csv(warnings_table, spec$convergence_path, row.names = FALSE)
        if (!is.null(spec$lag_sensitivity_path)) {
          dir.create(dirname(spec$lag_sensitivity_path), recursive = TRUE, showWarnings = FALSE)
          write.csv(lag_sensitivity, spec$lag_sensitivity_path, row.names = FALSE)
        }
        log_msg("[behavior hazard][R] Model bundle complete")
        """
    )
