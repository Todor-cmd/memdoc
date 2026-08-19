# Generate a structured split experimental design for the MemDoc study.
#
# Two-layer structure:
#   Layer 1 (D-optimal, blocked by question):
#     dist  (3 levels) × agent (4 levels) = 12 candidate cells.
#     Each question block receives `k` of these 12 cells, chosen by D-optimal
#     search to maximise information for ~ dist * agent.
#
#   Layer 2 (deterministic, fully crossed):
#     persona (3 levels) is fully crossed with every dist × agent cell.
#     For each question, every persona sees the IDENTICAL set of dist × agent
#     conditions — the only thing that varies is which memory corpus is loaded.
#
# This encodes the priority hierarchy: dist (primary RQ) and dist:agent (RQ1b)
# get all D-optimal attention; persona (moderator) is guaranteed within-question
# by construction, giving maximally precise persona comparisons at no optimiser
# cost. Flexibility in a flat design would be undirected — the D criterion treats
# all parameters equally and could sacrifice dist coverage for persona balance.
#
# Blocking factor (RANDOM effect, NOT in the design formula):
#   question – each sampled question is a block (random intercept). Keeps
#   question_type and hop_count estimable as fixed covariates (RQ2b).
#
# Design model (layer 1): ~ dist * agent    (11 parameters)
# Full model  (analysis): ~ persona * dist + dist * agent + question_type
#                          + hop_count + (1 | question)
#
# Runs per question = k × 3 (personas).  Total runs = n_questions × k × 3.
#
# Usage:
#   Rscript prepare_data/optimal_design.R
#   Rscript prepare_data/optimal_design.R --n-questions 80 --k 6
#   Rscript prepare_data/optimal_design.R --nsim 500 --effect-low 0.55 --effect-high 0.8
#   Rscript prepare_data/optimal_design.R --skip-power

library(skpr)

# ── Console progress (works under Rscript) ────────────────────────────────────
make_skpr_progress <- function(label) {
  cumulative <- 0
  function(...) {
    args <- list(...)
    if (length(args) >= 2L && all(vapply(args[1:2], is.numeric, logical(1)))) {
      i <- as.integer(args[[1]])
      n <- as.integer(args[[2]])
      pct <- round(100 * i / max(n, 1L))
      cat(sprintf("\r[%s] %s: %d / %d (%d%%)   ",
                  format(Sys.time(), "%H:%M:%S"), label, i, n, pct))
    } else {
      x <- suppressWarnings(as.numeric(args[[1]][1]))
      if (!is.finite(x)) return(invisible(NULL))
      cumulative <<- min(1, cumulative + x)
      cat(sprintf("\r[%s] %s: ~%d%%   ",
                  format(Sys.time(), "%H:%M:%S"), label,
                  round(100 * cumulative)))
    }
    flush.console()
    invisible(NULL)
  }
}

# ── CLI args ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  return(args[idx + 1])
}

n_questions <- as.integer(get_arg("--n-questions", "210"))
k           <- as.integer(get_arg("--k", "4"))               # dist×agent cells per question
n_repeats   <- as.integer(get_arg("--repeats", "20"))
out_path    <- get_arg("--out", "data/experiment_design.csv")
alpha       <- as.numeric(get_arg("--alpha", "0.05"))

effect_low  <- as.numeric(get_arg("--effect-low", "0.6"))
effect_high <- as.numeric(get_arg("--effect-high", "0.8"))
block_var   <- as.numeric(get_arg("--block-variance", "1"))
n_sim       <- as.integer(get_arg("--nsim", "1000"))
skip_power  <- "--skip-power" %in% args

personas      <- paste0("persona_", 1:3)
n_personas    <- length(personas)
runs_per_q    <- k * n_personas
n_trials_da   <- n_questions * k            # rows in the dist×agent layer
n_trials_full <- n_questions * runs_per_q   # rows after persona cross-join

progress_opts <- function(label) {
  list(progressBarUpdater = make_skpr_progress(label))
}

# ── Layer 1: dist × agent candidate set (12 cells) ───────────────────────────
cand_da <- expand.grid(
  agent = factor(paste0("agent_", 1:4)),
  dist  = factor(c("document_only", "memory_only", "integrated"))
)

cat(sprintf("dist × agent factorial: %d cells\n", nrow(cand_da)))
cat(sprintf("Question blocks: %d  ×  k = %d cells/question  ×  %d personas  =  %d total runs\n",
            n_questions, k, n_personas, n_trials_full))
n_full <- 12 * n_questions * n_personas
cat(sprintf("Full crossing (12 × %d × %d): %d runs\n",
            n_questions, n_personas, n_full))
cat(sprintf("Savings: %d runs avoided (%.0f%% reduction)\n",
            n_full - n_trials_full,
            100 * (1 - n_trials_full / n_full)))

# ── Layer 1: D-optimal on dist × agent, blocked by question ──────────────────
da_formula <- ~ dist * agent

set.seed(2026)
cat(sprintf("\n[%s] gen_design (dist × agent): %d runs, %d blocks of %d (%d repeats)...\n",
            format(Sys.time(), "%H:%M:%S"), n_trials_da, n_questions, k, n_repeats))
flush.console()
t0 <- Sys.time()
design_da <- gen_design(
  candidateset         = cand_da,
  model                = da_formula,
  trials               = n_trials_da,
  blocksizes           = k,
  add_blocking_columns = TRUE,
  optimality           = "D",
  repeats              = n_repeats,
  progress             = TRUE,
  advancedoptions      = progress_opts("gen_design")
)
cat("\n")
cat(sprintf("[%s] gen_design done in %.1f s\n",
            format(Sys.time(), "%H:%M:%S"),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

cat("\n── dist × agent design summary ──\n")
print(summary(design_da))

# ── Layer 2: cross-join with persona (deterministic, fully within-question) ───
persona_df <- data.frame(persona = factor(personas))
design <- merge(design_da, persona_df, all = TRUE)
design <- design[order(design$Block1, design$persona, design$dist, design$agent), ]
rownames(design) <- NULL

cat(sprintf("\n── Full design: %d rows (%d questions × %d dist·agent cells × %d personas) ──\n",
            nrow(design), n_questions, k, n_personas))
print(summary(design))

# ── Power evaluation (Monte Carlo, binomial, blocked) ─────────────────────────
# The full model includes persona * dist (moderator interaction, RQ2d) plus
# dist * agent (primary interaction, RQ1b). Power is evaluated on the cross-
# joined design with blocking = TRUE (question as random intercept).
full_formula <- ~ persona * dist + dist * agent

if (skip_power) {
  cat("\n--skip-power: skipping eval_design_mc (design CSV still written below)\n")
} else {
  cat(sprintf("\n[%s] eval_design_mc: %d binomial glmer simulations...\n",
              format(Sys.time(), "%H:%M:%S"), n_sim))
  flush.console()
  t1 <- Sys.time()
  power <- eval_design_mc(
    design           = design,
    model            = full_formula,
    alpha            = alpha,
    blocking         = TRUE,
    glmfamily        = "binomial",
    effectsize       = c(effect_low, effect_high),
    varianceratios   = block_var,
    nsim             = n_sim,
    detailedoutput   = TRUE,
    progress         = TRUE,
    advancedoptions  = progress_opts("eval_design_mc")
  )
  cat("\n")
  cat(sprintf("[%s] eval_design_mc done in %.1f s\n",
              format(Sys.time(), "%H:%M:%S"),
              as.numeric(difftime(Sys.time(), t1, units = "secs"))))
  print(power)

  # Save power results to CSV alongside the design
  power_path <- sub("\\.csv$", "_power.csv", out_path)
  write.csv(power, power_path, row.names = FALSE)
  cat(sprintf("Power analysis written to %s\n", power_path))
}

# ── Export ─────────────────────────────────────────────────────────────────────
# Block1 = question block id (map to question_id downstream).
# Each question has k × 3 rows: k dist×agent cells × 3 personas.
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.csv(design, out_path, row.names = FALSE)
cat(sprintf("\nDesign written to %s (%d rows, %d question blocks)\n",
            out_path, nrow(design), n_questions))
