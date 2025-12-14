
# tests/testthat/test-crossfit.R


# tests/testthat/test-crossfit-focus.R
# Focused tests for: FE.glm, RE.glm, RE.glm.rs, SuperLearner branches (fix.mlr, cwc.FE)
# These are designed to be fast + robust on CI, and to skip gracefully if optional pkgs missing.

# -------------------------
# Helpers
# -------------------------

.make_cluster_features <- function(df, yname, xnames, Sname) {
  # Adds *_clmean and *_cwc columns for each x in xnames, plus y_clmean and y_cwc
  for (x in xnames) {
    df[[paste0(x, "_clmean")]] <- ave(df[[x]], df[[Sname]], FUN = mean)
    df[[paste0(x, "_cwc")]] <- df[[x]] - df[[paste0(x, "_clmean")]]
  }
  df[[paste0(yname, "_clmean")]] <- ave(df[[yname]], df[[Sname]], FUN = mean)
  df[[paste0(yname, "_cwc")]] <- df[[yname]] - df[[paste0(yname, "_clmean")]]
  df
}

.make_cluster_dummies <- function(df, Sname, prefix = "Sd") {
  # Creates one dummy per observed cluster level in df[[Sname]]
  levs <- sort(unique(df[[Sname]]))
  dnames <- paste0(prefix, levs)
  for (k in seq_along(levs)) {
    df[[dnames[k]]] <- as.integer(df[[Sname]] == levs[k])
  }
  list(df = df, dnames = dnames, levs = levs)
}

.expect_preds_ok <- function(out, n, bounded = FALSE) {
  expect_type(out, "list")
  expect_true(all(c("fit", "preds") %in% names(out)))
  expect_true(is.numeric(out$preds))
  expect_length(out$preds, n)
  expect_true(all(is.finite(out$preds)))
  if (bounded) expect_true(all(out$preds >= 0 & out$preds <= 1))
}

# -------------------------
# FE.glm
# -------------------------
testthat::test_that("crossfit FE.glm (binomial) returns bounded predictions and uses cluster FE", {
  set.seed(101)
  n_tr <- 180
  n_va <- 80
  J <- 8

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.45),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.45),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "FE.glm",
    type = "binomial",
    learners = NULL,
    bounded = TRUE
  )

  .expect_preds_ok(out, n_va, bounded = TRUE)

  # sanity: fit is glm with S in formula
  expect_true(inherits(out$fit, "glm"))
  expect_true(grepl("\\bS\\b", deparse(stats::formula(out$fit))))
})

# testthat::test_that("crossfit FE.glm respects ipw weights (does not error; finite preds)", {
#   set.seed(102)
#   n_tr <- 150
#   n_va <- 60
#   J <- 6
#
#   train <- data.frame(
#     Y = rbinom(n_tr, 1, 0.4),
#     X1 = rnorm(n_tr),
#     X2 = rnorm(n_tr),
#     S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
#   )
#   valid <- data.frame(
#     Y = rbinom(n_va, 1, 0.4),
#     X1 = rnorm(n_va),
#     X2 = rnorm(n_va),
#     S  = factor(sample(seq_len(J), n_va, replace = TRUE))
#   )
#
#   ipw <- runif(n_tr, 0.5, 2)
#
#   out <- crossfit(
#     train = train,
#     valid.list = list(valid),
#     yname = "Y",
#     xnames = c("X1", "X2"),
#     varnames = list(S = "S", Sdumm = NULL, W = NULL),
#     ipw = ipw,
#     cluster_opt = "FE.glm",
#     type = "binomial",
#     learners = NULL,
#     bounded = TRUE
#   )
#
#   .expect_preds_ok(out, n_va, bounded = TRUE)
# })

# -------------------------
# RE.glm and RE.glm.rs
# -------------------------

testthat::test_that("crossfit RE.glm (binomial) runs and returns bounded predictions", {
  # testthat::skip_if_not_installed("lme4")

  set.seed(201)
  n_tr <- 220
  n_va <- 90
  J <- 10

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.35),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.35),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "RE.glm",
    type = "binomial",
    learners = NULL,
    bounded = TRUE
  )

  .expect_preds_ok(out, n_va, bounded = TRUE)
  expect_true(inherits(out$fit, c("glmerMod", "merMod")))
})

testthat::test_that("crossfit RE.glm.rs warns and falls back when random_slope_vars missing", {
  # testthat::skip_if_not_installed("lme4")

  set.seed(202)
  n_tr <- 200
  n_va <- 70
  J <- 9

  train <- data.frame(
    Y = rnorm(n_tr),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rnorm(n_va),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  expect_warning(
    out <- crossfit(
      train = train,
      valid.list = list(valid),
      yname = "Y",
      xnames = c("X1", "X2"),
      varnames = list(S = "S", Sdumm = NULL, W = NULL),
      cluster_opt = "RE.glm.rs",
      type = "gaussian",
      learners = NULL,
      bounded = FALSE,
      random_slope_vars = NULL,
      source_label = "test"
    ),
    regexp = "no random_slope_vars|random intercept only",
    ignore.case = TRUE
  )

  .expect_preds_ok(out, n_va, bounded = FALSE)
})

testthat::test_that("crossfit RE.glm.rs fits random slopes when provided (gaussian)", {
  # testthat::skip_if_not_installed("lme4")

  set.seed(203)
  n_tr <- 240
  n_va <- 80
  J <- 12

  train <- data.frame(
    Y = rnorm(n_tr),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rnorm(n_va),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "RE.glm.rs",
    type = "gaussian",
    learners = NULL,
    bounded = FALSE,
    random_slope_vars = c("X1")  # random slope for X1
  )

  .expect_preds_ok(out, n_va, bounded = FALSE)
  expect_true(inherits(out$fit, "merMod"))

  # sanity: random-effects term contains X1 (very light check)
  ftxt <- paste(deparse(stats::formula(out$fit)), collapse = " ")
  expect_true(grepl("X1", ftxt))
})

# -------------------------
# SuperLearner-focused tests
# -------------------------

# testthat::test_that("crossfit fix.mlr produces bounded predictions using SL.mean (fast)", {
#   # testthat::skip_if_not_installed("SuperLearner")
#
#   set.seed(301)
#   n_tr <- 160
#   n_va <- 70
#   J <- 5
#
#   train <- data.frame(
#     Y = rbinom(n_tr, 1, 0.5),
#     X1 = rnorm(n_tr),
#     X2 = rnorm(n_tr),
#     S  = sample(seq_len(J), n_tr, replace = TRUE)
#   )
#   valid <- data.frame(
#     Y = rbinom(n_va, 1, 0.5),
#     X1 = rnorm(n_va),
#     X2 = rnorm(n_va),
#     S  = sample(seq_len(J), n_va, replace = TRUE)
#   )
#
#   # Create matching dummies in train and valid
#   trD <- .make_cluster_dummies(train, "S", prefix = "Sd")
#   vaD <- .make_cluster_dummies(valid, "S", prefix = "Sd")
#   train <- trD$df
#   valid <- vaD$df
#   dnames <- trD$dnames
#
#   out <- crossfit(
#     train = train,
#     valid.list = list(valid),
#     yname = "Y",
#     xnames = c("X1", "X2"),
#     varnames = list(S = "S", Sdumm = dnames, W = NULL),
#     cluster_opt = "fix.mlr",
#     type = "binomial",
#     learners = c("SL.mean"),
#     bounded = TRUE
#   )
#
#   .expect_preds_ok(out, n_va, bounded = TRUE)
#   expect_true(inherits(out$fit, "SuperLearner"))
# })

testthat::test_that("crossfit cwc.FE produces bounded predictions (requires derived cols + dummies)", {
  # testthat::skip_if_not_installed("SuperLearner")

  set.seed(302)
  n_tr <- 180
  n_va <- 80
  J <- 6

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.45),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = sample(seq_len(J), n_tr, replace = TRUE)
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.45),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = sample(seq_len(J), n_va, replace = TRUE)
  )

  # Add cluster mean + cwc columns needed by cwc.FE (and y_clmean used in the code)
  train <- .make_cluster_features(train, "Y", c("X1", "X2"), "S")
  valid <- .make_cluster_features(valid, "Y", c("X1", "X2"), "S")

  # Add cluster dummies needed by cwc.FE
  trD <- .make_cluster_dummies(train, "S", prefix = "Sd")
  vaD <- .make_cluster_dummies(valid, "S", prefix = "Sd")
  train <- trD$df
  valid <- vaD$df
  dnames <- trD$dnames

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = dnames, W = NULL),
    cluster_opt = "cwc.FE",
    type = "binomial",
    learners = c("SL.mean"),
    bounded = TRUE
  )

  .expect_preds_ok(out, n_va, bounded = TRUE)
  expect_true(inherits(out$fit, "SuperLearner"))
})

testthat::test_that("crossfit cwc.FE errors if required dummies are missing", {
  # testthat::skip_if_not_installed("SuperLearner")

  set.seed(303)
  n_tr <- 120
  n_va <- 50
  J <- 5

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.4),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = sample(seq_len(J), n_tr, replace = TRUE)
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.4),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = sample(seq_len(J), n_va, replace = TRUE)
  )

  train <- .make_cluster_features(train, "Y", c("X1", "X2"), "S")
  valid <- .make_cluster_features(valid, "Y", c("X1", "X2"), "S")

  # Sdumm not provided -> should fail when trying train[, Sname_dummies] / selecting missing cols
  expect_error(
    crossfit(
      train = train,
      valid.list = list(valid),
      yname = "Y",
      xnames = c("X1", "X2"),
      varnames = list(S = "S", Sdumm = NULL, W = NULL),
      cluster_opt = "cwc.FE",
      type = "binomial",
      learners = c("SL.mean"),
      bounded = TRUE
    ),
    regexp = "cluster_opt = 'cwc\\.FE' requires varnames\\$Sdumm \\(cluster dummy column names\\)\\.",
    ignore.case = TRUE
  )
})






testthat::test_that("crossfit FE.glm returns bounded predictions of correct length", {
  # skip_if_not_installed("stats")

  set.seed(1)
  n_tr <- 120
  n_va <- 60
  J <- 6

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.4),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = sample(seq_len(J), n_tr, replace = TRUE)
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.4),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = sample(seq_len(J), n_va, replace = TRUE)
  )

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "FE.glm",
    type = "binomial",
    learners = NULL,
    bounded = TRUE
  )

  testthat::expect_type(out, "list")
  testthat::expect_true(all(c("fit", "preds") %in% names(out)))
  testthat::expect_length(out$preds, n_va)
  testthat::expect_true(all(is.finite(out$preds)))
  testthat::expect_true(all(out$preds >= 0 & out$preds <= 1))
})

# testthat::test_that("crossfit accepts ipw and produces finite predictions", {
#   set.seed(3)
#   n_tr <- 80
#   n_va <- 40
#   J <- 4
#
#   train <- data.frame(
#     Y = rbinom(n_tr, 1, 0.5),
#     X1 = rnorm(n_tr),
#     X2 = rnorm(n_tr),
#     S  = sample(seq_len(J), n_tr, replace = TRUE)
#   )
#   valid <- data.frame(
#     Y = rbinom(n_va, 1, 0.5),
#     X1 = rnorm(n_va),
#     X2 = rnorm(n_va),
#     S  = sample(seq_len(J), n_va, replace = TRUE)
#   )
#
#   ipw <- runif(n_tr, 0.5, 2)
#
#   out <- crossfit(
#     train = train,
#     valid.list = list(valid),
#     yname = "Y",
#     xnames = c("X1", "X2"),
#     varnames = list(S = "S", Sdumm = NULL, W = NULL),
#     ipw = ipw,
#     cluster_opt = "FE.glm",
#     type = "binomial",
#     learners = NULL,
#     bounded = TRUE
#   )
#
#   testthat::expect_true(all(is.finite(out$preds)))
#   testthat::expect_true(all(out$preds >= 0 & out$preds <= 1))
# })

testthat::test_that("crossfit RE.glm works when lme4 is available (random intercept)", {
  testthat::skip_if_not_installed("lme4")

  set.seed(4)
  n_tr <- 150
  n_va <- 50
  J <- 10

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.4),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.4),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "RE.glm",
    type = "binomial",
    learners = NULL,
    bounded = TRUE
  )

  testthat::expect_length(out$preds, n_va)
  testthat::expect_true(all(out$preds >= 0 & out$preds <= 1))
})

testthat::test_that("crossfit RE.glm.rs warns and falls back to random intercept when slopes missing", {
  testthat::skip_if_not_installed("lme4")

  set.seed(5)
  n_tr <- 120
  n_va <- 40
  J <- 8

  train <- data.frame(
    Y = rnorm(n_tr),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = factor(sample(seq_len(J), n_tr, replace = TRUE))
  )
  valid <- data.frame(
    Y = rnorm(n_va),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = factor(sample(seq_len(J), n_va, replace = TRUE))
  )

  testthat::expect_warning(
    out <- crossfit(
      train = train,
      valid.list = list(valid),
      yname = "Y",
      xnames = c("X1", "X2"),
      varnames = list(S = "S", Sdumm = NULL, W = NULL),
      cluster_opt = "RE.glm.rs",
      type = "gaussian",
      learners = NULL,
      bounded = FALSE,
      random_slope_vars = NULL,
      source_label = "test"
    ),
    regexp = "no random_slope_vars"
  )

  testthat::expect_length(out$preds, n_va)
  testthat::expect_true(all(is.finite(out$preds)))
})

# Helper to add cluster means and centered-within-cluster columns required by cwc / sufficient_stats / cwc.FE
.make_cluster_features <- function(df, yname, xnames, Sname) {
  # cluster means for x's and y
  for (x in xnames) {
    df[[paste0(x, "_clmean")]] <- ave(df[[x]], df[[Sname]], FUN = mean)
    df[[paste0(x, "_cwc")]]    <- df[[x]] - df[[paste0(x, "_clmean")]]
  }
  df[[paste0(yname, "_clmean")]] <- ave(df[[yname]], df[[Sname]], FUN = mean)
  df[[paste0(yname, "_cwc")]]    <- df[[yname]] - df[[paste0(yname, "_clmean")]]
  df
}

testthat::test_that("crossfit cwc runs with SuperLearner if installed (fast library) and returns bounded preds", {
  testthat::skip_if_not_installed("SuperLearner")

  set.seed(6)
  n_tr <- 120
  n_va <- 50
  J <- 6

  train <- data.frame(
    Y = rbinom(n_tr, 1, 0.4),
    X1 = rnorm(n_tr),
    X2 = rnorm(n_tr),
    S  = sample(seq_len(J), n_tr, replace = TRUE)
  )
  valid <- data.frame(
    Y = rbinom(n_va, 1, 0.4),
    X1 = rnorm(n_va),
    X2 = rnorm(n_va),
    S  = sample(seq_len(J), n_va, replace = TRUE)
  )

  train <- .make_cluster_features(train, "Y", c("X1", "X2"), "S")
  valid <- .make_cluster_features(valid, "Y", c("X1", "X2"), "S")

  out <- crossfit(
    train = train,
    valid.list = list(valid),
    yname = "Y",
    xnames = c("X1", "X2"),
    varnames = list(S = "S", Sdumm = NULL, W = NULL),
    cluster_opt = "cwc",
    type = "binomial",
    learners = c("SL.mean"),  # fast + stable
    bounded = TRUE
  )

  testthat::expect_length(out$preds, n_va)
  testthat::expect_true(all(out$preds >= 0 & out$preds <= 1))
})


testthat::test_that("crossfit errors informatively when required cwc columns are missing", {
  testthat::skip_if_not_installed("SuperLearner")

  set.seed(9)
  train <- data.frame(
    Y = rbinom(80, 1, 0.4),
    X1 = rnorm(80),
    X2 = rnorm(80),
    S  = sample(1:5, 80, replace = TRUE)
  )
  valid <- data.frame(
    Y = rbinom(30, 1, 0.4),
    X1 = rnorm(30),
    X2 = rnorm(30),
    S  = sample(1:5, 30, replace = TRUE)
  )

  # Intentionally DO NOT create *_cwc / *_clmean columns
  testthat::expect_error(
    crossfit(
      train = train,
      valid.list = list(valid),
      yname = "Y",
      xnames = c("X1", "X2"),
      varnames = list(S = "S", Sdumm = NULL, W = NULL),
      cluster_opt = "cwc",
      type = "binomial",
      learners = c("SL.mean"),
      bounded = TRUE
    ),
    regexp = "object.*_cwc|subscript out of bounds|undefined columns",
    ignore.case = TRUE
  )
})

