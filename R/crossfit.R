# {UPDATE DOCUMENTATION AT SOMEPOINT}
# As of 2025-01-02: only modified comments in code; did not modify code yet
# On 2025-07-21: added arguments & code so users can set random slopes for specific variables  & source_label argument for warning messages


#' @title crossfit
#'
#' @description
#' Fits a model (either random effects, fixed effects, or a SuperLearner approach) on
#' a training dataset and uses it to predict outcomes on a set of validation datasets.
#'
#' @details
#' The `crossfit()` function is primarily used for cross-fitting.
#' The \code{cluster_opt} argument selects:
#' \itemize{
#'   \item \code{"RE.glm"}: Random-effect model via \code{lme4::lmer} (Gaussian)
#'   or \code{lme4::glmer} (Binomial), with a random intercept for the cluster identifier `S`.
#'
#'   \item \code{"RE.glm.rs"}: Random-effect model with random slopes for variables in
#'   `random_slope_vars` via \code{lme4::lmer}/\code{lme4::glmer}.
#'
#'   \item \code{"FE.glm"}: GLM with cluster fixed effects by including `S`.
#'
#'   \item \code{"noncluster.glm"}: GLM without any fixed effects.
#'
#'   \item \code{"fix.mlr"} or \code{"noncluster.mlr"}: SuperLearner-based prediction using cluster dummies or not.
#'
#'   \item \code{"sufficient_stats"}: Centered within clusters or sufficient statistics approach.
#'
#'   \item \code{"cwc"}: SuperLearner using centered-within-cluster covariates
#'   (expects columns \code{paste0(xnames, "_cwc")} and \code{paste0(xnames, "_clmean")}
#'   and \code{paste0(yname, "_clmean")} in the input data).
#'
#'   \item \code{"cwc.FE"}: SuperLearner using centered-within-cluster covariates
#'   plus cluster dummy indicators (expects dummy names in `varnames$Sdumm` and
#'   centered features \code{paste0(xnames, "_cwc")}).
#' }
#'
#' When `bounded = TRUE`, predictions are passed through `bound()` to keep
#' predictions in a valid range.
#'
#' @param train A data frame containing the training set/fold (used to fit the model).
#'   Must include the outcome `yname`, predictors `xnames` (and any `varnames$W`),
#'   and the cluster identifier `varnames$S`.
#'
#' @param valid.list A list of one or more data frames. Each data frame is a
#'   validation set/fold used for prediction. Must contain the same variables as
#'   `train` needed by the chosen `cluster_opt`.
#'
#' @param yname Character string specifying the outcome variable name (e.g., \code{"Y"}).
#'
#' @param xnames Character vector of predictor variable names (e.g., \code{c("X1", "X2")}).
#'
#'
#'
#' @param varnames List containing named elements that specify additional variables:
#'   \itemize{
#'     \item \code{S}: The cluster identifier variable name.
#'     \item \code{Sdumm}: Vector of dummy variable names (required for
#'     `cluster_opt = "fix.mlr"` and used by `cluster_opt = "cwc.FE"`). Can be `NULL`
#'     otherwise.
#'     \item \code{W}: (Optional) Additional covariate names to include.
#'   }
#'
#' @param ipw (Optional) Numeric vector of inverse probability weights to use during model fitting.
#'   If \code{NULL} (default), all observations receive weight 1.
#'
#' @param cluster_opt Character string specifying the clustering or modeling approach.
#'   Possible values include \code{"RE.glm"}, \code{"FE.glm"}, \code{"noncluster.glm"}, \code{"fix.mlr"},
#'   \code{"noncluster.mlr"}, \code{"cwc"}, \code{"sufficient_stats"}, \code{"cwc.FE"}.
#'   See Details for how each is handled.
#'
#' @param type Character string specifying the outcome family/distribution, such as \code{"binomial"} or \code{"gaussian"}.
#'
#' @param learners Character vector specifying a SuperLearner library
#' (passed to \code{SuperLearner::SuperLearner} as `SL.library`) when
#' `cluster_opt` uses SuperLearner. Ignored for GLM / random-effect models.
#'
#' @param bounded Logical. If `TRUE`, apply `bound()` to predictions before
#'   returning. Default is `FALSE`.
#'
#' @param random_slope_vars Optional character vector of variable names to use
#'   as random slopes when `cluster_opt = "RE.glm.rs"`. If `NULL` or empty, the
#'   function falls back to a random-intercept-only structure and issues a
#'   warning.
#'
#' @param source_label Optional character scalar used to prefix informational
#'   messages/warnings (useful when `crossfit()` is called inside other
#'   functions).
#'
#' @return A list with two elements:
#' \describe{
#'   \item{\code{fit}}{The fitted model object (e.g., a \code{glm}, \code{lmer}, or \code{SuperLearner} object).}
#'   \item{\code{preds}}{A matrix or vector of predictions, with one column per validation set if there are multiple
#'                       entries in \code{valid.list}.}
#' }
#'
#' @examples
#' \dontrun{
#' # Example: binomial GLM with cluster fixed effects
#' set.seed(1)
#' train <- data.frame(
#'   Y = rbinom(100, 1, 0.3),
#'   X1 = rnorm(100),
#'   X2 = rnorm(100),
#'   S  = sample(1:5, 100, replace = TRUE)
#' )
#' valid <- data.frame(
#'   Y = rbinom(50, 1, 0.3),
#'   X1 = rnorm(50),
#'   X2 = rnorm(50),
#'   S  = sample(1:5, 50, replace = TRUE)
#' )
#'
#' out <- crossfit(
#'   train = train,
#'   valid.list = list(valid),
#'   yname = "Y",
#'   xnames = c("X1", "X2"),
#'   varnames = list(S = "S", Sdumm = NULL, W = NULL),
#'   cluster_opt = "FE.glm",
#'   type = "binomial",
#'   learners = NULL,
#'   bounded = TRUE
#' )
#' head(out$preds)
#'
#' # Example: SuperLearner with explicit cluster dummies (user must create them)
#' # train$Sd1 <- as.integer(train$S == 1)
#' # train$Sd2 <- as.integer(train$S == 2)
#' # valid$Sd1 <- as.integer(valid$S == 1)
#' # valid$Sd2 <- as.integer(valid$S == 2)
#' #
#' # out_sl <- crossfit(
#' #   train = train,
#' #   valid.list = list(valid),
#' #   yname = "Y",
#' #   xnames = c("X1", "X2"),
#' #   varnames = list(S = "S", Sdumm = c("Sd1","Sd2"), W = NULL),
#' #   cluster_opt = "fix.mlr",
#' #   type = "binomial",
#' #   learners = c("SL.mean","SL.glm"),
#' #   bounded = TRUE
#' # )
#' }
#'
#' @seealso
#'  \code{\link[lme4]{lmer}}, \code{\link[lme4]{glmer}},
#'  \code{\link[stats]{glm}}, \code{\link[SuperLearner]{SuperLearner}}
#'
#' @importFrom SuperLearner SuperLearner
#' @importFrom lme4 lmer glmer
#' @importFrom glue glue
#' @importFrom dplyr bind_cols
#' @importFrom stats predict as.formula binomial gaussian glm
#'
#' @export

crossfit <- function(train, valid.list, yname, xnames, varnames,
                     ipw = NULL,
                     cluster_opt = "FE.glm",
                     type, learners, bounded = FALSE,
                     random_slope_vars = NULL, source_label = NULL) {

  # 1. SET UP VARIABLE NAMES AND FAMILIES
  # Set up variable names & families ----------------------------------------

  # Extract the name of the cluster variable (S) and any dummy variables (Sdumm)
  Sname <- varnames$S
  Sname_dummies <- varnames$Sdumm

  # Determine if the outcome is binomial or gaussian
  family <- ifelse(type == "binomial", stats::binomial(), stats::gaussian())

  # Create a data frame---with outcome (Y), covariates (Xs & W), & cluster IDs---used for model fitting
  df_lm <- data.frame(
    Y = train[[yname]],
    train[, c(xnames, varnames$W), drop = FALSE],
    S = train[[Sname]]
  )


  # 2. HANDLE OPTIONAL IPW (INVERSE PROBABILITY WEIGHTS)
  # Handle optional IPW (inverse probability weights) -----------------------

  # If ipw is provided, store these weights in the 'wreg' column
  if (length(ipw) > 0) {
    df_lm$wreg <- ipw
  }
  # Otherwise, if ipw is NULL or empty, use weights = 1
  if (length(ipw) == 0) {
    df_lm$wreg <- rep(1, nrow(df_lm))
  }

  # 3. RANDOM EFFECTS GLM (RE.glm)
  # Random effects glm (RE.glm) ---------------------------------------------

  # Using lmer/glmer from the lme4 package for random effects
  if (cluster_opt == "RE.glm") {

    # Build a formula that includes all covariates and a random intercept for S
    REformula <- paste("Y ~", paste(c(xnames, varnames$W), collapse = " + "), "+ (1 | S)")

    # Fit either a linear mixed model (gaussian) or a generalized linear mixed model (binomial)
    if (family[[1]] == "gaussian") {
      fit <- lme4::lmer(
        formula = REformula,
        weights = wreg,
        data = df_lm
      )
    }
    if (family[[1]] != "gaussian") {
      fit <- lme4::glmer(
        formula = REformula,
        weights = wreg,
        data = df_lm,
        family = family[[1]]
      )
    }

    # Generate predictions for each validation set in valid.list
    preds <- sapply(valid.list, function(validX) {
      # Construct a new data frame with the same columns used in the model
      newX <- data.frame(
        validX[, c(xnames, varnames$W), drop = FALSE],
        S = validX[[Sname]]
      )
      # Predict on this new data
      preds <- stats::predict(fit, newX, type = "response")

      # If bounded = TRUE, constrain predictions to [0,1]
      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)

  }



  # 3b. RANDOM EFFECTS GLM (RE.glm.rs)
  # Random effects glm with random slopes (RE.glm.rs)  ----------------------

  # random effects & random slopes ------------------------------------------
  if (cluster_opt == "RE.glm.rs") {

    # Build random effects formula
    # (B) Determine slope structure and emit informative warning if needed
    if (!is.null(random_slope_vars) && length(random_slope_vars) > 0) {
      slope_formula <- paste("1 +", paste(random_slope_vars, collapse = " + "))
      if (!is.null(source_label)) {
        message(paste0("In ", source_label, "(): Using random slopes for: ", paste(random_slope_vars, collapse = ", ")))
      }
    } else {
      msg <- "RE.glm.rs specified but no random_slope_vars provided; using random intercept only (RE.glm)."
      if (!is.null(source_label)) {
        msg <- paste0("In ", source_label, "(): ", msg)
      }
      warning(msg)
      slope_formula <- "1"
    }
    # if (!is.null(random_slope_vars) && length(random_slope_vars) > 0) {
    #     slope_formula <- paste("1 +", paste(random_slope_vars, collapse = " + "))
    # } else {
    #     warning("RE.glm.rs specified but no random_slope_vars provided; using random intercept only (RE.glm).")
    #     slope_formula <- "1"
    # }

    # Construct formula
    REformula <- paste("Y ~", paste(c(xnames, varnames$W), collapse = " + "),
                       "+ (", slope_formula, "| S)")

    # Fit either a linear mixed model (gaussian) or a generalized linear mixed model (binomial)
    if (family[[1]] == "gaussian") {
      fit <- lme4::lmer(
        formula = stats::as.formula(REformula),
        weights = df_lm$wreg,
        data = df_lm
      )
    } else {
      fit <- lme4::glmer(
        formula = stats::as.formula(REformula),
        weights = df_lm$wreg,
        data = df_lm,
        family = family[[1]]
      )
    }

    # Generate predictions
    preds <- sapply(valid.list, function(validX) {
      # Construct a new data frame with the same columns used in the model
      newX <- data.frame(
        validX[, c(xnames, varnames$W), drop = FALSE],
        S = validX[[Sname]]
      )
      # Predict on this new data
      preds <- stats::predict(fit, newdata = newX, type = "response") #, allow.new.levels = TRUE)
      # If bounded = TRUE, constrain predictions to [0,1]
      if (!bounded) return(preds)
      bound(preds)
    }, simplify = TRUE)

  }


  # 4. FIXED EFFECTS GLM (FE.glm) AND NON-CLUSTER GLM
  # Fixed effects glm (FE.glm) & non-cluster glm ----------------------------

  # S is either included as a fixed effect (FE.glm) or exclude it (noncluster.glm)


  ## glm ---------------------------------------------------------------------

  if (cluster_opt %in% c("FE.glm", "noncluster.glm")) {

    # Build different formulas depending on whether or not we include S
    if (cluster_opt == "FE.glm") {
      fit <- stats::glm(
        formula = paste("Y ~ S +", paste(xnames, collapse = " + ")),
        weights = wreg,
        data = df_lm,
        family = family[[1]]
      )
    }
    if (cluster_opt == "noncluster.glm") {
      fit <- stats::glm(
        formula = paste("Y ~", paste(xnames, collapse = " + ")),
        weights = wreg,
        data = df_lm,
        family = family[[1]]
      )
    }

    # Generate predictions for each validation set in valid.list
    preds <- sapply(valid.list, function(validX) {
      # Prepare the new data with the same columns as the model
      newX <- data.frame(
        validX[, xnames, drop = FALSE],
        S = validX[[Sname]]
      )
      # Make predictions using glm
      preds <- stats::predict(fit, newX, type = "response")

      # Bound predictions if requested
      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)
  }


  # 5. SUPERLEARNER WITH CLUSTER DUMMY INDICATORS (fix.mlr or noncluster.mlr)
  # Superlearner ------------------------------------------------------------
  # Superlearner with cluster dummy indicators (fix.mlr or noncluste --------

  # Use SuperLearner with cluster dummies ('fix.mlr') or without ('noncluster.mlr')

  ## with cluster dummy indicators -------------------------------------------
  if (cluster_opt %in% c("fix.mlr", "noncluster.mlr")) {                      ## NOTE: IF fix is fixed-effect, use "FE.mlr" instead of "fix.mlr" to be consistent with "FE.glm"

    # Combine our df_lm with the dummy variables for each cluster
    df_FE <- data.frame(df_lm, train[, Sname_dummies])

    # Build different models depending on whether cluster dummies are used
    set.seed(12345)
    if (cluster_opt == "fix.mlr") {
      fit <- SuperLearner::SuperLearner(
        Y = df_FE$Y,
        X = df_FE[, c(xnames, varnames$W, Sname_dummies), drop = FALSE],
        obsWeights = df_FE$wreg,
        family = family[[1]],
        SL.library = learners
      )
    }
    if (cluster_opt == "noncluster.mlr") {
      fit <- SuperLearner::SuperLearner(
        Y = df_FE$Y,
        X = df_FE[, c(xnames), drop = FALSE],
        obsWeights = df_FE$wreg,
        family = family[[1]],
        SL.library = learners
      )
    }

    # Use the fitted SuperLearner model to predict on the validation sets
    preds <- sapply(valid.list, function(validX) {
      # Build the new data frame with the same variables used in training
      newX <- data.frame(
        validX[, c(xnames, varnames$W, Sname_dummies), drop = FALSE]
      )
      # Predict returns a list; we extract the 'pred' element
      preds <- stats::predict(fit, newX[, fit$varNames])$pred

      # Bound if necessary
      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)
  }


  # 6. CENTERED WITHIN-CLUSTER (CWC) APPROACH WITH SUPERLEARNER
  ## with centered- within-cluster (CWC) -------------------------------------
  if (cluster_opt == "cwc") {
    # Create data frame for training that includes: outcome (Y), cluster-mean-centered covariates (xnames_cwc), cluster mean of Y (yname_clmean), & additional covariates (varnames$W)
    df_cwc <- data.frame(
      Y = train[, glue::glue("{yname}"), drop=TRUE],
      train[, c(glue::glue("{xnames}_cwc"), glue::glue("{yname}_clmean"), varnames$W), drop = FALSE]
    )

    # Fit a SuperLearner model on these centered variables
    fit <- SuperLearner::SuperLearner(
      Y = df_cwc$Y,
      X = df_cwc[, -1, drop = FALSE],
      family = family[[1]],
      SL.library = learners,
      env = environment(SuperLearner::SuperLearner)
    )

    # Predict on validation data
    preds <- sapply(valid.list, function(validX) {

      # Rebuild the new data with cluster means and centered covariates
      newX <- data.frame(
        validX[, c(xnames, glue::glue("{xnames}_clmean"), glue::glue("{yname}_clmean"), varnames$W), drop = FALSE]
      )
      # Create the within-cluster centered version of the X variables
      validX_cwc <- validX[, xnames] - validX[, glue::glue("{xnames}_clmean")]
      colnames(validX_cwc) <- glue::glue("{xnames}_cwc")
      # Combine centered covariates with newX
      newX <- newX |> dplyr::bind_cols(validX_cwc)
      # colnames(newX)
      # Get predictions using the SuperLearner model
      preds <- stats::predict(fit, newX[, fit$varNames])$pred
      # preds <- preds + validX[, glue("{yname}_clmean"), drop=TRUE]

      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)
  }


  # 7. SUFFICIENT STATISTICS APPROACH
  # Sufficient stats --------------------------------------------------------
  if (cluster_opt == "sufficient_stats")  { # continuous outcome
    if (family[[1]] == "binomial") {
      # For binomial, we use Y and cluster-level variables
      df_ss <- data.frame(
        Y = train[, glue::glue("{yname}"), drop=TRUE],
        train[, c(glue::glue("{xnames}_cwc"), glue::glue("{xnames}_clmean"), glue::glue("{yname}_clmean"), varnames$W), drop = FALSE]
      )
    }
    if (family[[1]] == "gaussian") {
      # For gaussian, we use the cluster-mean-centered Y (yname_cwc)
      df_ss <- data.frame(
        Y = train[, glue::glue("{yname}_cwc"), drop=TRUE],
        train[, c(glue::glue("{xnames}_cwc"), glue::glue("{xnames}_clmean"), #glue("{yname}_clmean"),
                  varnames$W), drop = FALSE]
      )
    }

    # Fit a SuperLearner model on the data
    fit <- SuperLearner::SuperLearner(
      Y = df_ss$Y,
      X = df_ss[, -1, drop = FALSE],
      family = family[[1]],
      SL.library = learners,
      env = environment(SuperLearner::SuperLearner)
    )

    # Predict on each validation set
    preds <- sapply(valid.list, function(validX) {

      # Build the new data frame with cluster means, centered covariates, etc.
      newX <- data.frame(
        validX[, c(glue::glue("{yname}_clmean"), xnames, glue::glue("{xnames}_clmean"), varnames$W), drop = FALSE]
      )
      # Again, compute the within-cluster centered covariates
      validX_cwc <- validX[, xnames] - validX[, glue::glue("{xnames}_clmean")]
      colnames(validX_cwc) <- glue::glue("{xnames}_cwc")
      # Bind them
      newX <- newX |> dplyr::bind_cols(validX_cwc)
      # colnames(newX)
      # Get predictions from the model
      preds <- stats::predict(fit, newX[, fit$varNames])$pred

      # For Gaussian outcomes, we add back the cluster mean of Y
      if (family[[1]] == "gaussian") {
        preds <- preds + validX[, glue::glue("{yname}_clmean"), drop=TRUE]
      }

      # Bound if needed
      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)
  }


  # 8. CWC WITH FIXED EFFECTS (cwc.FE)
  # CWC with Fixed effects (cluster dummies) (cwc.FE) -----------------------

  # Similar to 'cwc' but also including cluster dummy variables for FE.

  # ═══════════════════
  #    Debugging version of code
  # ═══════════════════
  # if (cluster_opt == "cwc.FE") {
  #     print("Fitting CWC.FE...")
  #     print(glue::glue("Columns in training data: {paste(colnames(df_cwc), collapse=', ')}"))
  #
  #     fit <- SuperLearner::SuperLearner(
  #         Y = df_cwc$Y,
  #         X = df_cwc[, -1, drop = FALSE],
  #         family = family[[1]],
  #         SL.library = learners,
  #         env = environment(SuperLearner::SuperLearner)
  #     )
  #
  #     if (is.null(fit)) stop("Error: `fit` returned NULL in `cwc.FE`")
  #
  #     preds <- sapply(valid.list, function(validX) {
  #         print(glue::glue("Validation fold size: {nrow(validX)}"))
  #
  #         newX <- data.frame(
  #             validX[, c(xnames, glue::glue("{xnames}_clmean"), glue::glue("{yname}_clmean"), Sname_dummies, varnames$W), drop = FALSE]
  #         )
  #
  #         # Center X within clusters
  #         validX_cwc <- validX[, xnames] - validX[, glue::glue("{xnames}_clmean")]
  #         colnames(validX_cwc) <- glue::glue("{xnames}_cwc")
  #         newX <- newX |> dplyr::bind_cols(validX_cwc)
  #
  #         if (is.null(newX) || ncol(newX) == 0) stop("Error: Validation data is NULL or empty.")
  #         if (!all(fit$varNames %in% colnames(newX))) stop("Error: Missing predictors in newX")
  #
  #         preds <- stats::predict(fit, newX[, fit$varNames])$pred
  #
  #         if (is.null(preds) || nrow(preds) == 0) {
  #             stop(glue::glue("Error: Predictions are NULL or empty for fold {v}."))
  #         }
  #
  #         preds
  #     }, simplify = TRUE)
  # }

  # ═══════════════════
  #    origianl version of code
  # ═══════════════════
  if (cluster_opt == "cwc.FE") { # continuous outcome

    # check cluster dummies are provided
    if (is.null(Sname_dummies) || length(Sname_dummies) == 0) {
      stop("cluster_opt = 'cwc.FE' requires varnames$Sdumm (cluster dummy column names).")
    }

    # Build the training data with cluster-centered covariates, cluster means, and dummy variables
    # colnames(train) # note: train[, glue("{yname}"), drop=TRUE] doesn't work well
    if (family[[1]]=="gaussian") {
      df_cwc <- data.frame(Y = train[, glue::glue("{yname}"), drop=TRUE],
                           train[, c(glue::glue("{xnames}_cwc"), Sname_dummies, glue::glue("{yname}_clmean"), #glue("{varnames$Xnames}_clmean"),# only covariates' cluster means
                                     #glue("{xnames}_clmean"),
                                     varnames$W
                           ), drop = FALSE])
    }
    if (family[[1]]=="binomial") {
      df_cwc <- data.frame(Y = train[, glue::glue("{yname}"), drop=TRUE],
                           train[, c(glue::glue("{xnames}_cwc"), Sname_dummies, glue::glue("{yname}_clmean"), #glue("{varnames$Xnames}_clmean"),# only covariates' cluster means
                                     #glue("{xnames}_clmean"),
                                     varnames$W
                           ), drop = FALSE])
    }

    ##
    # print("Fitting CWC.FE...")
    # print(glue::glue("Columns in training data: {paste(colnames(df_cwc), collapse=', ')}"))

    # Fit a SuperLearner model, now with cluster dummies + cwc columns
    fit <- SuperLearner::SuperLearner(
      Y = df_cwc$Y,
      X = df_cwc[, -1, drop = FALSE],
      family = family[[1]],
      SL.library = learners,
      env = environment(SuperLearner::SuperLearner)
    )

    ##
    if (is.null(fit)) stop("Error: `fit` returned NULL in `cwc.FE`")

    # Predict on validation sets
    preds <- sapply(valid.list, function(validX) {

      ##
      # print(glue::glue("Validation fold size: {nrow(validX)}"))

      # Construct new data that includes xnames, cluster means, cluster dummies, etc.
      newX <- data.frame(
        validX[, c(xnames, glue::glue("{xnames}_clmean"), glue::glue("{yname}_clmean"), Sname_dummies, varnames$W), drop = FALSE]
      )
      # Compute within-cluster centered X
      validX_cwc <- validX[, xnames] - validX[, glue::glue("{xnames}_clmean")]
      colnames(validX_cwc) <- glue::glue("{xnames}_cwc")
      newX <- newX |> dplyr::bind_cols(validX_cwc)
      # colnames(newX)

      ##
      if (is.null(newX) || ncol(newX) == 0) stop("Error: Validation data is NULL or empty.")
      if (!all(fit$varNames %in% colnames(newX))) stop("Error: Missing predictors in newX")

      # Predict using SL
      preds <- stats::predict(fit, newX[, fit$varNames])$pred

      # For gaussian, we might add back the cluster mean
      # e.g. preds <- preds + validX[, glue("{yname}_clmean"), drop=TRUE]
      if (family[[1]]=="gaussian") {
        preds <- preds #+ validX[, glue("{yname}_clmean"), drop=TRUE]
      }

      ##
      if (is.null(preds) || nrow(preds) == 0) {
        stop(glue::glue("Error: Predictions are NULL or empty for fold {v}."))
      }

      if (!bounded) {
        return(preds)
      }
      bound(preds)
    }, simplify = TRUE)
  }

  # 9. RETURN OUTPUT
  # return output -----------------------------------------------------------
  # The final output is a list with:
  # - 'fit': the fitted model object (glm, lmer, SuperLearner, etc.)
  # - 'preds': a matrix or vector of predictions for each validation set
  out <- list(
    fit   = fit,
    preds = preds
  )
  return(out)

}


################################## END #########################################
