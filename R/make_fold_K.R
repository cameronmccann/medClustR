#' @title make_fold_K
#'
#' @description
#' Creates cross-validation folds by first splitting each cluster internally
#' into \code{cv_folds}, then recombining the v-th fold from each cluster to
#' form global folds. The result is that each validation fold contains
#' observations from *all* clusters, and each cluster contributes to every fold.
#'
#' If \code{Yname} is provided and refers to a binary variable, the function
#' attempts to stratify folds \emph{within each cluster} by the outcome. Within a
#' cluster, stratification is only performed when both outcome classes are present
#' and the minority class count is at least \code{cv_folds}.
#'
#' @param data_in A data frame containing at least an \code{id} column
#'   (unique row IDs) and the cluster variable specified by \code{Sname}.
#'   Internally, the function adds/overwrites a temporary integer cluster-label
#'   column \code{K}.
#' @param Sname Character string giving the name of the cluster variable in \code{data_in}.
#' @param Yname Optional character string giving the name of a binary outcome variable
#'   in \code{data_in} used for within-cluster stratification. Use \code{NULL} to disable
#'   stratification.
#' @param cv_folds Integer; number of cross-validation folds (default = 4).
#'
#' @return A list of length \code{cv_folds}, where each element is a fold containing:
#'   \itemize{
#'     \item \code{validation_set}: integer vector of \code{data_in$id} values used for validation.
#'     \item \code{training_set}: integer vector of \code{data_in$id} values used for training.
#'   }
#'   Together, the folds cover all rows in \code{data_in}.
#'
#' @examples
#' library(origami)
#' set.seed(1)
#' df <- data.frame(
#'   id = 1:20,
#'   cluster = rep(letters[1:4], each = 5),
#'   y = rbinom(20, 1, 0.5),
#'   x = rnorm(20)
#' )
#'
#' # Cluster-balanced folds without stratification
#' folds1 <- make_fold_K(df, Sname = "cluster", Yname = NULL, cv_folds = 2)
#' str(folds1[[1]])
#'
#' # Attempt within-cluster stratification by a binary outcome
#' folds2 <- make_fold_K(df, Sname = "cluster", Yname = "y", cv_folds = 2)
#' str(folds2[[1]])
#'
#' @seealso \code{\link[origami]{make_folds}}, \code{\link[origami]{folds_vfold}}
#'
#' @importFrom origami make_folds folds_vfold
#'
#' @export

make_fold_K <- function(data_in, Sname, Yname = NULL, cv_folds = 4) {

  # Assign integer cluster labels
  S_levels <- unique(data_in[[Sname]])
  data_in$K <- match(data_in[[Sname]], S_levels)

  # Guardrails
  if (!("id" %in% names(data_in))) stop("`data_in` must contain an `id` column.")
  if (!(Sname %in% names(data_in))) stop("`Sname` must be a column in `data_in`.")
  # if (cv_folds < 2) stop("`cv_folds` must be >= 2.")

  # For each unique cluster, split its rows into V folds
  fold_K <- lapply(unique(data_in[[Sname]]), FUN = function(k) {

    # Map label to index (helps if clusters are character, instead of numeric)
    k_idx <- match(k, S_levels)

    # Compute the cluster row indices once
    ind_k <- which(data_in$K == k_idx)

    # (3) Initialize fold_k so it's always defined (robustness)
    fold_k <- NULL

    # Only split if cluster has at least one row
    if (length(ind_k) >= 1) {

      # Build strata within cluster (only when feasible)
      strata_local <- NULL
      if (!is.null(Yname) && Yname %in% names(data_in)) {
        y_k <- data_in[ind_k, Yname, drop = TRUE]   # use ind_k instead of re-subsetting
        tab <- table(y_k)

        if (length(tab) == 2L) {
          if (min(tab) >= cv_folds) {
            strata_local <- as.integer(as.factor(y_k))
          } else {
            warning(sprintf(
              "Cluster '%s': minority count = %d < V = %d; not stratifying this cluster.",
              as.character(k), min(tab), cv_folds
            ))
          }
        } else {
          warning(sprintf(
            "Cluster '%s': binary outcome only has one value; not stratifying this cluster.",
            as.character(k)
          ))
        }
      }

      # Create V-folds within this cluster (stratified when feasible)
      fk <- origami::make_folds(
        data_in[ind_k, , drop = FALSE],      # use ind_k
        fold_fun = origami::folds_vfold,
        V = cv_folds,
        strata_ids = strata_local
      )
      fold_k <- fk

      # Remap fold indices (relative to the cluster) back to global row IDs
      ids_k <- data_in$id[ind_k]            # (4) cache these ids once (efficiency)
      for (v in 1:cv_folds) {
        fold_k[[v]]$validation_set <- ids_k[fk[[v]]$validation_set]
        fold_k[[v]]$training_set <- ids_k[fk[[v]]$training_set]
      }
    }

    return(fold_k)
  })

  # Initialize global folds
  folds <- origami::make_folds(
    data_in,
    fold_fun = origami::folds_vfold,
    V = cv_folds
  )

  # For each v-th fold, combine v-th within-cluster folds across clusters
  for (v in 1:cv_folds) {
    folds[[v]]$validation_set <- unlist(lapply(seq_along(fold_K), FUN = function(k) {
      fold_K[[k]][[v]]$validation_set
    }), use.names = FALSE) # Note: ", use.names = FALSE" was not included previously

    folds[[v]]$training_set <- unlist(lapply(seq_along(fold_K), FUN = function(k) {
      fold_K[[k]][[v]]$training_set
    }), use.names = FALSE)
  }

  folds
}

