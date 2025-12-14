

# {UPDATE DOCUMENTATION AT SOMEPOINT}
# As of 2025-01-07: did not modify anything yet


bound <- function(vals, tol = 0.025) {
  vals[vals < tol] <- tol
  vals[vals > 1 - tol] <- 1 - tol
  return(vals)
}
