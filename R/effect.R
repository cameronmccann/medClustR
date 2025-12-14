
# {UPDATE DOCUMENTATION AT SOMEPOINT}
# As of 2025-01-07: did not modify anything yet

effect <- function(thetas) {
  thetas[["DE(,M(0))"]] <- thetas[["Y(1,M(0))"]] - thetas[["Y(0,M(0))"]]
  thetas[["IE(1,)"]] <- thetas[["Y(1,M(1))"]] - thetas[["Y(1,M(0))"]]

  thetas
}
