################################################################################
# Script:        01_setup.R
# Purpose:       Package metadata + dependencies + file stubs
# Side effects:  Updates DESCRIPTION; may create R/ and test files if missing.
################################################################################


# dependencies ------------------------------------------------------------

# Add packages as dependencies
usethis::use_package(package = "SuperLearner", min_version = TRUE)
usethis::use_package(package = "lme4", min_version = TRUE)
usethis::use_package(package = "dplyr", min_version = TRUE)
usethis::use_package(package = "glue", min_version = TRUE)
usethis::use_package(package = "origami", min_version = TRUE)

# usethis::use_package(package = "ggplot2", min_version = TRUE)
# usethis::use_package(package = "mvtnorm", min_version = TRUE)
# usethis::use_package(package = "purrr", min_version = TRUE)
# usethis::use_package("rlang", min_version = TRUE)

usethis::use_package("R", type = "Depends", min_version = "4.1.0")


# files -------------------------------------------------------------------

# Start scripts if not already exist
if (!file.exists("R/bound.R")) usethis::use_r("bound")
if (!file.exists("R/crossfit.R")) usethis::use_r("crossfit")

if (!file.exists("R/effect.R")) usethis::use_r("effect")

# # Original make_fold_K function
# if (!file.exists("R/OG-make_fold_K.R")) usethis::use_r("OG-make_fold_K")
if (!file.exists("R/make_fold_K.R")) usethis::use_r("make_fold_K")

# if (!file.exists("R/eif.R")) usethis::use_r("eif")
#
# if (!file.exists("R/a_c.R")) usethis::use_r("a_c")
# if (!file.exists("R/a_mc.R")) usethis::use_r("a_mc")
# if (!file.exists("R/mu_mac.R")) usethis::use_r("mu_mac")
# if (!file.exists("R/v_ac.R")) usethis::use_r("v_ac")
# if (!file.exists("R/mu_ac.R")) usethis::use_r("mu_ac")
#
# if (!file.exists("R/get_inference.R")) usethis::use_r("get_inference")


# Start test script if not already exist
if (!file.exists("tests/testthat/test-crossfit.R")) {
  usethis::use_test("crossfit")
}

# if (!file.exists("tests/testthat/test-generate_data.R")) {
#   usethis::use_test("generate_data")
# }

# Create function file (if it does not already exist and ready for editing); don't need or ".R" suffix
usethis::use_r("crossfit")

# Create test file (if it does not already exist and ready for editing); don't need "test-" prefix or ".R" suffix
usethis::use_test("crossfit")
## can even use usethis::use_test() while in function script & it will
##    automatically go to that functions corresponding test file; no need
##    for writing the function/file name

# Update readme
devtools::build_readme()

# optional DESCRIPTION fields, authorship, etc.
# usethis::use_author("Cameron McCann", "cameron@…")


# write NAMESPACE and man files
devtools::document()

devtools::check()

# ══════════════════════════════
#    Informal checks
# ══════════════════════════════
# Load package
# devtools::load_all()
#
# # informal check
# ttt <- generate_data(
#   J = 100,
#   njrange = c(50, 100),
#   seed = 123
# )
#
# str(ttt)
#
#


# ══════════════════════════════
#    Test files
# ══════════════════════════════


# Load package
devtools::load_all()

# Run test scripts
testthat::test_file("tests/testthat/test-crossfit.R")












#
#
# library(devtools)
# library(usethis)
# library(testthat)
#
# # Create a 'dev' folder if it doesn't exist
# if (!dir.exists("dev")) dir.create("dev")
#
# # Ignore the dev folder in the build
# usethis::use_build_ignore("dev")
#
# # # (Optional) Ignore project files, notes, etc.
# # usethis::use_build_ignore("README_dev.md")
# # usethis::use_build_ignore(".Rproj.user")
#
#
#
# use_r("generate_data")
#
#
#
#
# load_all()
#
# # informal check
# ttt <- generate_data(
#   J = 100,
#   njrange = c(50, 100),
#   seed = 123
# )
#
# str(ttt)
#
# #
# check()
#
# # set license
# use_mit_license()
#
# # write NAMESPACE and man files
# document()
#
# # Add dependencies in alphabetical order
# usethis::use_package(package = "dplyr",
#                      type = "Imports")
# usethis::use_package(package = "mvtnorm")
#
# #
# check()
#
# # write NAMESPACE and man files
# document()
#
# #
# check()
#
# # Add glue and purrr packages as dependencies
# usethis::use_package(package = "dplyr",
#                      min_version = TRUE)
# usethis::use_package(package = "ggplot2",
#                      min_version = TRUE)
# usethis::use_package(package = "glue",
#                      min_version = TRUE)
# usethis::use_package(package = "mvtnorm",
#                      min_version = TRUE)
# usethis::use_package(package = "purrr",
#                      min_version = TRUE)
#
# # Fixed many roxygen issues in R scripts
#
# # write NAMESPACE and man files
# document()
#
# #
# check()
#
# # Set R version as dependent since we use "|>"
# usethis::use_package("R", type = "Depends", min_version = "4.1.0")
#
# # write NAMESPACE and man files
# document()
#
# # Set rlang as a package used
# usethis::use_package("rlang",
#                      min_version = TRUE)
#
# #
# check()
#
#
# # initial setup for testing (3rd edition)
# usethis::use_testthat(3)
#
# # Create function file (if it does not already exist and ready for editing); don't need or ".R" suffix
# usethis::use_r("generate_clusters")
#
# # Create test file (if it does not already exist and ready for editing); don't need "test-" prefix or ".R" suffix
# usethis::use_test("generate_clusters")
# ## can even use usethis::use_test() while in function script & it will
# ##    automatically go to that functions corresponding test file; no need
# ##    for writing the function/file name
#
# # you might work at level of individual tests initially then work towards running the entire test files and eventually the entire test suite
#
# # Micro-iteration:
# #     This is the interactive phase where you initiate and refine a function and
# #     its tests in tandem. Here you will run devtools::load_all() often, and then
# #     execute individual expectations or whole tests interactively in the console.
# #
# # Mezzo-iteration:
# #     As one file’s-worth of functions and their associated tests start to shape
# #     up, you will want to execute the entire file of associated tests, perhaps
# #     with testthat::test_file():
# #
# # Macro-iteration:
# #     As you near the completion of a new feature or bug fix, you will want to
# #     run the entire test suite. Most frequently, you’ll do this with devtools::test()
# #
# #     devtools::test() is mapped to Ctrl/Cmd + Shift + T
# #     devtools::check() is mapped to Ctrl/Cmd + Shift + E
# #
# # Expectation is the finest level of testing. It makes a binary assertion of whether correct
# # There are over 40 expectations in testthat package: https://testthat.r-lib.org/reference/index.html
# #
# # expect_equal() checks for equality
# # expect_identical() for exact equivalence
# # expect_error() to check whether an expression throws an error.
# # expect_warning() to check whether an expression throws an warning
# # expect_message() to check whether an expression provides message
# #
# # Snapshot tests
# #
#
# # Load package
# devtools::load_all()
#
# #
# testthat::test_file("tests/testthat/test-generate_clusters.R")
#
#
#
# generate_clusters(J = 100,
#                   njrange = c(50, 100),
#                   seed = 123)$N
#
#








