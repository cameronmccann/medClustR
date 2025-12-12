################################################################################
# Script:        00_initialize.R
# Package:       medClustR
# Purpose:       One-time project initialization (scaffold, git, license, etc.)
# Author:        Cameron McCann
# Created:       2025-12-12
# Last Updated:  2025-12-12
#
# Preconditions:
#   - Run from package root (usethis::proj_get() == getwd()).
#   - Do NOT run repeatedly unless you know what you’re doing.
#
# Side effects:
#   - Modifies DESCRIPTION, adds files/folders, writes .Rbuildignore, etc.
#
# Notes / TODO:
#   - [ ] Add pkgdown once README is stable
#   - [ ] Configure CI (GitHub Actions)
#   - Note:
#       This code was not ran. The actual code that initiated is somewhat
#       recorded in "99_old-learning-code.R"
#
# Note:
#   - Steps for initial project setup:
#     1) Project was initially created in github then cloned to desktop.
#     2) To change project into a package, I ran the following code:
#           usethis::create_package(".", open = FALSE)
#         and selected "I agree" with project nested within existing project
#         (this makes the R project into an R package) & "Absolutely" to
#         overwrite pre-existing .Rproj
#     3) Then I ran the code in this script
#
# Changelog:
#   - 2025-12-12: Initial version.
################################################################################

# setup -------------------------------------------------------------------

# Check root
stopifnot(getwd() == usethis::proj_get())

# Create a 'dev' folder if it doesn't exist
if (!dir.exists("dev")) dir.create("dev")

# keep dev/ out of builds
usethis::use_build_ignore("dev")

# license/readme/test infra
usethis::use_mit_license()
# initiate readme
usethis::use_readme_md()
# initial setup for testing (3rd edition)
usethis::use_testthat(3)
usethis::use_roxygen_md()
usethis::use_package("R", type = "Depends", min_version = "4.1.0")

# optional: git and gitignore
# usethis::use_git()
# usethis::use_git_ignore(c(".Rproj.user", ".DS_Store"))

# optional: pkgdown, data-raw
# usethis::use_pkgdown()
# usethis::use_data_raw()

# These functions setup parts of the package and are typically called once per package:
# create_package()
# use_git()
# use_mit_license()
# use_testthat()
# use_github()
# use_readme_rmd()


