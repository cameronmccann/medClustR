################################################################################
# Script:        04_release.R
# Purpose:       Pre-release heavy checks, version bump, build
# Side effects:  Builds package; may change DESCRIPTION/NEWS.md.
################################################################################

devtools::document()
# testthat::test()
devtools::check()

# optional:
# usethis::use_news_md()
# usethis::use_version("minor")  # or "patch"/"major"
# devtools::build()

# Last check
devtools::check()

#
devtools::install()
