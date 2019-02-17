.onLoad <- function(libname, pckgname){
  packageStartupMessage("Import python module in this repository: stge, data_manager")
  library(reticulate)
  reticulate::py_install(c("numpy", "numba", "scipy", "progressbar2", "pandas"))
  reticulate::use_condaenv("r-reticulate")
  reticulate::use_virtualenv("r-reticulate")
  pd <<- reticulate::import("pandas")
  py_data_manager  <<- reticulate::import_from_path("data_manager", path = file.path(libname, pckgname, "src"))
  py_STGE  <<- reticulate::import_from_path("STGE", path = file.path(libname, pckgname, "src"))
}
