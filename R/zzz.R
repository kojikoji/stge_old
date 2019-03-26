safe_import_from_path <- function(module.name, libname, pckgname){
  tryCatch(
    assign(paste("py", module.name, sep="_"), reticulate::import_from_path(module.name, path = file.path(libname, pckgname, "src")), pos = .GlobalEnv),
    error = function(e){
      packageStartupMessage("Install python dependencies...")
      reticulate::py_install(c("numpy", "numba", "scipy", "progressbar2", "pandas"))
      delayedAssign(module.name, reticulate::import_from_path(module.name, path = file.path(libname, pckgname, "src")), assign.env = .GlobalEnv)
      
    })
}

.onLoad <- function(libname, pckgname){
  packageStartupMessage("Import python module in this repository: stge, data_manager") 
  reticulate::use_condaenv("r-reticulate")
  reticulate::use_virtualenv("r-reticulate")
  importlib <<- reticulate::import("importlib", delay_load = TRUE)
  py_builtin <<- reticulate::import_builtins()
  safe_import_from_path("data_manager", libname, pckgname)
  safe_import_from_path("STGE", libname, pckgname)
  safe_import_from_path("utils", libname, pckgname)
  safe_import_from_path("cell_tracker", libname, pckgname)
  safe_import_from_path("data_manupulation", libname, pckgname)
}

