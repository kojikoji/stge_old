##' Reload python modules used in STGE
##'
##' This reload python modules data_manager, STGE, utils, cell_tracker
##' @title reloadPythonModule
##' @author Yasuhiro Kojima
reloadPythonModule <- function(){
  py_data_manager <<- importlib$reload(py_data_manager)
  py_STGE <<- importlib$reload(py_STGE)
  py_utils <<- importlib$reload(py_utils)
  py_cell_tracker <<- importlib$reload(py_cell_tracker)
  py_data_manupulation <<- importlib$reload(py_data_manupulation)
}
