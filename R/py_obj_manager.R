recall_py_obj <- function(py.obj.ref){
  py_run_string(paste("from joblib import MemorizedResult; x =", py.obj.ref))
  py.obj <- py$x$get()
}
py_shelve_call <- function(py_func, ...){
  py_builtin$str(py_func$call_and_shelve(...))
}
