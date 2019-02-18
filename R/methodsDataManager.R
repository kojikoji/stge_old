##' Initialize data_manager (python module)
##'
##' Use data storage for single cell RNA seq, Tomo seq and cell trajectory.
##' @title 
##' @param sc.data.dict List, each elements are single cell RNA seq count matrix. Labeled by obseved time points.
##' @param ct.obj Python object, instance of \code{cell_tracker}
##' @return dm.py.obj Python object, instance of \code{data_manager}
##' @author Yasuhiro Kojima
initializeDataManager <- function(sc.data.dict, ct.obj, stage.time.dict){
  dm.py.obj <- py_data_manager$data_manager()
  dm.py.obj$ct <- ct.obj
  dm.py.obj$stage_time_dict <- stage.time.dict
  dm.py.obj$register_sc_dict(sc.data.dict)
  dm.py.obj$process()
  dm.py.obj$normalize_sc_dict()
  dm.py.obj$ts_args_list <- list()
  return(dm.py.obj)
}

addTsData <- function(dm.py.obj, prefix, hpf, stage = "shield"){
  if(stage == "shield"){
    dm.py.obj$register_tomoseq(prefix, hpf)
  }else{
    dm.py.obj$register_tomoseq_ss(prefix, hpf)
  }
  dm.py.obj$ts_args_list <- c(dm.py.obj$ts_args_list,
                              list(prefix =  prefix, hpf =  hpf, stage =  stage)
                              )
  return(dm.py.obj)
}

setupPrecomputedCellTracker <- function(base.ct.path, ancestor.dict.path, sample.idx.vec.path ,point.num){
  ct.py.obj <- py_utils$load_obj(base.ct.path)
  ct.py.obj$set_precomputed(ancestor.dict.path, sample.idx.vec.path, point.num)
  ct.py.obj$args <- list(base.ct.path = base.ct.path,
                              ancestor.dict.path = ancestor.dict.path,
                              sample.idx.vec.path = sample.idx.vec.path,
                              point.num = point.num)
  return(ct.py.obj)
}


setupScDataList <- function(sc.data.path.dict){
  purrr::map(sc.data.path.dict, ~ readr::read_csv(.x))
}

