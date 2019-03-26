##' Initialize data_manager (python module)
##'
##' Use data storage for single cell RNA seq, Tomo seq and cell trajectory.
##' @title 
##' @param sc.data.dict List, each elements are single cell RNA seq count matrix. Labeled by obseved time points.
##' @param ct.obj Python object, instance of \code{cell_tracker}
##' @return dm.py.obj Python object, instance of \code{data_manager}
##' @author Yasuhiro Kojima
##'
##' @import purrr
initializeDataManager <- function(sc.data.dict, ct.obj, stage.time.dict, gene.df, ts.arg.list){
  dm.py.obj <- py_data_manager$data_manager()
  dm.py.obj$register_use_gene(gene.df)
  dm.py.obj$point_num <- ct.obj$point_num
  dm.py.obj$ct <- ct.obj
  dm.py.obj$stage_time_dict <- stage.time.dict
  dm.py.obj$register_sc_dict(sc.data.dict)
  dm.py.obj <- purrr::reduce(ts.arg.list,
                             function(pre.dm.py.obj, ts.arg){
                               addTsData(pre.dm.py.obj, ts.arg[["prefix"]], ts.arg[["hpf"]], ts.arg[["shield"]])
                             },
                             .init = dm.py.obj
                             )
  dm.py.obj$process()
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


saveDm2Rlist <- function(dm.py.obj){
  list(
    t.vec = dm.py.obj$t_vec,
    sc.t.vec = dm.py.obj$sc_t_vec,
    ts.t.vec = dm.py.obj$ts_t_vec,
    sc.t.nums = dm.py.obj$sc_t_nums,
    sc.t.breaks = dm.py.obj$sc_t_breaks,
    ref.t.nums = dm.py.obj$ref_t_nums,
    ref.t.breaks = dm.py.obj$ref_t_breaks,
    gene.df = dm.py.obj$gene_df,
    A = dm.py.obj$A,
    Ys = dm.py.obj$Ys,
    Yt = dm.py.obj$Yt
  )
}

readRlist2Dm <- function(dm.list, ct.py.obj){
  dm.py.obj <- py_data_manager$data_manager()
  dm.py.obj$point_num <- ct.py.obj$point_num
  dm.py.obj$ct <- ct.py.obj
  dm.py.obj$t_vec <- dm.list$t.vec
  dm.py.obj$sc_t_vec <- dm.list$sc.t.vec
  dm.py.obj$ts_t_vec <- dm.list$ts.t.vec
  dm.py.obj$sc_t_nums <- dm.list$sc.t.nums
  dm.py.obj$sc_t_breaks <- dm.list$sc.t.breaks
  dm.py.obj$ref_t_nums <- dm.list$ref.t.nums
  dm.py.obj$ref_t_breaks <- dm.list$ref.t.breaks
  dm.py.obj$gene_df <- dm.list$gene.df
  dm.py.obj$A <- dm.list$A
  dm.py.obj$Ys <- dm.list$Ys
  dm.py.obj$Yt <- dm.list$Yt
  return(dm.py.obj)
}

setUpCellTracker <- function(base.ct.path, ancestor.dict.path, sample.idx.vec.path ,point.num){
  ct.py.ref <- py_shelve_call(
    py_data_manupulation$set_up_precomputed_cell_tracker,
    base.ct.path, ancestor.dict.path, sample.idx.vec.path, point.num)
  return(ct.py.ref)
}

setUpDataManager <- function(sc.data.dict, ct.obj, ts.prefix.list, stage.time.dict, gene.df, fix.angle){
  dm.py.ref <- py_shelve_call(
    py_data_manupulation$set_up_data_manager,
    sc.data.dict, ct.obj, ts.prefix.list, stage.time.dict, gene.df, fix.angle)
  return(dm.py.ref)
}
