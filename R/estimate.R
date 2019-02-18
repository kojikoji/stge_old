estimateVB <- function(dm.py.obj, marker.gene.df,
                       t_corr, sigma_f, sigma_s, sigma_t, l_corr,
                       vb_iter, iter_num){
  stge.py.obj <- py_data_manupulation$initiate_stge(dm.py.obj, marker.gene.df, l_corr=l_corr, t_corr=t_corr, sigma_f=sigma_f, sigma_s=sigma_s, sigma_t=sigma_t)
  stge.py.obj <- py_data_manupulation$optimize_stge(stge, vb_iter=vb_iter, iter_num=iter_num)
  return(stge.py.obj)
}

estimateOnlyMuNoTs <- function(stge.py.obj, gene.df,
                                          t_corr, sigma_f, sigma_s, sigma_t, l_corr,
                                          vb_iter, iter_num){
  stge.py.obj$set_params(l_corr=l_corr, t_corr=t_corr,
		  sigma_f=sigma_f, sigma_s=sigma_s, sigma_t=sigma_t)
  stge.py.obj$sc_mode()
  return(stge.py.obj)
}

saveStge2Rlist <- function(stge.py.obj){
  list(
    gene_id_list = stge.py.obj$gene_id_list,
    gene_name_list = stge.py.obj$gene_name_list,
    sample_idx_vec_dict = stge.py.obj$dm$ct$sample_idx_vec_dict,
    Pi_list = stge.py.obj$Pi_list,
    mDelta_list = stge.py.obj$mDelta_list,
    L = stge.py.obj$L,
    params = list(
      l_corr = stge.py.obj$l_corr,
      t_corr = stge.py.obj$t_corr,
      sigma_f = stge.py.obj$sigma_f,
      sigma_s = stge.py.obj$sigma_s,
      sigma_t = stge.py.obj$sigma_t
    ),
    stge.time.dict = stge.py.obj$dm$stage_time_dict,
    sc.data.dict = stge.py.obj$dm$sc_dict,
    ts.args.list = stge.py.obj$dm$ts_args_list,
    ct.args = stge.py.obj$dm$ct$args
  )
}

readRlist2Stge <- function(stge.save.list){
  ct.obj <- do.call(setupPrecomputedCellTracker, stge.save.list$ct.args)
  dm.py.obj <- initializeDataManager(
    stge.save.list$sc.data.dict, ct.obj, stge.save.list$stage.time.dict
  )
  for(ts.args in stge.save.list$ts.args.list){
    dm.py.obj <- do.call(ts.args, addTsData(dm.py.obj))
  }
  stge.py.obj <- py_STGE$STGE()
  stge.py.obj$dm <- dm.py.obj
  stge.py.obj$gene_id_list <- stge.save.list$gene_id_list
  stge.py.obj$gene_name_list <- stge.save.list$gene_name_list
  stge.py.obj$dm$ct$sample_idx_vec_dict <- stge.save.list$sample_idx_vec_dict
  stge.py.obj$Pi_list <- stge.save.list$Pi_list
  stge.py.obj$mDelta_list <- stge.save.list$mDelta_list
  stge.py.obj$L <- stge.save.list$L
  do.call(stge.py.obj$set_params, stge.save.list$params)
  stge.py.obj$A <- stge.py.obj$dm$get_ts_assignment_matrix()
  stge.py.obj$Ys <- stge.py.obj$dm$get_sc_exp_mat(stge.py.obj$gene_id_list)
  stge.py.obj$Yt <- stge.py.obj$dm$get_ts_exp_mat(stge.py.obj$gene_id_list)
  return(stge.py.obj)
}
