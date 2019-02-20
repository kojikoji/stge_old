estimateVB <- function(dm.py.obj, marker.gene.df,
                       l_corr, t_corr, sigma_f, sigma_s, sigma_t,
                       vb_iter, iter_num){
  stge.py.obj <- py_data_manupulation$initiate_stge(dm.py.obj, marker.gene.df, l_corr=l_corr, t_corr=t_corr, sigma_f=sigma_f, sigma_s=sigma_s, sigma_t=sigma_t)
  stge.py.obj <- py_data_manupulation$optimize_stge(stge.py.obj, vb_iter=vb_iter, iter_num=iter_num)
  return(stge.py.obj)
}

estimateOnlyMuNoTs <- function(stge.py.obj, gene.df,
                               l_corr, t_corr, sigma_f, sigma_s, sigma_t){
  stge.py.obj$set_params(l_corr=l_corr, t_corr=t_corr,
		  sigma_f=sigma_f, sigma_s=sigma_s, sigma_t=sigma_t)
  stge.py.obj$sc_mode()
  return(stge.py.obj)
}

saveStge2Rlist <- function(stge.py.obj){
  list(
    Pi_list = stge.py.obj$Pi_list,
    mDelta_list = stge.py.obj$mDelta_list,
    L = stge.py.obj$L,
    params = list(
      l_corr = stge.py.obj$l_corr,
      t_corr = stge.py.obj$t_corr,
      sigma_f = stge.py.obj$sigma_f,
      sigma_s = stge.py.obj$sigma_s,
      sigma_t = stge.py.obj$sigma_t
    )
  )
}

readRlist2Stge <- function(stge.save.list, dm.py.obj){
  stge.py.obj <- py_STGE$STGE()
  stge.py.obj$register_data_manager(dm.py.obj)
  stge.py.obj$Pi_list <- stge.save.list$Pi_list
  stge.py.obj$mDelta_list <- stge.save.list$mDelta_list
  stge.py.obj$L <- stge.save.list$L
  do.call(stge.py.obj$set_params, stge.save.list$params)
  return(stge.py.obj)
}
