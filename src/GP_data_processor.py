#-*- coding: utf-8 -*-
import numpy as np
import itertools
import h5py


class GP_data_processor:
    def gp2cartesian(R, two_sin, phi):
        # angle for azim axis
        theta = np.arcsin(two_sin/2)
        # distance from azim axis
        Rp = R*np.cos(theta)
        return(np.array([Rp*np.cos(phi), Rp*np.sin(phi), R*two_sin/2]))

    def gp_grid2point_mat(r_array, two_sin_array, phi_array):
        # iterator for grid point in gp
        gp_iterator = itertools.product(r_array, two_sin_array, phi_array)
        return(np.array
               ([GP_data_processor.gp2cartesian(R, two_sin, phi)
                 for R, two_sin, phi in gp_iterator]))

    def sample_point_mat(point_mat, dvec, size=10000):
        # sample index from 0 ~ nrow(point_mat) -1 based on dvec
        idx_all = np.array(range(point_mat.shape[0]))
        idx = np.random.choice(idx_all, size=size, p=dvec/np.sum(dvec))
        return(point_mat[idx, :])

    def nearest_array_index(val, array):
        return((np.abs(array - val)).argmin())
    # instance methods below

    def register_arrays(self, r_array, two_sin_array, phi_array):
        # register all array
        self.r_array = r_array
        self.two_sin_array = two_sin_array
        self.phi_array = phi_array
        # register cartesian point mat
        self.point_mat = GP_data_processor.gp_grid2point_mat(
            r_array, two_sin_array, phi_array)

    def register_file(self, fname):
        f = h5py.File(fname)
        self.hpf_array = np.array([f[ref[0]][0, 0]
                                   for ref in f['model_data']['time_hpf']])
        self.density_mat_list = [np.array(f[ref[0]])
                                 for ref in f['model_data']['Density']]
        # nearest_time_index(10)
        r_array = np.array([x[0] for x in f['model_info']['BinsDist']])
        two_sin_array = np.array(
            [x[0] for x in f['model_info']['BinsElevDens']])
        phi_array = np.array([x[0] for x in f['model_info']['BinsAzim']])
        self.register_arrays(r_array, two_sin_array, phi_array)

    # given density
    def sample_point_density(self, dmat, size=10000):
        dvec = dmat.flatten()
        return(GP_data_processor.sample_point_mat(self.point_mat,
                                                  dvec, size=size))
    # neaest time in hpf_array

    def nearest_time_index(self, hpf):
        return(GP_data_processor.nearest_array_index(hpf, self.hpf_array))
    # given time

    def sample_point_time(self, hpf, size=10000):
        time_index = self.nearest_time_index(hpf)
        dvec = self.density_mat_list[time_index].flatten()
        return(GP_data_processor.sample_point_mat(self.point_mat,
                                                  dvec, size=size))
