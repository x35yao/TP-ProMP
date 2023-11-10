import numpy as np

def prepare_data_for_pmp(gripper_trajs_in_objs, objs, demos, dims):
    '''
    Concatenate data from reference frames attached to each object

    :param gripper_trajs_in_objs: dict
        The dictionary that contains the gripper trajectory in each object reference frame
    :param objs: list
        A list that objects used in this task
    :param demos: list
        A list of demo numbers
    :return:

    data_concat: list
        The concatenated trajectories, where each entry is for a demo.
    times: list
        A list ot time dimension for each trajectory, where each entry is for a demo.
    '''
    Q = []
    times = []
    for d in demos:
        t = gripper_trajs_in_objs['global'][d]['Time'].to_numpy().flatten()
        t = t / t[-1]
        times.append(t)
        temp = []
        for obj in objs:
            temp.append(
            gripper_trajs_in_objs[obj][d].loc[:, dims].to_numpy())
        Q.append(np.hstack(temp))
    return Q, times

def prepare_data_for_gmm(gripper_trajs_in_objs, objs, demos, dims, normalize_t = False):
    '''
    Concatenate data from reference frames attached to each object

    :param gripper_trajs_in_objs: dict
        The dictionary that contains the gripper trajectory in each object reference frame
    :param objs: list
        A list that objects used in this task
    :param demos: list
        A list of demo numbers
    :return:

    data_concat: list
        The concatenated trajectories, where each entry is for a demo.

    '''
    Q = []
    dims = ['Time'] + dims
    for obj in objs:
        temp = []
        for d in demos:
            if not normalize_t:
                temp.append(gripper_trajs_in_objs[obj][d][dims].to_numpy())
            else:
                traj = gripper_trajs_in_objs[obj][d][dims].to_numpy()
                traj[:, 0] = traj[:, 0] / traj[-1, 0]
                temp.append(traj)
        Q.append(np.vstack(temp))
    return Q

def get_demo_HTs(HTs, demo, objs):
    '''
    This function return the Homogeneous transformations of each object reference frame for one demo
    :param HTs: dict
        HTs['obj']['demo']
    :param demo: str
        The demo id.
    :param objs: list
        List of objects that we place reference frame on.
    :return:
    HTs_demo: list
       The homogeneous transformations of each object reference frame for a given demo.
    '''
    HTs_demo = []
    for obj in objs:
        HTs_demo.append(HTs[obj][demo])
    return HTs_demo

def get_mean_cov_hats(ref_means, ref_covs, min_len=None):
    '''
    This function computes the average mean and covariance across different object model.

    Parameters:
    ----------
    ref_means: list
        The means for models in each object reference frame.
    ref_covs: list
        The means for models in each object reference frame.
    min_len: int
        The minimum length that are desired. If None is given, it will be the minimum length ref_mean

    Returns:
    -------
    mean_hats: array
        N by D array, where N is the number of data points and D is the dimension of the data. Average mean at each data point.
    sigma_hats: array
        N * D * D array,where N is the number of data points and D is the dimension of the data. Average covariance at each data point.
    '''

    sigma_hats, ref_pts = [], len(ref_means)

    if not min_len:
        min_len = min([len(r) for r in ref_means])

    # solve for global covariance
    for p in range(min_len):
        covs = [cov[p] for cov in ref_covs]
        inv_sum = np.zeros(ref_covs[0][0].shape)
        for ref in range(ref_pts):
            inv_sum += np.linalg.inv(covs[ref])
        sigma_hat = np.linalg.inv(inv_sum)
        sigma_hats.append(sigma_hat)

    # solve for global mean
    mean_hats = []
    for p in range(min_len):
        mean_w_sum = np.zeros(ref_means[0][0].shape)
        for ref in range(ref_pts):
            mean_w_sum += np.matmul(np.linalg.inv(ref_covs[ref][p]), ref_means[ref][p])
        mu = np.matmul(sigma_hats[p], mean_w_sum)
        mean_hats.append(mu)
    return np.array(mean_hats), np.array(sigma_hats)

def get_position_difference_per_step(d1, d2):
    return np.linalg.norm(d1 - d2, axis = 1)

def min_max_var(covs):
    """
    returns the index and minimum overall dimension variance across all timestep
    Parameters:
    -----------
    covs: list
        A list of covariances index by time
    Returns:
    --------
    best_index: int
        index of the minimum.
    best_min_var: float
        the optimal variance founded
    """
    best_index, best_min_var = 0, float('inf')
    for i, cov in enumerate(covs):
        t_max_var = np.diag(cov).max()
        if t_max_var < best_min_var:
            best_index = i
            best_min_var = t_max_var
    return best_index, best_min_var

def obj_ref_traj(ref_covs):
    """
    return the name of the reference frame with the best minimum overall dimension variance
    across all timestep.
    Parameters:
    -----------
    ref_covs: dict
        a dictionary containing the name of the reference frames as keys and the covariance
        across timesteps as values.
    Returns:
    --------
    best_ref: str
        name of the best reference frame
    """
    best_ref, best_min_var_all = None, float('inf')
    for ref, covs in ref_covs.items():
        _, min_ref_var = min_max_var(covs)
        if min_ref_var < best_min_var_all:
            best_min_var_all = min_ref_var
            best_ref = ref
    return best_ref

def get_data_covs(data_all_frames_tp_pmp, objs):
    covs_all_rfs = {}
    n_dims = int(np.array(data_all_frames_tp_pmp).shape[2] / len(objs))
    for obj_ind, obj in enumerate(objs):
        data_per_frame = np.array(data_all_frames_tp_pmp)[:, :, obj_ind * n_dims: (obj_ind + 1) * n_dims]
        covs = []
        for j in range(data_per_frame.shape[1]):
            cov = np.cov(data_per_frame[:, j, :].T)
            covs.append(cov)
        covs_all_rfs[obj] = np.array(covs)
    return covs_all_rfs
def select_rfs(covs_all_rfs, n_selection):
    '''
    Parameters:
    covs_all_rfs: covirance matrix for all reference frames as a dictionary
    n_selection: number of reference frames need to be selected

    Returns:
    selected_objs: The names of the objects selected.
    '''

    selected_objs = []
    for i in range(n_selection):
        selected_obj = obj_ref_traj(covs_all_rfs)
        selected_objs.append(selected_obj)
        covs_all_rfs.pop(selected_obj)
        if '-' in selected_obj: ### paried-obj is selected e.g. puck-net
            covs_all_rfs.pop(selected_obj.split('-')[0]) ### remove obj puck
        else: ### obj is selected
            paired_objs = [obj for obj in covs_all_rfs.keys() if obj.split('-')[0] == selected_obj]
            for paired_obj in paired_objs:
                covs_all_rfs.pop(paired_obj) ### remove all paired-objs start from puck
    return selected_objs