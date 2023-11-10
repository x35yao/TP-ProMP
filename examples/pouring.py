import pickle
import random
from tp_pmp import tp_pmp
from matplotlib import pyplot as plt
from plot import remove_repetitive_labels
import numpy as np
from utils import get_position_difference_per_step

if __name__ == '__main__':
    # load data
    task = 'pouring'  ## Tasks are: 'extrapolation', 'pouring', 'shooting', and 'sweeping'
    with open(f'../data/{task}/saved_split/{task}.pickle', 'rb') as f:
        data = pickle.load(f)
    ind = random.randint(0, len(data)-1)
    data_all_frames_tp_pmp, times = data[ind]['train_traj_tp_pmp'], data[ind]['train_times_tp_pmp']

    # Define varialbes and hyperparameters
    dims =  ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] ## Only modeling quaternions for the pouring water task
    n_dims = len(dims)
    if_gmm = True ## gmm is only needed for the pouring water task
    n_components = 2
    sigma = 0.035
    # paired-object reference frames is used for shooting and sweeping task
    reference_frames = data[ind]['objs']
    max_iter = 200
    # Train TP-ProMP model
    data_all_frames_tp_pmp, times = data[ind]['train_traj_tp_pmp'], data[ind]['train_times_tp_pmp']
    model_tp_pmp = tp_pmp.PMP(data_all_frames_tp_pmp, times, n_dims, reference_frames, sigma= sigma, n_components=n_components,
                              covariance_type='diag', max_iter=max_iter, gmm=if_gmm)
    model_tp_pmp.train()
    model_tp_pmp.refine(max_iter)

    # Predict
    test_traj = data[ind]['test_traj_global']
    t = data[ind]['test_t']
    HTs_test_tp_pmp = data[ind]['HTs_test'] ## only use object reference frames
    ## Select the mode closest to the groundtruth for the pouring task
    dist = np.inf
    for i in range(n_components):

        mu_tp_pmp_temp, sigma_tp_pmp_temp = model_tp_pmp.predict(t, HTs_test_tp_pmp, reference_frames, mode_selected=i)
        d_temp = get_position_difference_per_step(test_traj[:, :3], mu_tp_pmp_temp[:, :3])
        if np.mean(d_temp) < dist:
            dist = np.mean(d_temp)
            mu_tp_pmp = mu_tp_pmp_temp

    ### Plot Position
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor('white')
    ax.locator_params(nbins=3, axis='z')
    ### Plot test demo
    line = ax.plot(test_traj[:, 2], test_traj[:, 1], -test_traj[:, 0], '-', color= 'r',
                   label='Test demo')
    ax.plot(test_traj[0, 2], test_traj[0, 1], -test_traj[0, 0], 'o',
            color=line[0].get_color(), label='start')
    ax.plot(test_traj[-1, 2], test_traj[-1, 1], -test_traj[-1, 0], 'x',
            color=line[0].get_color(), label='end')
    mid_ind = int(0.7 * len(test_traj)) ## estimated time when the pouring is happing
    ax.plot(test_traj[mid_ind, 2], test_traj[mid_ind, 1], -test_traj[mid_ind, 0], 's',
            color=line[0].get_color(), label='middle')
    ### Plot train demos
    train_trajs = data[ind]['train_trajs_global']
    for traj in train_trajs:
        line = ax.plot(traj[:, 2], traj[:, 1], -traj[:, 0], '--', color='gray',
                       label='Train demos')
        ax.plot(traj[0, 2], traj[0, 1], -traj[0, 0], 'o',
                color=line[0].get_color(), label='start')
        ax.plot(traj[-1, 2], traj[-1, 1], -traj[-1, 0], 'x',
                color=line[0].get_color(), label='end')
        if task == 'pouring':
            ax.plot(traj[mid_ind, 2], traj[mid_ind, 1], -traj[mid_ind, 0], 's',
                    color=line[0].get_color(), label='middle')
    ### Plot prediction
    pred = mu_tp_pmp
    line = ax.plot(pred[:, 2], pred[:, 1], -pred[:, 0], '-', color='b', label='TP-ProMP')
    ax.plot(pred[0, 2], pred[0, 1], -pred[0, 0], 'o',
            color=line[0].get_color(), label='start')
    ax.plot(pred[-1, 2], pred[-1, 1], -pred[-1, 0], 'x',
            color=line[0].get_color(), label='end')
    ax.plot(pred[mid_ind, 2], pred[mid_ind, 1], -pred[mid_ind, 0], 's',
            color=line[0].get_color(), label='middle')

    ax.set_xlabel('x (mm)', labelpad=30)
    ax.set_ylabel('y (mm)', labelpad=20)
    ax.set_zlabel('z (mm)')
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    handles, labels = ax.get_legend_handles_labels()
    newHandles_temp, newLabels_temp = remove_repetitive_labels(handles, labels)
    newLabels, newHandles = [], []
    for handle, label in zip(newHandles_temp, newLabels_temp):
        if label not in ['start', 'middle', 'end']:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels, prop={'size': 14})
    plt.show()

    ### Plot orientation (for pouring task only)
    fig2, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    ## Train demos
    for traj in train_trajs:
        quats = traj[:, 3:]
        for i, ax in enumerate(axes):
            ax.plot(quats[:, i], '--', color='grey', label='Training demos')
    ## Test demo and prediction
    quats_test = test_traj[:, 3:]
    for i, ax in enumerate(axes):
        ax.plot(quats_test[:, i], '-', color='red', label='Test demo')
        ax.plot(pred[:, i + 3], '-', color= 'blue', label='TP-ProMP')
        ax.set_title(dims[i + 3])
        if i == 3:
            ax.set_xlabel('Time')
    plt.tight_layout(pad=0.05)
    plt.show()








