import pickle
import random
from tp_pmp import tp_pmp
from matplotlib import pyplot as plt
from plot import remove_repetitive_labels
import numpy as np
from utils import get_position_difference_per_step

if __name__ == '__main__':
    # load data
    task = 'extrapolation'
    with open(f'../data/{task}/saved_split/{task}.pickle', 'rb') as f:
        data = pickle.load(f)
    ind = random.randint(0, len(data)-1)
    data_all_frames_tp_pmp, times = data[ind]['train_traj_tp_pmp'], data[ind]['train_times_tp_pmp']

    # Define varialbes and hyperparameters
    dims = ['x', 'y', 'z']
    n_dims = len(dims)
    if_gmm = False
    n_components = 1
    sigma = 0.4
    # paired-object reference frames is used for shooting and sweeping task
    reference_frames = data[ind]['objs']
    max_iter = 50

    # Train TP-ProMP model
    data_all_frames_tp_pmp, times = data[ind]['train_traj_tp_pmp'], data[ind]['train_times_tp_pmp']
    model_tp_pmp = tp_pmp.PMP(data_all_frames_tp_pmp, times, n_dims, reference_frames, sigma= sigma, n_components=n_components, max_iter=max_iter, gmm=if_gmm)
    model_tp_pmp.train(print_lowerbound=False)

    # Predict
    test_traj = data[ind]['test_traj_global']
    t = data[ind]['test_t']
    HTs_test_tp_pmp = data[ind]['HTs_test'] ## only use object reference frames
    mu_tp_pmp, sigma_tp_pmp = model_tp_pmp.predict(t, HTs_test_tp_pmp, reference_frames)

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

    ### Plot train demos
    train_trajs = data[ind]['train_trajs_global']
    for traj in train_trajs:
        line = ax.plot(traj[:, 2], traj[:, 1], -traj[:, 0], '--', color='gray',
                       label='Train demos')
        ax.plot(traj[0, 2], traj[0, 1], -traj[0, 0], 'o',
                color=line[0].get_color(), label='start')
        ax.plot(traj[-1, 2], traj[-1, 1], -traj[-1, 0], 'x',
                color=line[0].get_color(), label='end')

    ### Plot prediction
    pred = mu_tp_pmp
    line = ax.plot(pred[:, 2], pred[:, 1], -pred[:, 0], '-', color='b', label='TP-ProMP')
    ax.plot(pred[0, 2], pred[0, 1], -pred[0, 0], 'o',
            color=line[0].get_color(), label='start')
    ax.plot(pred[-1, 2], pred[-1, 1], -pred[-1, 0], 'x',
            color=line[0].get_color(), label='end')

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
