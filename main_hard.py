import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from tqdm import tqdm
from glob import glob

from architectures.utils import eom_2d_perturb
from architectures.pde_solver_2d import EchoNet, solid_mechanics_network, time_derivative_network


if '__main__' == __name__:

    # Generate data
    # 1. Training
    run_time = 12e-5
    Npts = 6000
    Ncol = 120000

    lb = np.array([0, 0, 0])
    ub = np.array([0.05, 0.05, 12e-5])

    grid_initial = lb + np.array([0.05, 0.05, 0])*lhs(3, Npts)
    grid_initial = torch.from_numpy(grid_initial).float()
    t_init, x_init, y_init = grid_initial.T
    t_init = t_init[:, None]
    x_init = x_init[:, None]
    y_init = y_init[:, None]

    left = np.array([0.0, 0.05, run_time])*lhs(3, Npts)
    right = np.array([0.05, 0.0, run_time])+np.array([0.0, 0.05, run_time])*lhs(3, Npts)
    bottom = np.array([0.05, 0, run_time])*lhs(3, Npts)
    top = np.array([0.0, 0.05, run_time])+np.array([0.05, 0, run_time])*lhs(3, Npts)

    grid_boundary = np.concatenate((left, right, bottom, top), axis=0)
    grid_boundary = torch.from_numpy(grid_boundary).float()
    t_bound, x_bound, y_bound = grid_boundary.T
    t_bound = t_bound[:, None]
    x_bound = x_bound[:, None]
    y_bound = y_bound[:, None]

    grid_col = ub*lhs(3, Ncol)
    grid_col = torch.from_numpy(grid_col).float()

    ptr_path = './data/impact_echo/impact_profile/Han_pulse.npy'
    t_ptb, x_ptb, y_ptb, u_ptb, v_ptb = eom_2d_perturb(ptr_path, [0.025, 0.025], 1e-8, 100)
    
    
    # 2. Test data
    mesh_path = './data/impact_echo/plane_echo/triangle_mesh.npy'
    mesh = np.load(mesh_path)
    mesh = torch.from_numpy(mesh).float()

    xx = mesh.T[0][:, None]
    yy = mesh.T[1][:, None]
    tt = torch.zeros_like(xx).float()


    # Define networks
    common_input = torch.tensor([3]) # t, x, y

    dist_hidden = torch.ones(3, dtype=int)*30
    dist_output = torch.tensor([5]) # u, v, s11, s22, s12

    par_hidden = torch.ones(3, dtype=int)*20
    # par_hidden = torch.tensor([12, 24, 24, 12])
    par_output = torch.tensor([4]) # u, v, ut, vt

    gen_hidden = torch.ones(6, dtype=int)*140
    gen_output = torch.tensor([7]) # u, v, ut, vt, s11, s22, s12

    dist_net = EchoNet(common_input, dist_hidden, dist_output)
    par_net = EchoNet(common_input, par_hidden, par_output)
    gen_net = EchoNet(common_input, gen_hidden, gen_output)

    # Train parameters in common
    loss_func = nn.MSELoss()
    
    # 1. particular network
    part_epochs = 10000
    part_lr = 1.4e-6
    part_optimizer = optim.Adam(par_net.parameters(), lr=part_lr)
    # part_scheduler = optim.lr_scheduler.LinearLR(part_optimizer, start_factor=1.0, end_factor=0.005, total_iters=5000)
    part_train_loss = []
    test_error = []

    for epoch in tqdm(range(part_epochs)):
        field_init_est = par_net(t_init, x_init, y_init)
        ic_loss = (field_init_est**2).mean(axis=0).sum()
        
        field_bound_est = par_net(t_bound, x_bound, y_bound)
        bc_loss = (field_bound_est[:, :2]**2).mean(axis=0).sum()

        part_loss = 10*ic_loss + 10*bc_loss

        part_loss.backward()
        part_optimizer.step()
        # part_scheduler.step()
        if (epoch+1)%200 == 0: print(ic_loss.item(), bc_loss.item())

        part_train_loss.append(ic_loss.item() + bc_loss.item())

        with torch.no_grad():
            # u, v, ut, vt = par_net(testt_ic, testx_ic, testy_ic).T
            u, v, ut, vt = par_net(tt, xx, yy).T
            error = (u**2 + v**2 + ut**2 + vt**2).mean()

            test_error.append(error.item())


    # 2. Distance network


    # 3. PDE network






    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(part_train_loss, label='total loss', color='blue')
    ax[0].legend()
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('mean squared error')
    ax[0].set_yscale('log')

    ax[1].plot(test_error, label='discrepancy', color='blue')
    ax[1].legend()
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('mean squared error')
    ax[1].set_yscale('log')

    plt.show()



