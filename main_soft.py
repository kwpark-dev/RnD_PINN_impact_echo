import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob

from architectures.utils import perturbation, eom_2d_random_ic, eom_2d_random_bc, \
                                material_info, GridDataset
from architectures.pde_solver_2d import EchoNet, solid_mechanics_network, time_derivative_network


if '__main__' == __name__:

    ptr_path = './data/impact_echo/impact_profile/Han_pulse.npy'
    mesh_path = './data/impact_echo/plane_echo/triangle_mesh.npy'
    field_path = './data/impact_echo/plane_echo/uv*.npy'

    t_pscb, x_pscb, y_pscb, v_pscb = perturbation(ptr_path, 0, x_pos=0.025, y_pos=0.025, scale=1e-8)
    print(len(t_pscb))
    # fig, ax = plt.subplots(1,2, figsize=(16, 6))
    # ax[0].plot(t, impact, 'b')
    # ax[0].set_xlabel('time')
    # ax[0].set_ylabel('impact')
    # ax[0].grid()

    # ax[1].plot(t, u_pscb, 'b')
    # ax[1].set_xlabel('time')
    # ax[1].set_ylabel('prescribed displacement')
    # ax[1].grid()

    # plt.show()

    mesh = np.load(mesh_path)
    mesh = torch.from_numpy(mesh).float()
    # plt.triplot(mesh.T[0], mesh.T[1])
    # plt.axis('square')
    # plt.show()
    
    size = 1000
    t_setp = 4e-8
    run_time = 12e-6
    
    xbound = np.array([0, 0.05])
    ybound = np.array([0, 0.05])
    tbound = np.array([0, run_time])
    bval = np.array([[0, 0]])
    ival = np.array([[0, 0]])
    
    t_ic, x_ic, y_ic, U_ic = eom_2d_random_ic(xbound, ybound, ival, size)
    t_bc, x_bc, y_bc, U_bc = eom_2d_random_bc(tbound, xbound, ybound, bval, size)
    time = torch.arange(0, run_time+t_setp, step=t_setp)
    # t_col, x_col, y_col = eom_2d_collocation(time, mesh)
    eom_data_2d = GridDataset(time, mesh)
    mini_batch = 30000
    collocation_loader = DataLoader(eom_data_2d, batch_size=mini_batch, shuffle=True)

    rho = 7.85e3 # density of steel
    nu = 0.28 # Poisson's ratio of steel

    info = material_info(rho, nu)

    input_node = torch.tensor([3])
    hidden_node = torch.ones(8, dtype=int) * 20
    output_node = torch.tensor([2])
    
    pde_net = EchoNet(input_node, hidden_node, output_node)

    epochs = 10
    lr = 0.001
    optimizer = optim.Adam(pde_net.parameters(), lr=lr)

    mse_loss_func = nn.MSELoss()

    loss_total_train = []
    ic_loss_train = []
    dudt_ic_loss_train = []
    bc_loss_train = []
    perturb_loss_train = []
    violation_train = []

    discrepancy_test = []

    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()
        # u_ic_est, v_ic_est = pde_net(t_ic, x_ic, y_ic).T
        # u_bc_est, v_bc_est = pde_net(t_bc, x_bc, y_bc).T
        # u_pscb_est, v_pscb_est = pde_net(t_pscb, x_pscb, y_pscb).T

        violation_loss = 0
        dtdu_loss = 0
        ic_loss = 0
        bc_loss = 0
        perturb_loss = 0
        for batch_grid in tqdm(collocation_loader):
            u_ic_est, v_ic_est = pde_net(t_ic, x_ic, y_ic).T
            u_bc_est, v_bc_est = pde_net(t_bc, x_bc, y_bc).T
            u_pscb_est, v_pscb_est = pde_net(t_pscb, x_pscb, y_pscb).T

            ic_loss_mini = (u_ic_est**2 + v_ic_est**2).mean()
            bc_loss_mini = (u_bc_est**2 + v_bc_est**2).mean()
            perturb_loss_mini = (u_pscb_est**2).mean() + mse_loss_func(v_pscb_est, v_pscb.flatten())
            
            # loss = ic_loss + bc_loss + perturb_loss

            t_col, x_col, y_col = batch_grid.T
            t_col = t_col[:, None]
            x_col = x_col[:, None]
            y_col = y_col[:, None]

            fu_est_mini, fv_est_mini = solid_mechanics_network(t_col, x_col, y_col, pde_net, info)
            dudt_est_mini, dvdt_est_mini = time_derivative_network(x_col, y_col, pde_net)

            violation_loss_mini = (fu_est_mini**2 + fv_est_mini**2).mean()
            dt_ic_loss_mini = (dudt_est_mini**2 + dvdt_est_mini**2).mean()

            loss_mini = violation_loss_mini + dt_ic_loss_mini + ic_loss_mini + bc_loss_mini + perturb_loss_mini

            violation_loss += violation_loss_mini.item()
            dtdu_loss += dt_ic_loss_mini.item()
            ic_loss += ic_loss_mini.item()
            bc_loss += bc_loss_mini.item()
            perturb_loss += perturb_loss_mini.item()

            loss_mini.backward()
            optimizer.step()
        
        loss_total_train.append(ic_loss + bc_loss + dtdu_loss + perturb_loss + violation_loss + dtdu_loss)
        ic_loss_train.append(ic_loss)
        bc_loss_train.append(bc_loss)
        dudt_ic_loss_train.append(dtdu_loss)
        perturb_loss_train.append(perturb_loss)
        violation_train.append(violation_loss)

        with torch.no_grad():

            discrepancy = 0
            xx = mesh.T[0][:, None]
            yy = mesh.T[1][:, None]
                
            for i, file in enumerate(glob(field_path)):
                tt = torch.ones_like(xx)*time[i]
                
                u_est = pde_net(tt, xx, yy)
                u_sim = torch.from_numpy(np.load(file)).float()

                disc = mse_loss_func(u_est, u_sim)
                discrepancy += disc.item()

            discrepancy_test.append(discrepancy)


fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(loss_total_train, label='total loss', color='blue')
ax[0].legend()
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('mean squared error')

ax[1].plot(discrepancy_test, label='discrepancy', color='blue')
ax[1].legend()
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('mean squared error')

plt.show()