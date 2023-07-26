import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from tqdm import tqdm
from glob import glob

from architectures.utils import eom_2d_perturb, eom_2d_distance, MeshDataset
from architectures.pde_solver_2d import EchoNet, physics_informed_network, dD_dt_network


if '__main__' == __name__:

    torch.set_printoptions(precision=8) # set precision point 8

    # Material info: assume that linear, isotropic & homogeneous
    info = {}
    info['density'] = 7.85e3
    info['Young'] = 2.05e5 # scaled by Mpa
    info['Poisson'] = 0.28

    # Generate data
    # 1. Training for particular data
    run_time = 12e-6
    Npts = 20000
    Ncol = 240000

    lb = np.array([0, 0, 0])
    ub = np.array([0.05, 0.05, 12e-6])

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
    mesh_data = MeshDataset(grid_col)
    mini_batch = 10000

    mesh_loader = DataLoader(mesh_data, batch_size=mini_batch)
    
    ptb_path = './data/impact_echo/impact_profile/Han_pulse.npy'
    N_axis = 25
    t_ptb, x_ptb, y_ptb, uv_ptb = eom_2d_perturb(ptb_path, [0.025, 0.025], 1e-8, N_axis)
    
    # for i in range(121):
    #     target = 1e-6 * i

    #     mask = np.where(t_ptb==target, True, False)
    #     t_slice = t_ptb[mask]
    #     x_slice = x_ptb[mask]
    #     y_slice = y_ptb[mask]
    #     uv_ = np.sqrt((uv_ptb**2).sum(axis=1))[:, None]
    #     uv_slice = uv_[mask]

    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     ax.scatter(x_slice, y_slice, c=uv_slice, vmin=-5e-10, vmax=5e-10)
    #     plt.savefig('./models/echo/'+'check_'+str(i)+'.png')
    #     plt.close()
    # plt.show()

    # 2. Training for collocation data
    t_col, x_col, y_col = grid_col.T
    t_col = t_col[:, None]
    x_col = x_col[:, None]
    y_col = y_col[:, None]

   
    # 3. Training for distance data
    distance = eom_2d_distance(grid_col, 0.05, 0.05)
    
    # 4. Test data
    mesh_path = './data/impact_echo/plane_echo/triangle_mesh.npy'
    mesh = np.load(mesh_path)
    mesh = torch.from_numpy(mesh).float()

    xx = mesh.T[0][:, None]
    yy = mesh.T[1][:, None]
    tt = torch.zeros_like(xx).float()


    # Define networks
    common_input = torch.tensor([3]) # t, x, y

    dist_hidden = torch.ones(6, dtype=int)*30
    dist_output = torch.tensor([5]) # distance of all physical variables, u, v, s11, s22, s12

    par_hidden = torch.ones(6, dtype=int)*20
    # par_hidden = torch.tensor([12, 24, 24, 12])
    par_output = torch.tensor([4]) # u, v, ut, vt

    gen_hidden = torch.ones(12, dtype=int)*160
    gen_output = torch.tensor([7]) # u, v, ut, vt, s11, s22, s12

    dist_net = EchoNet(common_input, dist_hidden, dist_output)
    par_net = EchoNet(common_input, par_hidden, par_output)
    gen_net = EchoNet(common_input, gen_hidden, gen_output)

    # Train parameters in common
    loss_func = nn.MSELoss()
    
    # 1. particular network
    part_epochs = 3000
    part_lr = 1.2e-4
    part_optimizer = optim.Adam(par_net.parameters(), lr=part_lr)
    # part_scheduler = optim.lr_scheduler.LinearLR(part_optimizer, start_factor=1.0, end_factor=0.005, total_iters=5000)
    part_train_loss = []
    ic_train = []
    bc_train = []
    ptb_train = []

    test_error = []

    for epoch in tqdm(range(part_epochs)):
        part_optimizer.zero_grad()

        field_init_est = par_net(t_init, x_init, y_init)
        ic_loss = (field_init_est**2).mean(axis=0).sum()
        
        field_bound_est = par_net(t_bound, x_bound, y_bound)
        bc_loss = (field_bound_est[:, :2]**2).mean(axis=0).sum()

        field_ptb_est = par_net(t_ptb, x_ptb, y_ptb)
        ptb_loss = loss_func(field_ptb_est[:, :2], uv_ptb) 

        part_loss = ic_loss + bc_loss + ptb_loss

        part_loss.backward()
        part_optimizer.step()
        # part_scheduler.step()
        if (epoch+1)%100 == 0: print(ic_loss.item(), bc_loss.item(), ptb_loss.item())

        part_train_loss.append(part_loss.item())
        ic_train.append(ic_loss.item())
        bc_train.append(bc_loss.item())
        ptb_train.append(ptb_loss.item())

        # with torch.no_grad():
        #     # u, v, ut, vt = par_net(testt_ic, testx_ic, testy_ic).T
        #     u, v, ut, vt = par_net(tt, xx, yy).T
        #     error = (u**2 + v**2 + ut**2 + vt**2).mean()

        #     test_error.append(error.item())


    model_path = './models/echo/'
    torch.save(par_net.state_dict(), model_path+'par_net')

    # 2. Distance network
    dist_epochs = 1500
    dist_lr = 1.2e-5 # 1e-5
    dist_optimizer = optim.Adam(dist_net.parameters(), lr=dist_lr)
    # part_scheduler = optim.lr_scheduler.LinearLR(part_optimizer, start_factor=1.0, end_factor=0.005, total_iters=5000)
    dist_train_loss = []
    
    for epoch in tqdm(range(dist_epochs)):
        dist_optimizer.zero_grad()

        dist_est = dist_net(t_col, x_col, y_col)
        dist_loss = loss_func(dist_est, distance)

        dt_dist = dD_dt_network(t_col, x_col, y_col, dist_net)
        dt_dist_loss = (dt_dist**2).mean(axis=0).sum()

        sum_dist_loss = dist_loss + dt_dist_loss
        sum_dist_loss.backward()
        dist_optimizer.step()
        # part_scheduler.step()
        if (epoch+1)%100 == 0: print(dist_loss.item(), dt_dist_loss.item())

        dist_train_loss.append(sum_dist_loss.item())

    torch.save(dist_net.state_dict(),  model_path+'dist_net')
        
    # 3. PDE network
    par_net.eval() # freeze network for particular solution
    dist_net.eval() # freeze network for distance

    gen_epochs = 2000
    gen_lr = 0.1e-7
    gen_optimizer = optim.Adam(gen_net.parameters(), lr=gen_lr)
    gen_train_loss = []

    for epoch in tqdm(range(gen_epochs)):
        gen_optimizer.zero_grad()
        
        batch_gen_loss = 0
        for grid_batch in tqdm(mesh_loader):
            tc, xc, yc = grid_batch.T
            tc = tc[:, None]
            xc = xc[:, None]
            yc = yc[:, None]

            f_val = physics_informed_network(tc, xc, yc, gen_net, info)
            
            gen_loss = (f_val**2).mean(axis=0).sum()
            
            gen_loss.backward()
            gen_optimizer.step()
            batch_gen_loss += gen_loss.item()
            if (epoch+1)%100 == 0 :
                print(f_val[:, 0])
                print(f_val[:, 1])
                print(f_val[:, 2])
                print(f_val[:, 3])
                print(f_val[:, 4])
                print(f_val[:, 5])
                print(f_val[:, 6])

        gen_train_loss.append(batch_gen_loss)

    torch.save(gen_net.state_dict(),  model_path+'gen_net')


    fig, ax = plt.subplots(1, 3, figsize=(21, 6))

    ax[0].plot(part_train_loss, label='particular loss', color='blue')
    ax[0].plot(ic_train, label='ic loss', color='red')
    ax[0].plot(bc_train, label='bc loss', color='black')
    ax[0].plot(ptb_train, label='ptb loss', color='orange')

    ax[0].legend(loc='upper right')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('mean squared error')
    ax[0].set_yscale('log')

    ax[1].plot(dist_train_loss, label='dist loss', color='blue')
    ax[1].legend()
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('mean squared error')
    ax[1].set_yscale('log')

    ax[2].plot(gen_train_loss, label='gen loss', color='blue')
    ax[2].legend()
    ax[2].set_xlabel('epoch')
    ax[2].set_ylabel('mean squared error')
    ax[2].set_yscale('log')

    # plt.show()
    plt.savefig(model_path+'test.png')




