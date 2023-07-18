import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from architectures.pde_solver_1d import BurgerNet, physics_informed_network
from architectures.utils import BurgerDataset, burger_random_ic, burger_random_bc, burger_collocation


if __name__ == "__main__":
    path = 'data/raissi/burgers_shock.mat'
    nu = 0.01/np.pi

    burger_data_1d = BurgerDataset(path)
    domain = burger_data_1d.X_star

    x_ic, t_ic, u_ic = burger_random_ic(-1, 1, 100) # has 100 pts
    x_bc, t_bc, u_bc = burger_random_bc(0, 1, 100)  # has 100 pts
    domain_col = burger_collocation([-1, 1], [0, 1], 100) # has 10000 pts
    
    mini_batch = 5000
    test_loader = DataLoader(burger_data_1d, batch_size=mini_batch, shuffle=True)

    input_node = torch.tensor([2])
    hidden_node = torch.ones(8, dtype=int) * 20
    output_node = torch.tensor([1])
    
    pde_net = BurgerNet(input_node, hidden_node, output_node)

    epochs = 2000
    lr = 3e-4
    optimizer = optim.Adam(pde_net.parameters(), lr=lr)
    ic_loss_func = MSELoss()
    bc_loss_func = MSELoss()
    violation_func = MSELoss()

    loss_total_train = []
    ic_loss_train = []
    bc_loss_train = []
    violation_train = []

    sol_discrepancy = []
    
    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()
            
        u_ic_est = pde_net(t_ic, x_ic)
        u_bc_est = pde_net(t_bc, x_bc)
        f_est = physics_informed_network(domain_col, pde_net)

        ic_loss = ic_loss_func(u_ic, u_ic_est)
        bc_loss = bc_loss_func(u_bc, u_bc_est)
        violation = torch.mean(f_est**2)
        loss = ic_loss + bc_loss + violation

        loss.backward()
        optimizer.step()

        ic_loss_train.append(ic_loss.item())
        bc_loss_train.append(bc_loss.item())
        violation_train.append(violation.item())
        loss_total_train.append(loss.item())

        with torch.no_grad():
            batch_dscr = 0
            for i, sample in enumerate(test_loader):
                grid, u_sol = sample
                x = grid[:, 0][:, None]
                t = grid[:, 1][:, None]
                
                u_pred = pde_net(t, x)
                discrepancy = violation_func(u_sol, u_pred)
                batch_dscr += discrepancy.item()

            sol_discrepancy.append(batch_dscr)
        

    torch.save(pde_net.state_dict(), './burgernet_train_epoch10000')


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(loss_total_train, label='total loss', color='blue')
    ax[0].plot(ic_loss_train, ls='--', label='ic loss', color='red')
    ax[0].plot(bc_loss_train, ls='-.', label='bc loss', color='red')
    ax[0].plot(violation_train, ls=':', label='violation', color='red')
    ax[0].legend()
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('mean squared error')

    ax[1].plot(sol_discrepancy, label='discrepancy', color='blue')
    ax[1].legend()
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('mean squared error')
    
    plt.show()