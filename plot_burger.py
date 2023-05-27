import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy

from architectures.utils import BurgerDataset
from architectures.pde_solver_1d import BurgerNet


if __name__ == "__main__":
    net_path = './burgernet_train_epoch10000'
    
    input_node = torch.tensor([2])
    hidden_node = torch.ones(8, dtype=int) * 20
    output_node = torch.tensor([1])
    
    pde_net = BurgerNet(input_node, hidden_node, output_node)
    pde_net.load_state_dict(torch.load(net_path))
    pde_net.eval()

    path = 'data/raissi/burgers_shock.mat'

    burger_data_1d = BurgerDataset(path)
    domain = burger_data_1d.X_star
    usol = burger_data_1d.u_star

    # data = scipy.io.loadmat('data/raissi/burgers_shock.mat')
    
    # t = data['t'].flatten()[:,None]
    # x = data['x'].flatten()[:,None]
    # Exact = np.real(data['usol']).T
    
    # X, T = np.meshgrid(x,t)
    
    # fig,ax=plt.subplots(1,1, figsize=(15, 5))
    # cp = ax.contourf(T, X, Exact, cmap='seismic')
    # fig.colorbar(cp) # Add a colorbar to a plot
    # ax.set_title('u(t,x)')
    # ax.set_xlabel('time')
    # ax.set_ylabel('x')

    # plt.show()



    fig, ax = plt.subplots(2,2, figsize=(12, 12))

    unit = 256
    time_list = [1, 30, 55, 89]

    for idx, i in enumerate(time_list):
        k = idx//2
        l = idx%2
        
        x = domain[unit*(i-1):unit*i, 0][:, None]
        t = domain[unit*(i-1):unit*i, 1][:, None]
        u_pred_01 = pde_net(t, x).detach()
        
        u = usol[unit*(i-1):unit*i].flatten()
        u_pred_01 = u_pred_01.flatten()
        
        ax[k][l].plot(x, u, 'blue', label='exact solution')
        ax[k][l].plot(x, u_pred_01, ls='--', color='red', label='predicted solution')
        ax[k][l].set_xlabel('x')
        ax[k][l].set_ylabel('u')
        ax[k][l].grid()
        ax[k][l].set_title('time at {} s'.format(0.01*(i-1)))
        ax[k][l].legend()

plt.show()
