import torch
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors
import numpy as np
import scipy

from architectures.utils import BurgerDataset
from architectures.pde_solver_1d import BurgerNet


if __name__ == "__main__":
    net_path = './models/burgernet_train_epoch10000'
    
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

    data = scipy.io.loadmat('data/raissi/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    norm= matplotlib.colors.Normalize(vmin=-1, vmax=1)
    fig, ax = plt.subplots(2,1, figsize=(12, 16))
    
    tmp = np.linspace(0, 0.99, 10)

    ax[0].plot(tmp, np.zeros_like(tmp), ls=':', c='red', label='baseline')
    cp1 = ax[0].contourf(T, X, Exact, 100, cmap='jet')
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cp1.cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax[0]) # Add a colorbar to a plot
    ax[0].set_title('exact u(t,x)')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('x')
    ax[0].legend()

    pred = pde_net(torch.from_numpy(T.flatten()[:,None]).float(), torch.from_numpy(X.flatten()[:,None]).float())
    pred = pred.detach().numpy()
    pred = pred.reshape(X.shape)

    ax[1].plot(tmp, np.zeros_like(tmp), ls=':', c='red', label='baseline')
    cp2 = ax[1].contourf(T, X, pred, 100, cmap='jet')
    sm2 = plt.cm.ScalarMappable(norm=norm, cmap = cp2.cmap)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax[1]) # Add a colorbar to a plot
    ax[1].set_title('PINN u(t,x)')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('x')
    ax[1].legend()

    plt.show()



#     fig, ax = plt.subplots(2,2, figsize=(12, 12))

#     unit = 256
#     time_list = [1, 30, 55, 89]

#     for idx, i in enumerate(time_list):
#         k = idx//2
#         l = idx%2
        
#         x = domain[unit*(i-1):unit*i, 0][:, None]
#         t = domain[unit*(i-1):unit*i, 1][:, None]
#         u_pred_01 = pde_net(t, x).detach()
        
#         u = usol[unit*(i-1):unit*i].flatten()
#         u_pred_01 = u_pred_01.flatten()
        
#         ax[k][l].plot(x, u, 'blue', label='exact solution')
#         ax[k][l].plot(x, u_pred_01, ls='--', color='red', label='predicted solution')
#         ax[k][l].set_xlabel('x')
#         ax[k][l].set_ylabel('u')
#         ax[k][l].grid()
#         ax[k][l].set_title('time at {} s'.format(0.01*(i-1)))
#         ax[k][l].legend()

# plt.show()
