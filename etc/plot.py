import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from architectures.pde_solver_2d import EchoNet


if __name__ == '__main__':
    net_path = './models/echonet_train_epoch_1500'
    mesh_path = './data/impact_echo/plane_echo/triangle_mesh.npy'

    mesh = np.load(mesh_path)
    mesh = torch.from_numpy(mesh).float()

    input_node = torch.tensor([3])
    hidden_node = torch.ones(8, dtype=int) * 20
    output_node = torch.tensor([2])
    
    pde_net = EchoNet(input_node, hidden_node, output_node)
    pde_net.load_state_dict(torch.load(net_path))
    pde_net.eval()
    print(len(mesh))
    x, y = mesh.T
    x = x[:, None]
    y = y[:, None]
    
    unit = 4e-8
    t = torch.ones_like(x)*unit

    triangles = tri.Triangulation(x.flatten(), y.flatten())
    
    # for i in range(301):
    #     u_sim, v_sim = pde_net(t*i, x, y).T
    #     u_sim = u_sim.detach().numpy()
    #     v_sim = v_sim.detach().numpy()
    #     intensity = np.sqrt(u_sim**2 + v_sim**2)
        
    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     tpcc = ax.tripcolor(triangles, intensity, shading='flat')
    #     fig.colorbar(tpcc)
    #     ax.set_title('field_intensity_'+str(i))

    #     plt.savefig('./images/model_wave/timeslice_'+str(i)+'.png')
    #     plt.close()

    # plt.show()


    response = []
    tt = np.linspace(0, 12e-6, 301)
    for i in range(301):
        echou, echov = pde_net(torch.tensor([[1]])*unit*i, torch.tensor([[0.025]]), torch.tensor([[0.035]])).T
        response.append(echov.detach().numpy())

    plt.plot(tt, response)
    plt.xlabel('time')
    plt.ylabel('y displacement')
    plt.show()
    




    