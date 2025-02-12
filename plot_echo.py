import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio

from architectures.pde_solver_2d import EchoNet


if __name__ == '__main__':
    net_path = './models/echo/'
    mesh_path = './data/impact_echo/plane_echo/triangle_mesh.npy'

    mesh = np.load(mesh_path)
    mesh = torch.from_numpy(mesh).float()

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

    par_net.load_state_dict(torch.load(net_path+'par_net'))
    par_net.eval()

    dist_net.load_state_dict(torch.load(net_path+'dist_net'))
    dist_net.eval()

    gen_net.load_state_dict(torch.load(net_path+'gen_net'))
    gen_net.eval()

    x, y = mesh.T
    x = x[:, None]
    y = y[:, None]
    
    run_time = 12e-6
    unit = 4e-8
    t = torch.ones_like(x)*unit

    triangles = tri.Triangulation(x.flatten(), y.flatten())
    frames = []
    for i in range(301):
        u0, v0, _, _ = par_net(t*i, x, y).T
        d = dist_net(t*i, x, y)
        # d = d.flatten()
        ug, vg, _, _, _, _, _ = gen_net(t*i, x, y).T
        
        u = u0 + d.T[0]*ug
        v = v0 + d.T[1]*vg
                
        u = u.detach().numpy()
        v = v.detach().numpy()
        intensity = np.sqrt(u**2 + v**2)
        print(min(intensity), max(intensity))
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        tpcc = ax.tripcolor(triangles, intensity, shading='flat', vmin=3.6e-4, vmax=1.36e-3)
        fig.colorbar(tpcc)
        ax.set_title('field_intensity_'+str(i))

        plt.savefig('./models/echo/images/timeslice_'+str(i)+'.png')
        plt.close()

        frames.append(imageio.v2.imread(f'./models/echo/images/timeslice_'+str(i)+'.png'))

    imageio.mimsave('./models/echo/trained_echo.gif', frames, duration=5)


    # plt.show()


    # response = []
    # tt = np.linspace(0, 12e-6, 301)
    # for i in range(301):
    #     u0, v0, _, _ = par_net(torch.tensor([[1]])*unit*i, torch.tensor([[0.025]]), torch.tensor([[0.035]])).T
    #     d = dist_net(torch.tensor([[1]])*unit*i, torch.tensor([[0.025]]), torch.tensor([[0.035]]))
        
    #     ug, vg, _, _, _, _, _ = gen_net(torch.tensor([[1]])*unit*i, torch.tensor([[0.025]]), torch.tensor([[0.035]])).T
    #     u = u0 + d.T[0]*ug
    #     v = v0 + d.T[1]*vg

    #     # print(torch.tensor([[1]])*unit*i, d, u0)
    #     response.append(v.detach().numpy())

    # plt.plot(tt, response)
    # plt.xlabel('time')
    # plt.ylabel('v')
    # plt.savefig('./models/echo/impact_echo.png')
    # plt.show()
    




    