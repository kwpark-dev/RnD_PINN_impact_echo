import torch
import torch.nn as nn
from torch.autograd import grad, Variable

import numpy as np



def burger_1d_mse_loss(u_pred, u_sol, f_pred=None):

    loss = torch.mean((u_sol - u_pred)**2)
    
    if f_pred is not None:
        loss = torch.mean((u_sol - u_pred)**2) + torch.mean(f_pred**2)

    return loss


def physics_informed_network(domain, model, mode=True):
    
    x = domain[:, 0][:, None]
    t = domain[:, 1][:, None]
    x.requires_grad = True
    t.requires_grad = True

    u = model(t, x)
    if not mode:
        u.requires_grad = True

    ut = grad(outputs=u.sum(), inputs=t, create_graph=True)[0]
    ux = grad(outputs=u.sum(), inputs=x, create_graph=True)[0]
    uxx = grad(outputs=ux.sum(), inputs=x, create_graph=True)[0]

    f = ut + u*ux - 0.01/np.pi*uxx
    
    return f


class BurgerNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(BurgerNet, self).__init__()

        layer_info = torch.cat((input, hidden, output))
        consts = nn.Parameter(torch.randn(2))

        self.model = self.__gen_model(layer_info)
        self.lamb01 = consts[0]
        self.lamb02 = consts[1]
        

    def __gen_model(self, layer_info):
        layers = []

        for i in range(len(layer_info)-1):

            if i == len(layer_info)-2:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                
            else:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                # layers.append(nn.BatchNorm1d(layer_info[i+1]))

        return nn.Sequential(*layers)


    def forward(self, t, x):
        grid = torch.concat([t,x], axis=1)
        u = self.model(grid)
        
        return u
    

    def print_expr(self):

        print('u_t + uu_x = C u_xx')


    def count_model_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())

        return total_params




class LinearStringNet(nn.Module):
    def __init__(self):
        pass