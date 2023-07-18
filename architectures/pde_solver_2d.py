import torch
import torch.nn as nn
from torch.autograd import grad

import numpy as np



def time_derivative_network(x, y, model):
    t = nn.Parameter(torch.zeros_like(x))
    x = nn.Parameter(x)
    y = nn.Parameter(y)
    
    u, v = model(t, x, y).T

    ut = grad(outputs=u.sum(), inputs=t, create_graph=True)[0]
    vt = grad(outputs=v.sum(), inputs=t, create_graph=True)[0]
    
    return ut, vt


def solid_mechanics_network(t, x, y, model, mat_info):
    
    rho = mat_info['density']
    elasticity = mat_info['elasticity']

    t = nn.Parameter(t)
    x = nn.Parameter(x)
    y = nn.Parameter(y)
    
    u, v = model(t, x, y).T
    
    eps_xx = grad(outputs=u.sum(), inputs=x, create_graph=True)[0]
    eps_yy = grad(outputs=v.sum(), inputs=y, create_graph=True)[0]
    eps_xy = 0.5*(grad(outputs=u.sum(), inputs=y, create_graph=True)[0] + grad(outputs=v.sum(), inputs=x, create_graph=True)[0])
    
    sig_xx = elasticity[0][0]*eps_xx+elasticity[0][1]*eps_yy+elasticity[0][3]*eps_xy
    sig_yy = elasticity[1][0]*eps_xx+elasticity[1][1]*eps_yy+elasticity[1][3]*eps_xy
    sig_xy = elasticity[3][0]*eps_xx+elasticity[3][1]*eps_yy+elasticity[3][3]*eps_xy

    sig_xx_x = grad(outputs=sig_xx.sum(), inputs=x, create_graph=True)[0]
    sig_xy_x = grad(outputs=sig_xy.sum(), inputs=x, create_graph=True)[0]
    sig_xy_y = grad(outputs=sig_xy.sum(), inputs=y, create_graph=True)[0]
    sig_yy_y = grad(outputs=sig_yy.sum(), inputs=y, create_graph=True)[0]
    
    ut = grad(outputs=u.sum(), inputs=t, create_graph=True)[0]
    utt = grad(outputs=ut.sum(), inputs=t, create_graph=True)[0]
    
    vt = grad(outputs=v.sum(), inputs=t, create_graph=True)[0]
    vtt = grad(outputs=vt.sum(), inputs=t, create_graph=True)[0]

    fu = rho*utt - sig_xx_x - sig_xy_y
    fv = rho*vtt - sig_xy_x - sig_yy_y
    
    return fu, fv


class EchoNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(EchoNet, self).__init__()

        layer_info = torch.cat((input, hidden, output))
        self.model = self.__gen_model(layer_info)
        

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


    def forward(self, t, x, y):
        grid = torch.concat([t,x,y], axis=1)
        u = self.model(grid)
        
        return u
    

def dD_dt_network(t, x, y, N2):
    t = nn.Parameter(t)
    x = nn.Parameter(x)
    y = nn.Parameter(y)

    d1, d2, d3, d4, d5 = N2(t, x, y).T

    f1 = grad(outputs=d1.sum(), inputs=t, create_graph=True)[0]
    f2 = grad(outputs=d2.sum(), inputs=t, create_graph=True)[0]
    f3 = grad(outputs=d3.sum(), inputs=t, create_graph=True)[0]
    f4 = grad(outputs=d4.sum(), inputs=t, create_graph=True)[0]
    f5 = grad(outputs=d5.sum(), inputs=t, create_graph=True)[0]

    return torch.concat([f1, f2, f3, f4, f5], axis=1)



def physics_informed_network(t, x, y, N3, info):
    rho = info['density']
    E = info['Young']
    nu = info['Poisson']

    t = nn.Parameter(t)
    x = nn.Parameter(x)
    y = nn.Parameter(y)

    # u0, v0, ut0, vt0 = N1(t, x, y).T
    # d = N2(t, x, y)
    # d = d.flatten()
    ug, vg, utg, vtg, sxx, syy, sxy = N3(t, x, y).T
    
    # u = u0 + d*ug
    # v = v0 + d*vg
    # ut = ut0 + d*utg
    # vt = vt0 + d*vtg

    # ---------------- this maybe problematic
    strain_xx = grad(outputs=ug.sum(), inputs=x, create_graph=True)[0]
    strain_yy = grad(outputs=vg.sum(), inputs=y, create_graph=True)[0]
    strain_xy = 0.5*(grad(outputs=ug.sum(), inputs=y, create_graph=True)[0] + grad(outputs=vg.sum(), inputs=x, create_graph=True)[0])
    
    # see https://github.com/Raocp/PINN-elastodynamics/blob/master/ElasticWaveConfined/ElasticWave.py#L304
    coef = E/((1+nu)*(1-2*nu))
    s_xx = coef*(1-nu)*strain_xx + coef*nu*strain_yy
    s_yy = coef*nu*strain_xx + coef*(1-nu)*strain_yy
    s_xy = E/(2*(1+nu))*strain_xy
    #----------------------------------------

    f_sxx = sxx[:, None] - s_xx
    f_syy = syy[:, None] - s_yy
    f_sxy = sxy[:, None] - s_xy
    
    u_t = grad(outputs=ug.sum(), inputs=t, create_graph=True)[0]
    v_t = grad(outputs=vg.sum(), inputs=t, create_graph=True)[0]

    f_ut = u_t - utg[:, None]
    f_vt = v_t - vtg[:, None]
    
    sxx_x = grad(outputs=sxx.sum(), inputs=x, create_graph=True)[0]
    syy_y = grad(outputs=syy.sum(), inputs=y, create_graph=True)[0]
    sxy_x = grad(outputs=sxy.sum(), inputs=x, create_graph=True)[0]
    sxy_y = grad(outputs=sxy.sum(), inputs=y, create_graph=True)[0]

    u_tt = grad(outputs=u_t.sum(), inputs=t, create_graph=True)[0]
    v_tt = grad(outputs=v_t.sum(), inputs=t, create_graph=True)[0]

    f_u = rho*u_tt - sxx_x - sxy_y
    f_v = rho*v_tt - syy_y - sxy_x
    
    return torch.concat([f_u, f_v, f_ut, f_vt, f_sxx, f_syy, f_sxy], axis=1)