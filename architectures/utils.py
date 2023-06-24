import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np


sine_pulse = lambda x: -np.sin(np.pi*x)


def burger_random_ic(lower, upper, nums, callback=sine_pulse):
    callback = np.vectorize(callback)
    
    x = np.random.uniform(lower, upper, (nums, 1))
    t = np.zeros_like(x)
    u = callback(x)

    x = torch.tensor(x).float()
    t = torch.tensor(t).float()
    u = torch.tensor(u).float()
    
    return x, t, u


def burger_random_bc(lower, upper, nums):
    x = np.random.choice([-1, 1], (nums, 1))
    t = np.random.uniform(lower, upper, (nums, 1))
    u = np.zeros_like(x)

    x = torch.tensor(x).float()
    t = torch.tensor(t).float()
    u = torch.tensor(u).float()

    return x, t, u


def burger_collocation(x_range, t_range, nums):
    x = np.linspace(x_range[0], x_range[1], nums)
    t = np.linspace(t_range[0], t_range[1], nums)

    X, T = np.meshgrid(x, t)
    domain = torch.from_numpy(np.hstack((X.flatten()[:,None], T.flatten()[:,None]))).float()
    
    return domain


class BurgerDataset(Dataset):
    def __init__(self, file_path):
        data =  loadmat(file_path)

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        usol = np.real(data['usol']).T
        
        X, T = np.meshgrid(x,t)
        
        self.X_star = torch.from_numpy(np.hstack((X.flatten()[:,None], T.flatten()[:,None]))).float()
        self.u_star = torch.from_numpy(usol.flatten()[:,None]).float()

        self.lb = self.X_star.min(axis=0) # lower bounds of x & t
        self.ub = self.X_star.max(axis=0) # upper bounds of x & t


    def __len__(self):
        
        return self.X_star.shape[0]


    def __getitem__(self, idx):
        
        return self.X_star[idx], self.u_star[idx]



class GridDataset(Dataset):
    def __init__(self, time, mesh):
        t, x, y = eom_2d_collocation(time, mesh)
        self.grid = torch.concat([t,x,y], axis=1)
        

    def __len__(self):
        
        return self.grid.shape[0]


    def __getitem__(self, idx):
        
        return self.grid[idx]


def perturbation(path, v_init, x_pos, y_pos, scale):
    data = np.load(path)

    t = torch.from_numpy(data[:, 0]).float()
    t = t[:, None]
    u = data[:, 1]*scale
    x = torch.ones_like(t)*x_pos
    y = torch.ones_like(t)*y_pos
    N = len(t)
    
    up = np.array([np.trapz(u[:i]) for i in range(N)]) + v_init
    up = up[:, None]

    return t, x.float(), y.float(), torch.from_numpy(up).float() 


def eom_2d_random_ic(xb, yb, val, size):
    t = torch.zeros((size,1))
    x = (xb[1]-xb[0])*torch.rand(size,1)+xb[0]
    y = (yb[1]-yb[0])*torch.rand(size,1)+yb[0]
    U = torch.zeros((size,2))+torch.from_numpy(val).float()
    
    return t, x, y, U


def eom_2d_random_bc(tb, xb, yb, val, size):
    size_ = int(size/4)
    size__ = int(size/2)

    t = (tb[1]-tb[0])*torch.rand(size,1)+tb[0]
    U = torch.zeros((size,2))+torch.from_numpy(val).float()

    binary = torch.from_numpy(np.random.choice(xb, (size,1))).float()
    randy = (yb[1]-yb[0])*torch.rand(size,1)+yb[0]
    x = torch.cat((binary[:size__], randy[:size__]))
    y = torch.cat((binary[size__:], randy[size__:]))

    return t, x, y, U


def eom_2d_collocation(time, mesh):
    tmp = torch.ones(len(mesh),1)
    t = (time*tmp).T
    t = t.flatten()[:, None]

    x, y = mesh.T
    tmp2 = torch.ones(len(time),1)
    x = (x*tmp2).flatten()[:, None]
    y = (y*tmp2).flatten()[:, None]

    return t, x, y


def eom_2d_ic_grid(xb, yb, size):
    x = torch.linspace(xb[0], xb[1], size)
    y = torch.linspace(yb[0], yb[1], size)
    
    x, y = torch.meshgrid(x,y, indexing='xy')
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    t = torch.zeros_like(x)

    return t.float(), x.float(), y.float()


def eom_2d_perturb(path, center, scale, N):
    data = np.load(path)

    t = data[:, 0]
    amp = data[:, 1]*scale

    mask = np.array(range(len(t)))
    mask = np.where(mask%3==0, True, False)
    mask[-1] = True # add the last point

    t = t[mask] # to reduce data by 1/3
    amp = amp[mask]

    x = np.linspace(0, 0.05, N, endpoint=True)
    y = np.linspace(0, 0.05, N, endpoint=True)

    xx, tt  = np.meshgrid(x, t)
    yy, _ = np.meshgrid(y, t)
    
    tt = tt.flatten()[:, None]
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    
    u = amp*(xx-center[0])
    v = amp*(yy-center[1])

    tt = torch.from_numpy(tt).float()
    xx = torch.from_numpy(xx).float()
    yy = torch.from_numpy(yy).float()
    u = torch.from_numpy(u).float()
    u = torch.from_numpy(v).float()

    return tt, xx, yy, u, v


def material_info(rho, nu):
    info = {}
    info['density'] = torch.tensor(rho).float()

    A = np.array([[1-nu, nu, nu],
                  [nu, 1-nu, nu],
                  [nu, nu, 1-nu]])
    B = np.diag([(1-2*nu)/2, (1-2*nu)/2, (1-2*nu)/2])
    Z = np.zeros_like(A)

    info['elasticity'] = torch.from_numpy(np.bmat([[A, Z], [Z, B]])).float()

    return info
