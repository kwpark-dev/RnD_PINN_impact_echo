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


def perturbation(path, v_init, scale):
    data = np.load(path)

    t = data[:, 0]
    u = data[:, 1]*scale
    N = len(t)
    print(N)

    up = np.array([np.trapz(u[:i]) for i in range(N)]) + v_init

    return t, u, up    