import matplotlib.pyplot as plt

from architectures.utils import perturbation


if '__main__' == __name__:
    path = './data/impact_echo/impact_profile/Han_pulse.npy'

    t, impact, u_pscb = perturbation(path, 0, 1e-8)
    
    fig, ax = plt.subplots(1,2, figsize=(16, 6))
    ax[0].plot(t, impact, 'b')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('impact')
    ax[0].grid()

    ax[1].plot(t, u_pscb, 'b')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('prescribed displacement')
    ax[1].grid()

    plt.show()