import numpy as np
from glob import glob

unit = 4e-8
for idx, file in enumerate(glob('./data/from_simulation/dataset/uv_field*')):
    data = np.genfromtxt(file, skip_header=9)

    if idx == 0:
        np.save('./data/impact_echo/plane_echo/triangle_mesh', data[:, :2])

    field = data[:, 2:]
    _, m = field.shape

    flag = file.split('_')[-1]
    flag = flag.split('.')[0]

    if flag == 'last':
        time = 250

        for i in range(int(m/2)):
            time_slice = field[:, i:i+2]

            np.save('./data/impact_echo/plane_echo/uv_slice_'+str((time+i)), time_slice)

    else:
        time = int(flag)-50
        
        for i in range(int(m/2)):
            time_slice = field[:, i:i+2]

            np.save('./data/impact_echo/plane_echo/uv_slice_'+str((time+i)), time_slice)

