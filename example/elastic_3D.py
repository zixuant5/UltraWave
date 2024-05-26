import numpy as np
import matplotlib.pyplot as plt

from ultrawave import TimeAxis, Receiver, ToneBurstSource
from ultrawave.lib.model_3d import Model
from ultrawave.lib.operator import Elastic3DOperator

from devito import configuration
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'
configuration['language'] = 'openacc'

# Define grid spacing in [m]
spacing = (5.e-5, 5.e-5, 5.e-5) # [m]

# The number of grid points in each dimension
shape = (int(5e-3/spacing[0]),
         int(5e-3/spacing[1]),
         int(3.5e-3/spacing[2]))

# Define the origin coordinate, which is the top left corner of the grid
origin = (0., 0., 0.)

# Compressional wave speed
vp_background = 1500  # Background speed in [m/s].
vp = np.full(shape, vp_background, dtype=np.float32)

# Shear wave speed
vs_background = 0  # Background speed in [m/s].
vs = np.full(shape, vs_background, dtype=np.float32)

# Density
rho_background = 1000  # Background density in [kg/m^3].
rho = np.full(shape, rho_background, dtype=np.float32)

# Define a spherical scatterer.
r = int(0.25e-3 / spacing[0])  # Radius of the sphere in grid points.
center_x, center_y, center_z = (shape[0] // 2, shape[1] // 2, int(2.5e-3/spacing[2])) # Center of the sphere.
x, y, z = np.ogrid[-center_x:shape[0]-center_x,
                   -center_y:shape[1]-center_y,
                   -center_z:shape[2]-center_z]

mask = (x**2 + y**2 + z**2 <= r**2)
vp_scatter = 4030     # Compressional wave speed of scatterer in [m/s].
vs_scatter = 1645     # Shear wave speed of scatterer in [m/s].
rho_scatter = 1960     # Density of scatterer in [kg/m^3].
vp[mask] = vp_scatter
vs[mask] = vs_scatter
rho[mask] = rho_scatter

time_order = 2
space_order = 4
nbl = 30  # Number of boundary layers. Size of PML

# Define simulation time.
t0 = 0.0  # Start time of the simulation.
tn = 35.e-6  # End time of the simulation [s].
dt = 6e-10  # Time step [s].
time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt = time_range.num

# Create a model
model = Model(vp=vp, rho=rho, origin=origin, shape=shape, spacing=spacing, space_order=space_order, dt=dt, nbl=nbl)

# Define a planar source.
f0 = 2e6 # Central frequency in [Hz]
src_npoints = shape[0] * shape[1]
src = ToneBurstSource(name='src', grid=model.grid, f0=f0, cycles=3, npoint=src_npoints, time_range=time_range) # source signal is a 3-cycle tone burst

# Define source coordinates
x = np.arange(0., 5.e-3, spacing[0])
y = np.arange(0., 5.e-3, spacing[0])
xv, yv = np.meshgrid(x, y)
src.coordinates.data[:, 0] = np.reshape(xv, -1)
src.coordinates.data[:, 1] = np.reshape(yv, -1)
src.coordinates.data[:, 2] = 1.e-3  # Position in the z-direction.

# Define a point receiver
rec = Receiver(name='rec', grid=model.grid, npoint=1, time_range=time_range)
rec.coordinates.data[:, 0] = 5.e-3/2
rec.coordinates.data[:, 1] = 5.e-3/2
rec.coordinates.data[:, 2] = 1.e-3

# Define the operator
op = Elastic3DOperator(model, source=src, reciever=rec)

# Run the operator
op(time=time_range.num-1, dt=dt)