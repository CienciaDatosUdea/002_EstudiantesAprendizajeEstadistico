"""
Rayleigh-Taylor instability simulation using Stable Fluids method
2D incompressible Navier-Stokes with Boussinesq approximation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─── Grid and simulation parameters ───────────────────────────────────────────
NX, NY   = 128, 256          # cells (excluding ghost)
LX, LY   = 0.55, 1.0        # domain size
DX, DY   = LX / NX, LY / NY
DT       = 0.01             # time step
VISC     = 1e-4              # kinematic viscosity
DIFF     = 1e-4              # scalar diffusivity
BUOY     = 4.0               # buoyancy coefficient (g * Δρ/ρ₀)
N_ITER   = 40                # Gauss-Seidel iterations for pressure
FRAMES   = 300
STEPS_PER_FRAME = 4

# ─── Arrays (with 1 ghost cell on each side → size NX+2, NY+2) ────────────────
shape = (NX + 2, NY + 2)

u  = np.zeros(shape)   # x-velocity
v  = np.zeros(shape)   # y-velocity
p  = np.zeros(shape)   # pressure
rho = np.zeros(shape)  # density perturbation  (+1 heavy, -1 light)

u0 = np.zeros(shape)
v0 = np.zeros(shape)
rho0 = np.zeros(shape)

# ─── Helper index ranges (interior) ───────────────────────────────────────────
I  = slice(1, NX + 1)
J  = slice(1, NY + 1)

ix = np.arange(1, NX + 1)   # shape (NX,)
jy = np.arange(1, NY + 1)   # shape (NY,)

# Physical coordinates of cell centres
xc = (ix - 0.5) * DX        # shape (NX,)
yc = (jy - 0.5) * DY        # shape (NY,)
XC, YC = np.meshgrid(xc, yc, indexing='ij')   # (NX, NY)

# ─── Initialise density ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
pert = 0.05 * np.sin(2 * np.pi * XC / LX) + 0.01 * rng.standard_normal((NX, NY))
interface = LY / 2 + 0.03 * pert

# heavy (+1) above interface, light (-1) below
rho[I, J] = np.where(YC > interface, 1.0, -1.0)

# ─── Boundary conditions (no-slip walls, periodic in x) ──────────────────────
def apply_bc(u, v, rho):
    # Periodic in x
    u[0,  :] = u[NX, :]
    u[NX+1,:] = u[1,  :]
    v[0,  :] = v[NX, :]
    v[NX+1,:] = v[1,  :]
    rho[0,  :] = rho[NX, :]
    rho[NX+1,:] = rho[1,  :]

    # No-slip top & bottom walls
    u[:, 0]    = -u[:, 1]
    u[:, NY+1] = -u[:, NY]
    v[:, 0]    = 0.0
    v[:, NY+1] = 0.0
    rho[:, 0]    = rho[:, 1]
    rho[:, NY+1] = rho[:, NY]

# ─── Semi-Lagrangian advection ─────────────────────────────────────────────────
def advect(q, u, v, dt):
    """Back-trace cell centres and interpolate bilinearly."""
    # Velocity at cell centres (average from faces – here u,v are cell-centred)
    uc = 0.5 * (u[I, J] + u[I, J])   # already cell-centred for simplicity
    vc = 0.5 * (v[I, J] + v[I, J])

    # Back-trace
    xb = XC - dt * u[I, J]
    yb = YC - dt * v[I, J]

    # Clamp / wrap x (periodic), clamp y
    xb = xb % LX
    yb = np.clip(yb, 0.5 * DX, LY - 0.5 * DY)

    # Convert to fractional grid index (0-based interior)
    fi = xb / DX - 0.5          # fractional index in x
    fj = yb / DY - 0.5          # fractional index in y

    i0 = np.floor(fi).astype(int) % NX
    j0 = np.floor(fj).astype(int)
    j0 = np.clip(j0, 0, NY - 1)
    i1 = (i0 + 1) % NX
    j1 = np.clip(j0 + 1, 0, NY - 1)

    s = fi - np.floor(fi)
    t = fj - np.floor(fj)

    # Gather interior values (shift by 1 for ghost offset)
    Q = q[1:NX+1, 1:NY+1]
    q00 = Q[i0, j0]
    q10 = Q[i1, j0]
    q01 = Q[i0, j1]
    q11 = Q[i1, j1]

    return (1-s)*((1-t)*q00 + t*q01) + s*((1-t)*q10 + t*q11)

# ─── Diffusion (explicit) ─────────────────────────────────────────────────────
def diffuse(q, coeff, dt):
    lap = (
        q[2:NX+2, 1:NY+1] + q[0:NX,   1:NY+1]  # x neighbours
      + q[1:NX+1, 2:NY+2] + q[1:NX+1, 0:NY  ]  # y neighbours
      - 4 * q[1:NX+1, 1:NY+1]
    ) / (DX * DY)   # rough isotropic Laplacian
    return q[I, J] + coeff * dt * lap

# ─── Pressure projection ──────────────────────────────────────────────────────
def project(u, v):
    div = np.zeros(shape)
    div[I, J] = (
        (u[2:NX+2, J] - u[0:NX, J]) / (2 * DX) +
        (v[I, 2:NY+2] - v[I, 0:NY]) / (2 * DY)
    )

    p[:] = 0.0
    for _ in range(N_ITER):
        p[I, J] = (
            (p[2:NX+2, J] + p[0:NX,   J]) * DY**2 +
            (p[I, 2:NY+2] + p[I, 0:NY  ]) * DX**2 -
            div[I, J] * DX**2 * DY**2
        ) / (2 * (DX**2 + DY**2))
        # Periodic x
        p[0,  :] = p[NX, :]
        p[NX+1,:] = p[1,  :]
        # Neumann y
        p[:, 0]    = p[:, 1]
        p[:, NY+1] = p[:, NY]

    # Subtract gradient
    u[I, J] -= (p[2:NX+2, J] - p[0:NX, J]) / (2 * DX)
    v[I, J] -= (p[I, 2:NY+2] - p[I, 0:NY]) / (2 * DY)

# ─── One time step ────────────────────────────────────────────────────────────
def step(u, v, rho, dt):
    apply_bc(u, v, rho)

    # Advect velocities
    u[I, J] = advect(u, u, v, dt)
    v[I, J] = advect(v, u, v, dt)
    # Advect density
    rho[I, J] = advect(rho, u, v, dt)

    apply_bc(u, v, rho)

    # Diffuse velocities
    u[I, J] = diffuse(u, VISC, dt)
    v[I, J] = diffuse(v, VISC, dt)
    rho[I, J] = diffuse(rho, DIFF, dt)

    apply_bc(u, v, rho)

    # Buoyancy: upward force proportional to density perturbation
    v[I, J] -= BUOY * rho[I, J] * dt

    apply_bc(u, v, rho)

    # Project to enforce incompressibility
    project(u, v)
    apply_bc(u, v, rho)

# ─── Animation ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3, 8), facecolor='black')
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.axis('off')

img = ax.imshow(
    rho[I, J].T,
    origin='lower',
    extent=[0, LX, 0, LY],
    vmin=-1, vmax=1,
    cmap='RdBu_r',
    interpolation='bilinear',
    aspect='auto'
)

time_text = ax.text(
    0.02, 0.97, 't = 0.000',
    transform=ax.transAxes,
    color='white', fontsize=9,
    va='top', ha='left'
)

plt.tight_layout(pad=0.2)
sim_time = [0.0]

def update(frame):
    for _ in range(STEPS_PER_FRAME):
        step(u, v, rho, DT)
        sim_time[0] += DT
    img.set_data(rho[I, J].T)
    time_text.set_text(f't = {sim_time[0]:.3f}')
    return img, time_text

ani = animation.FuncAnimation(
    fig, update,
    frames=FRAMES,
    interval=30,
    blit=True
)

plt.show()
