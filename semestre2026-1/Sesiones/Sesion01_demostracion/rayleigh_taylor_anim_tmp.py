import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NX, NY = 128, 256
DT = 0.02
ITER = 40
VISC = 2e-4
DIFF = 1.5e-4
GRAVITY = 8.0
RHO_HEAVY, RHO_LIGHT = 1.6, 1.0
RHO_REF = 0.5 * (RHO_HEAVY + RHO_LIGHT)
TOTAL_FRAMES = 600
Lx, Ly = 0.25, 1.0
SAVE_ANIMATION = False
OUTPUT = "rayleigh_taylor.mp4"

def set_bnd(b, x):
    x[0, 1:-1] = -x[1, 1:-1] if b == 1 else x[1, 1:-1]
    x[NX + 1, 1:-1] = -x[NX, 1:-1] if b == 1 else x[NX, 1:-1]
    x[1:-1, 0] = -x[1:-1, 1] if b == 2 else x[1:-1, 1]
    x[1:-1, NY + 1] = -x[1:-1, NY] if b == 2 else x[1:-1, NY]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, NY + 1] = 0.5 * (x[1, NY + 1] + x[0, NY])
    x[NX + 1, 0] = 0.5 * (x[NX, 0] + x[NX + 1, 1])
    x[NX + 1, NY + 1] = 0.5 * (x[NX, NY + 1] + x[NX + 1, NY])

def lin_solve(b, x, x0, a, c):
    for _ in range(ITER):
        x[1:-1, 1:-1] = (
            x0[1:-1, 1:-1]
            + a * (
                x[2:, 1:-1] + x[:-2, 1:-1]
                + x[1:-1, 2:] + x[1:-1, :-2]
            )
        ) / c
        set_bnd(b, x)

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * NX * NY
    lin_solve(b, x, x0, a, 1 + 4 * a)

def advect(b, d, d0, u, v, dt):
    dt0x = dt * NX
    dt0y = dt * NY
    for i in range(1, NX + 1):
        for j in range(1, NY + 1):
            x = i - dt0x * u[i, j]
            y = j - dt0y * v[i, j]
            x = min(max(0.5, x), NX + 0.5)
            y = min(max(0.5, y), NY + 0.5)
            i0, j0 = int(np.floor(x)), int(np.floor(y))
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1.0 - s1, 1.0 - t1
            d[i, j] = (
                s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
                + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
            )
    set_bnd(b, d)

def project(u, v, p, div):
    div[1:-1, 1:-1] = -0.5 * (
        u[2:, 1:-1] - u[:-2, 1:-1] +
        v[1:-1, 2:] - v[1:-1, :-2]
    )
    p.fill(0.0)
    set_bnd(0, div)
    lin_solve(0, p, div, 1, 4)
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    set_bnd(1, u)
    set_bnd(2, v)

def initialize_density():
    x = np.linspace(0, Lx, NX + 2)
    y = np.linspace(-Ly / 2, Ly / 2, NY + 2)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    interface = 0.05 * np.cos(6 * np.pi * xx / Lx) + 0.1
    base = np.where(yy > interface, RHO_HEAVY, RHO_LIGHT)
    noise = 0.01 * (np.random.rand(*base.shape) - 0.5)
    return base + noise

class RayleighTaylorSolver:
    def __init__(self):
        self.u = np.zeros((NX + 2, NY + 2))
        self.v = np.zeros_like(self.u)
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        self.density = initialize_density()
        self.density_prev = np.zeros_like(self.density)
        self.pressure = np.zeros_like(self.density)
        self.divergence = np.zeros_like(self.density)

    def add_buoyancy(self):
        self.v[1:-1, 1:-1] += DT * GRAVITY * (self.density[1:-1, 1:-1] - RHO_REF) / RHO_REF
        set_bnd(2, self.v)

    def step(self):
        self.add_buoyancy()

        self.u_prev[:] = self.u
        diffuse(1, self.u, self.u_prev, VISC, DT)
        self.v_prev[:] = self.v
        diffuse(2, self.v, self.v_prev, VISC, DT)
        project(self.u, self.v, self.pressure, self.divergence)

        self.u_prev[:] = self.u
        self.v_prev[:] = self.v
        advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, DT)
        advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, DT)
        project(self.u, self.v, self.pressure, self.divergence)

        self.density_prev[:] = self.density
        diffuse(0, self.density, self.density_prev, DIFF, DT)
        self.density_prev[:] = self.density
        advect(0, self.density, self.density_prev, self.u, self.v, DT)

    def field(self):
        return self.density[1:-1, 1:-1].T

def main():
    np.random.seed(2)
    solver = RayleighTaylorSolver()

    fig, ax = plt.subplots(figsize=(4, 8))
    extent = (0, Lx, -Ly / 2, Ly / 2)
    img = ax.imshow(solver.field(), origin="lower", extent=extent, cmap="plasma", vmin=RHO_LIGHT, vmax=RHO_HEAVY)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    time_text = ax.text(0.02, 0.95, "", color="white", transform=ax.transAxes)

    def update(frame):
        solver.step()
        img.set_data(solver.field())
        time_text.set_text(f"t = {frame * DT:.2f}")
        return img, time_text

    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=30, blit=False)
    if SAVE_ANIMATION:
        ani.save(OUTPUT, fps=30, dpi=200)
        print(f"Animación guardada en {OUTPUT}")
    plt.show()

if __name__ == "__main__":
    main()
