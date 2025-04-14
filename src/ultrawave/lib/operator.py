from devito import *
import numpy as np
from devito.builtins import initialize_function

def Acoustic2DOperator(model, source=None, reciever=None, eta_1=None, eta_2=None, tau_1=None, tau_2=None):
    x, y = model.grid.dimensions

    p = TimeFunction(name='p', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    v1x = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    v1y = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    rho1x = TimeFunction(name='rhox', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    rho1y = TimeFunction(name='rhoy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    dt = model.critical_dt
    indices = np.floor(source.coordinates.data[0,:] / model.spacing).astype(int)
    c0 = model.vp.data[indices[0], indices[1]] # sound speed of the first source point
    src_rhox = source.inject(field=rho1x.forward, expr=source * (2 * dt / (2 * c0 * model.spacing[0])))
    src_rhoy = source.inject(field=rho1y.forward, expr=source * (2 * dt / (2 * c0 * model.spacing[0])))
    rec_term = reciever.interpolate(expr=p)

    p_dx = getattr(p, 'd%s' % p.space_dimensions[0].name)
    p_dy = getattr(p, 'd%s' % p.space_dimensions[1].name)
    vx_dx = getattr(v1x.forward, 'd%s' % v1x.space_dimensions[0].name)
    vy_dy = getattr(v1y.forward, 'd%s' % v1y.space_dimensions[1].name)

    eq_v_x = Eq(v1x.forward, v1x - dt * p_dx / model.rho, subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(v1y.forward, v1y - dt * p_dy / model.rho, subdomain=model.grid.subdomains['main'])
    eq_rho_x = Eq(rho1x.forward, rho1x - dt * model.rho * vx_dx, subdomain=model.grid.subdomains['main'])
    eq_rho_y = Eq(rho1y.forward, rho1y - dt * model.rho * vy_dy, subdomain=model.grid.subdomains['main'])
    eq_p = Eq(p.forward, model.vp * model.vp * (rho1x.forward + rho1y.forward))  # , subdomain=grid.subdomains['main'])

    alpha_max = 2 * 1540 / model.spacing[0]

    # Damping parameterisation
    d_l = (1 - x / model.nbl) ** 4  # Left side
    d_r = (1 - (model.grid.shape[0] - 1 - x) / model.nbl) ** 4  # Right side
    d_t = (1 - y / model.nbl) ** 4  # Top side
    d_b = (1 - (model.grid.shape[1] - 1 - y) / model.nbl) ** 4  # Base edge
    # staggered
    d_l_s = (1 - (x + 0.5) / model.nbl) ** 4  # Left side
    d_r_s = (1 - (model.grid.shape[0] - 1 - (x + 0.5)) / model.nbl) ** 4  # Right side
    d_t_s = (1 - (y + 0.5) / model.nbl) ** 4  # Top side
    d_b_s = (1 - (model.grid.shape[1] - 1 - (y + 0.5)) / model.nbl) ** 4  # Base edge

    # for the PML domain
    eq_v_damp_left_x = Eq(v1x.forward, (1 - dt * d_l_s * alpha_max) * v1x - dt * p_dx / model.rho,
                          subdomain=model.grid.subdomains['left'])
    eq_rho_damp_left_x = Eq(rho1x.forward, (1 - dt * d_l * alpha_max) * rho1x - dt * model.rho * vx_dx,
                            subdomain=model.grid.subdomains['left'])

    eq_v_damp_right_x = Eq(v1x.forward, (1 - dt * d_r_s * alpha_max) * v1x - dt * p_dx / model.rho,
                           subdomain=model.grid.subdomains['right'])
    eq_rho_damp_right_x = Eq(rho1x.forward, (1 - dt * d_r * alpha_max) * rho1x - dt * model.rho * vx_dx,
                             subdomain=model.grid.subdomains['right'])

    eq_v_damp_top_y = Eq(v1y.forward, (1 - dt * d_t_s * alpha_max) * v1y - dt * p_dy / model.rho,
                         subdomain=model.grid.subdomains['top'])
    eq_rho_damp_top_y = Eq(rho1y.forward, (1 - dt * d_t * alpha_max) * rho1y - dt * model.rho * vy_dy,
                           subdomain=model.grid.subdomains['top'])

    eq_v_damp_base_y = Eq(v1y.forward, (1 - dt * d_b_s * alpha_max) * v1y - dt * p_dy / model.rho,
                          subdomain=model.grid.subdomains['base'])
    eq_rho_damp_base_y = Eq(rho1y.forward, (1 - dt * d_b * alpha_max) * rho1y - dt * model.rho * vy_dy,
                            subdomain=model.grid.subdomains['base'])

    if eta_1 is None:
        op = Operator([eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_v_damp_left_x, eq_rho_damp_left_x,
                       eq_v_damp_right_x, eq_rho_damp_right_x,
                       eq_v_damp_top_y, eq_rho_damp_top_y,
                       eq_v_damp_base_y, eq_rho_damp_base_y] + src_rhox + src_rhoy + rec_term)
    else:
        eq_p = Eq(p.forward,
                  model.vp * model.vp * (1 + eta_1 / 2 + eta_2 / 2) ** 2 * (rho1x.forward + rho1y.forward) - S1 - S2)
        eq_S1 = Eq(S1.forward,
                   S1 - dt * (S1 / tau_1 - eta_1 / (tau_1 * (1 + eta_1 / 2 + eta_2 / 2) ** 2) * (p + S1 + S2)))
        eq_S2 = Eq(S2.forward,
                   S2 - dt * (S2 / tau_2 - eta_2 / (tau_2 * (1 + eta_1 / 2 + eta_2 / 2) ** 2) * (p + S1 + S2)))
        op = Operator([eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_S1, eq_S2,
                       eq_v_damp_left_x, eq_rho_damp_left_x,
                       eq_v_damp_right_x, eq_rho_damp_right_x,
                       eq_v_damp_top_y, eq_rho_damp_top_y,
                       eq_v_damp_base_y, eq_rho_damp_base_y] + src_rhox + src_rhoy + rec_term)

    return op


def Acoustic3DOperator(model, source=None, reciever=None):
    x, y, z = model.grid.dimensions

    p = TimeFunction(name='p', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    v1x = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    v1y = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    v1z = TimeFunction(name='vz', grid=model.grid, time_order=1, space_order=model.space_order, staggered=z)
    rho1x = TimeFunction(name='rhox', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    rho1y = TimeFunction(name='rhoy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    rho1z = TimeFunction(name='rhoz', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    dt = model.critical_dt
    indices = np.floor(source.coordinates.data[0, :] / model.spacing).astype(int)
    c0 = model.vp.data[indices[0], indices[1], indices[2]]  # sound speed of the first source point
    src_rhox = source.inject(field=rho1x.forward, expr=source * (2 * dt / (3 * c0 * model.spacing[0])))
    src_rhoy = source.inject(field=rho1y.forward, expr=source * (2 * dt / (3 * c0 * model.spacing[0])))
    src_rhoz = source.inject(field=rho1z.forward, expr=source * (2 * dt / (3 * c0 * model.spacing[0])))
    rec_term = reciever.interpolate(expr=p)

    p_dx = getattr(p, 'd%s' % p.space_dimensions[0].name)
    p_dy = getattr(p, 'd%s' % p.space_dimensions[1].name)
    p_dz = getattr(p, 'd%s' % p.space_dimensions[2].name)
    vx_dx = getattr(v1x.forward, 'd%s' % v1x.space_dimensions[0].name)  # (x0=shift_x0(shift, x, None, 0))
    vy_dy = getattr(v1y.forward, 'd%s' % v1y.space_dimensions[1].name)  # (x0=shift_x0(shift, y, None, 1))
    vz_dz = getattr(v1z.forward, 'd%s' % v1z.space_dimensions[2].name)

    dt = model.critical_dt
    eq_v_x = Eq(v1x.forward, v1x - dt * p_dx / model.rho, subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(v1y.forward, v1y - dt * p_dy / model.rho, subdomain=model.grid.subdomains['main'])
    eq_v_z = Eq(v1z.forward, v1z - dt * p_dz / model.rho, subdomain=model.grid.subdomains['main'])
    eq_rho_x = Eq(rho1x.forward, rho1x - dt * model.rho * vx_dx, subdomain=model.grid.subdomains['main'])
    eq_rho_y = Eq(rho1y.forward, rho1y - dt * model.rho * vy_dy, subdomain=model.grid.subdomains['main'])
    eq_rho_z = Eq(rho1z.forward, rho1z - dt * model.rho * vz_dz, subdomain=model.grid.subdomains['main'])
    eq_p = Eq(p.forward, model.vp * model.vp * (
                rho1x.forward + rho1y.forward + rho1z.forward))  # , subdomain=grid.subdomains['main'])

    alpha_max = 2 * 1540 / model.spacing[0]

    # Damping parameterization
    d_l = (1 - x / model.nbl) ** 4  # Left side
    d_r = (1 - (model.grid.shape[0] - 1 - x) / model.nbl) ** 4  # Right side
    d_f = (1 - y / model.nbl) ** 4  # Front side
    d_b = (1 - (model.grid.shape[1] - 1 - y) / model.nbl) ** 4  # Back side
    d_t = (1 - z / model.nbl) ** 4  # Top side
    d_bz = (1 - (model.grid.shape[2] - 1 - z) / model.nbl) ** 4  # Base side

    # staggered
    d_l_s = (1 - (x + 0.5) / model.nbl) ** 4  # Left side
    d_r_s = (1 - (model.grid.shape[0] - 1 - (x + 0.5)) / model.nbl) ** 4  # Right side
    d_f_s = (1 - (y + 0.5) / model.nbl) ** 4  # Front side
    d_b_s = (1 - (model.grid.shape[1] - 1 - (y + 0.5)) / model.nbl) ** 4  # Back side
    d_t_s = (1 - (z + 0.5) / model.nbl) ** 4  # Top side
    d_bz_s = (1 - (model.grid.shape[2] - 1 - (z + 0.5)) / model.nbl) ** 4  # Base side

    # for the PML domain
    eq_v_damp_left_x = Eq(v1x.forward, (1 - dt * d_l_s * alpha_max) * v1x - dt * p_dx / model.rho,
                          subdomain=model.grid.subdomains['left'])
    eq_rho_damp_left_x = Eq(rho1x.forward, (1 - dt * d_l * alpha_max) * rho1x - dt * model.rho * vx_dx,
                            subdomain=model.grid.subdomains['left'])

    eq_v_damp_right_x = Eq(v1x.forward, (1 - dt * d_r_s * alpha_max) * v1x - dt * p_dx / model.rho,
                           subdomain=model.grid.subdomains['right'])
    eq_rho_damp_right_x = Eq(rho1x.forward, (1 - dt * d_r * alpha_max) * rho1x - dt * model.rho * vx_dx,
                             subdomain=model.grid.subdomains['right'])

    eq_v_damp_front_y = Eq(v1y.forward, (1 - dt * d_f_s * alpha_max) * v1y - dt * p_dy / model.rho,
                           subdomain=model.grid.subdomains['front'])
    eq_rho_damp_front_y = Eq(rho1y.forward, (1 - dt * d_f * alpha_max) * rho1y - dt * model.rho * vy_dy,
                             subdomain=model.grid.subdomains['front'])

    eq_v_damp_back_y = Eq(v1y.forward, (1 - dt * d_b_s * alpha_max) * v1y - dt * p_dy / model.rho,
                          subdomain=model.grid.subdomains['back'])
    eq_rho_damp_back_y = Eq(rho1y.forward, (1 - dt * d_b * alpha_max) * rho1y - dt * model.rho * vy_dy,
                            subdomain=model.grid.subdomains['back'])

    eq_v_damp_top_z = Eq(v1z.forward, (1 - dt * d_t_s * alpha_max) * v1z - dt * p_dz / model.rho,
                         subdomain=model.grid.subdomains['top'])
    eq_rho_damp_top_z = Eq(rho1z.forward, (1 - dt * d_t * alpha_max) * rho1z - dt * model.rho * vz_dz,
                           subdomain=model.grid.subdomains['top'])

    eq_v_damp_base_z = Eq(v1z.forward, (1 - dt * d_bz_s * alpha_max) * v1z - dt * p_dz / model.rho,
                          subdomain=model.grid.subdomains['base'])
    eq_rho_damp_base_z = Eq(rho1z.forward, (1 - dt * d_bz * alpha_max) * rho1z - dt * model.rho * vz_dz,
                            subdomain=model.grid.subdomains['base'])

    op = Operator([eq_v_x, eq_v_y, eq_v_z, eq_rho_x, eq_rho_y, eq_rho_z, eq_p,
                   eq_v_damp_left_x, eq_rho_damp_left_x,
                   eq_v_damp_right_x, eq_rho_damp_right_x,
                   eq_v_damp_front_y, eq_rho_damp_front_y,
                   eq_v_damp_back_y, eq_rho_damp_back_y,
                   eq_v_damp_top_z, eq_rho_damp_top_z,
                   eq_v_damp_base_z, eq_rho_damp_base_z]
                  + src_rhox + src_rhoy + src_rhoz + rec_term)
    return op

def Elastic2DOperator(model, source=None, reciever=None):
    # No subdomain, slower than with subdomains, but stronger attenuation effect
    # first order (valid)
    x, y = model.grid.dimensions

    vx = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    vy = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    s_xx = TimeFunction(name='sxx', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_yy = TimeFunction(name='syy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_xy = TimeFunction(name='sxy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    vx_dx = getattr(vx.forward, 'd%s' % vx.space_dimensions[0].name)
    vx_dy = getattr(vx.forward, 'd%s' % vx.space_dimensions[1].name)
    vy_dx = getattr(vy.forward, 'd%s' % vy.space_dimensions[0].name)
    vy_dy = getattr(vy.forward, 'd%s' % vy.space_dimensions[1].name)

    s_xx_dx = getattr(s_xx, 'd%s' % s_xx.space_dimensions[0].name)
    s_xx_dy = getattr(s_xx, 'd%s' % s_xx.space_dimensions[1].name)
    s_yy_dx = getattr(s_yy, 'd%s' % s_yy.space_dimensions[0].name)
    s_yy_dy = getattr(s_yy, 'd%s' % s_yy.space_dimensions[1].name)
    s_xy_dx = getattr(s_xy, 'd%s' % s_xy.space_dimensions[0].name)
    s_xy_dy = getattr(s_xy, 'd%s' % s_xy.space_dimensions[1].name)

    src_sxx = source.inject(field=s_xx.forward, expr= -source)
    src_syy = source.inject(field=s_yy.forward, expr= -source)
    rec_term = reciever.interpolate(expr=-(s_xx + s_yy) / 2)  # -(s_xx+s_yy)/2

    alpha_max = 2 * 1540 / model.spacing[0]

    # Damping parameterisation
    # PML
    xx, yy = np.ogrid[0:(model.shape[0] + 2 * model.nbl), 0:(model.shape[1] + 2 * model.nbl)]
    d_l = (1 - xx / model.nbl) ** 4
    d_r = (1 - (model.shape[0] + 2 * model.nbl - 1 - xx) / model.nbl) ** 4
    d_x = d_l * (np.heaviside(xx, 1) - np.heaviside(xx - model.nbl, 1)) + d_r * (
                np.heaviside(xx - (model.shape[0] + 2 * model.nbl - model.nbl), 1) - np.heaviside(xx - (model.shape[0] + 2 * model.nbl), 1))
    d_x = np.repeat(d_x, model.shape[1] + 2 * model.nbl, axis=1)

    d_t = (1 - yy / model.nbl) ** 4  # top side
    d_b = (1 - (model.shape[1] + 2 * model.nbl - 1 - yy) / model.nbl) ** 4  # Base side
    d_y = d_t * (np.heaviside(yy, 1) - np.heaviside(yy - model.nbl, 1)) + d_b * (
                np.heaviside(yy - (model.shape[1] + 2 * model.nbl - model.nbl), 1) - np.heaviside(yy - (model.shape[1] + 2 * model.nbl), 1))
    d_y = np.repeat(d_y, model.shape[0] + 2 * model.nbl, axis=0)

    pml_x = Function(name='pml_x', grid=model.grid, space_order=model.space_order, parameter=True)
    initialize_function(pml_x, d_x, model.nbl)
    pml_x.data[:] = d_x
    pml_y = Function(name='pml_y', grid=model.grid, space_order=model.space_order, parameter=True)
    initialize_function(pml_y, d_y, model.nbl)
    pml_y.data[:] = d_y

    dt = model.critical_dt
    eq_v_x = Eq(vx.forward, (1 - dt * pml_x * alpha_max) * vx + dt / model.rho * (s_xx_dx + s_xy_dy),
                subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(vy.forward, (1 - dt * pml_y * alpha_max) * vy + dt / model.rho * (s_xy_dx + s_yy_dy),
                subdomain=model.grid.subdomains['main'])
    eq_s_xx = Eq(s_xx.forward, (1 - dt * pml_x * alpha_max) * s_xx + dt * (
                (model.lam + 2 * model.mu) * vx_dx + model.lam * vy_dy), subdomain=model.grid.subdomains['main'])
    eq_s_yy = Eq(s_yy.forward, (1 - dt * pml_y * alpha_max) * s_yy + dt * (
                (model.lam + 2 * model.mu) * vy_dy + model.lam * vx_dx), subdomain=model.grid.subdomains['main'])
    eq_s_xy = Eq(s_xy.forward,
                 (1 - dt * (pml_x + pml_y) * alpha_max) * s_xy + dt * model.mu * (vx_dy + vy_dx),
                 subdomain=model.grid.subdomains['main'])

    op = Operator([eq_v_x, eq_v_y, eq_s_xx, eq_s_yy, eq_s_xy] + src_sxx + src_syy + rec_term)
    return op

def Elastic3DOperator(model, source=None, reciever=None):
    # No subdomain, slower than with subdomains, but stronger attenuation effect
    # first order (valid)
    x, y, z = model.grid.dimensions

    vx = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    vy = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    vz = TimeFunction(name='vz', grid=model.grid, time_order=1, space_order=model.space_order, staggered=z)
    s_xx = TimeFunction(name='sxx', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_yy = TimeFunction(name='syy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_zz = TimeFunction(name='szz', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_xy = TimeFunction(name='sxy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_xz = TimeFunction(name='sxz', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    s_yz = TimeFunction(name='syz', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    vx_dx = getattr(vx.forward, 'd%s' % vx.space_dimensions[0].name)
    vx_dy = getattr(vx.forward, 'd%s' % vx.space_dimensions[1].name)
    vx_dz = getattr(vx.forward, 'd%s' % vx.space_dimensions[2].name)
    vy_dx = getattr(vy.forward, 'd%s' % vy.space_dimensions[0].name)
    vy_dy = getattr(vy.forward, 'd%s' % vy.space_dimensions[1].name)
    vy_dz = getattr(vy.forward, 'd%s' % vy.space_dimensions[2].name)
    vz_dx = getattr(vz.forward, 'd%s' % vz.space_dimensions[0].name)
    vz_dy = getattr(vz.forward, 'd%s' % vz.space_dimensions[1].name)
    vz_dz = getattr(vz.forward, 'd%s' % vz.space_dimensions[2].name)

    s_xx_dx = getattr(s_xx, 'd%s' % s_xx.space_dimensions[0].name)
    s_xx_dy = getattr(s_xx, 'd%s' % s_xx.space_dimensions[1].name)
    s_xx_dz = getattr(s_xx, 'd%s' % s_xx.space_dimensions[2].name)
    s_yy_dx = getattr(s_yy, 'd%s' % s_yy.space_dimensions[0].name)
    s_yy_dy = getattr(s_yy, 'd%s' % s_yy.space_dimensions[1].name)
    s_yy_dz = getattr(s_yy, 'd%s' % s_yy.space_dimensions[2].name)
    s_zz_dx = getattr(s_zz, 'd%s' % s_zz.space_dimensions[0].name)
    s_zz_dy = getattr(s_zz, 'd%s' % s_zz.space_dimensions[1].name)
    s_zz_dz = getattr(s_zz, 'd%s' % s_zz.space_dimensions[2].name)
    s_xy_dx = getattr(s_xy, 'd%s' % s_xy.space_dimensions[0].name)
    s_xy_dy = getattr(s_xy, 'd%s' % s_xy.space_dimensions[1].name)
    s_xy_dz = getattr(s_xy, 'd%s' % s_xy.space_dimensions[2].name)
    s_xz_dx = getattr(s_xz, 'd%s' % s_xz.space_dimensions[0].name)
    s_xz_dy = getattr(s_xz, 'd%s' % s_xz.space_dimensions[1].name)
    s_xz_dz = getattr(s_xz, 'd%s' % s_xz.space_dimensions[2].name)
    s_yz_dx = getattr(s_yz, 'd%s' % s_yz.space_dimensions[0].name)
    s_yz_dy = getattr(s_yz, 'd%s' % s_yz.space_dimensions[1].name)
    s_yz_dz = getattr(s_yz, 'd%s' % s_yz.space_dimensions[2].name)

    src_sxx = source.inject(field=s_xx.forward, expr=source)
    src_syy = source.inject(field=s_yy.forward, expr=source)
    src_szz = source.inject(field=s_zz.forward, expr=source)
    rec_term = reciever.interpolate(expr=-(s_xx + s_yy + s_zz) / 3)  # -(s_xx+s_yy)/2

    alpha_max = 2 * 1540 / model.spacing[0]
    # PML
    xx, yy, zz = np.ogrid[0:(model.shape[0] + 2 * model.nbl), 0:(model.shape[1] + 2 * model.nbl), 0:(model.shape[2] + 2 * model.nbl)]
    d_l = (1 - xx / model.nbl) ** 4
    d_r = (1 - (model.shape[0] + 2 * model.nbl - 1 - xx) / model.nbl) ** 4
    d_x = d_l * (np.heaviside(xx, 1) - np.heaviside(xx - model.nbl, 1)) + d_r * (
                np.heaviside(xx - (model.shape[0] + 2 * model.nbl - model.nbl), 1) - np.heaviside(xx - (model.shape[0] + 2 * model.nbl), 1))
    d_x = np.repeat(d_x, model.shape[1] + 2 * model.nbl, axis=1)
    d_x = np.repeat(d_x, model.shape[2] + 2 * model.nbl, axis=2)

    d_f = (1 - yy / model.nbl) ** 4  # front side
    d_b = (1 - (model.shape[1] + 2 * model.nbl - 1 - yy) / model.nbl) ** 4  # Back side
    d_y = d_f * (np.heaviside(yy, 1) - np.heaviside(yy - model.nbl, 1)) + d_b * (
                np.heaviside(yy - (model.shape[1] + 2 * model.nbl - model.nbl), 1) - np.heaviside(yy - (model.shape[1] + 2 * model.nbl), 1))
    d_y = np.repeat(d_y, model.shape[0] + 2 * model.nbl, axis=0)
    d_y = np.repeat(d_y, model.shape[2] + 2 * model.nbl, axis=2)

    d_t = (1 - zz / model.nbl) ** 4  # top side
    d_bz = (1 - (model.shape[2] + 2 * model.nbl - 1 - zz) / model.nbl) ** 4  # Base side
    d_z = d_t * (np.heaviside(zz, 1) - np.heaviside(zz - model.nbl, 1)) + d_bz * (
                np.heaviside(zz - (model.shape[2] + 2 * model.nbl - model.nbl), 1) - np.heaviside(zz - (model.shape[2] + 2 * model.nbl), 1))
    d_z = np.repeat(d_z, model.shape[0] + 2 * model.nbl, axis=0)
    d_z = np.repeat(d_z, model.shape[1] + 2 * model.nbl, axis=1)

    pml_x = Function(name='pml_x', grid=model.grid, space_order=model.space_order, parameter=True)
    initialize_function(pml_x, d_x, model.nbl)
    pml_x.data[:] = d_x
    pml_y = Function(name='pml_y', grid=model.grid, space_order=model.space_order, parameter=True)
    initialize_function(pml_y, d_y, model.nbl)
    pml_y.data[:] = d_y
    pml_z = Function(name='pml_z', grid=model.grid, space_order=model.space_order, parameter=True)
    initialize_function(pml_z, d_z, model.nbl)
    pml_z.data[:] = d_z

    dt = model.critical_dt
    eq_v_x = Eq(vx.forward, (1 - dt * pml_x * alpha_max) * vx + dt / model.rho * (s_xx_dx + s_xy_dy + s_xz_dz),
                subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(vy.forward, (1 - dt * pml_y * alpha_max) * vy + dt / model.rho * (s_xy_dx + s_yy_dy + s_yz_dz),
                subdomain=model.grid.subdomains['main'])
    eq_v_z = Eq(vz.forward, (1 - dt * pml_z * alpha_max) * vz + dt / model.rho * (s_xz_dx + s_yz_dy + s_zz_dz),
                subdomain=model.grid.subdomains['main'])
    eq_s_xx = Eq(s_xx.forward, (1 - dt * pml_x * alpha_max) * s_xx + dt * (
                (model.lam + 2 * model.mu) * vx_dx + model.lam * (vy_dy + vz_dz)),
                 subdomain=model.grid.subdomains['main'])
    eq_s_yy = Eq(s_yy.forward, (1 - dt * pml_y * alpha_max) * s_yy + dt * (
                (model.lam + 2 * model.mu) * vy_dy + model.lam * (vx_dx + vz_dz)),
                 subdomain=model.grid.subdomains['main'])
    eq_s_zz = Eq(s_zz.forward, (1 - dt * pml_z * alpha_max) * s_zz + dt * (
                (model.lam + 2 * model.mu) * vz_dz + model.lam * (vx_dx + vy_dy)),
                 subdomain=model.grid.subdomains['main'])
    eq_s_xy = Eq(s_xy.forward,
                 (1 - dt * (pml_x + pml_y) * alpha_max) * s_xy + dt * model.mu * (vx_dy + vy_dx),
                 subdomain=model.grid.subdomains['main'])
    eq_s_xz = Eq(s_xz.forward,
                 (1 - dt * (pml_x + pml_z) * alpha_max) * s_xz + dt * model.mu * (vx_dz + vz_dx),
                 subdomain=model.grid.subdomains['main'])
    eq_s_yz = Eq(s_yz.forward,
                 (1 - dt * (pml_y + pml_z) * alpha_max) * s_yz + dt * model.mu * (vy_dz + vz_dy),
                 subdomain=model.grid.subdomains['main'])

    op = Operator([eq_v_x, eq_v_y, eq_v_z, eq_s_xx, eq_s_yy, eq_s_zz, eq_s_xy, eq_s_xz,
                   eq_s_yz] + src_sxx + src_syy + src_szz + rec_term)
    return op