#!/usr/bin/env python

from qumba.solve import eq2
from qumba.isomorph import Tanner, search


def find_autos(Ax, Az):
    # find automorphisms of the XZ-Tanner graph
    ax, n = Ax.shape
    az, _ = Az.shape

    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Ax, Az)
    perms = []
    for g in search(U, V):
        # each perm acts on ax+az+n in that order
        assert len(g) == n+ax+az
        for i in range(ax):
            assert 0<=g[i]<ax
        for i in range(ax, ax+az):
            assert ax<=g[i]<ax+az
        g = [g[i]-ax-az for i in range(ax+az, ax+az+n)]
        perms.append(g)
    return perms


def find_zx_duality(Ax, Az):
    # Find a zx duality
    ax, n = Ax.shape
    az, _ = Az.shape
    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Az, Ax)
    for duality in search(U, V):
        break
    else:
        return None

    dual_x = [duality[i] for i in range(ax)]
    dual_z = [duality[i]-ax for i in range(ax,ax+az)]
    dual_n = [duality[i]-ax-az for i in range(ax+az,ax+az+n)]

    Ax1 = Ax[dual_z, :][:, dual_n]
    Az1 = Az[dual_x, :][:, dual_n]

    assert eq2(Ax1, Az)
    assert eq2(Az1, Ax)
    return dual_n



