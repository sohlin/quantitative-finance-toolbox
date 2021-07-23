#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy.solvers import solve
from sympy import solveset, S
from sympy import Symbol


def ho_lee_interest(bond_yield, vol):
    """ Function for Ho-Lee model interest calibration.

    Args:
        bond_yield: np.array, ZCB yield for calibration.
        vol: float, volatility for interest rate.

    Returns:
        res: np.array, Ho and Lee interest tree.
    """
    # initialize tree
    res = np.zeros((len(bond_yield), len(bond_yield)))
    res[0][0] = bond_yield[0]
    # start from 1y and work forward
    for i in range(1, len(bond_yield)):
        queue = []
        # set the first node at each layer as x
        x = Symbol('x')
        for j in range(i + 1):
            if j == 0 or j == i:
                queue.append(1 / (1 + x - 2 * j * vol))
            else:
                queue.extend([1 / (1 + x - 2 * j * vol)] * 2)

        tmp = 0
        level = i
        # compute price from last period and work back
        while len(queue) != 1:
            if tmp == level:
                level -= 1
                tmp = 0
            approx = 0.5 * (queue.pop(0) + queue.pop(0)) / (1 + res[level - 1][tmp])
            queue.extend([approx] * (2 if tmp != 0 and level - tmp >= 2 else 1))
            tmp += 1

        approx = queue[0]
        # solve for x
        ans = solveset(approx - 1 / (bond_yield[i] + 1) ** (i + 1), x, domain=S.Reals).sup
        res[i, :i+1] = [ans - 2 * k * vol for k in range(i + 1)]

    return res


def ho_lee_price(int_tree, cf_tree, prepay=None):
    """ Function to get price of asset using Ho and Lee interest tree.

    Args:
        int_tree: np.array, interest tree with Ho and Lee model.
        cf_tree: np.array, cash flow tree for payments.
        prepay: np.array or None, if not None it gives prepayment at each period.

    Returns:
        tmp[0]: float, price of asset.
        -delta / tmp[0]: float, spot rate duration of asset.
    """
    pays = cf_tree[-1]
    for i in range(len(int_tree)-1, -1, -1):
        tmp = pays / (1 + int_tree[i, :i+1])
        if prepay is not None:
            tmp[tmp > prepay[i]] = prepay[i]
        # compute spot rate duration
        if len(tmp) == 2:
             delta_r = (tmp[0] - tmp[1]) / (int_tree[1][0] - int_tree[1][1])
        if i != 0:
            next_p = np.array([np.mean(tmp[j:j+2]) for j in range(len(tmp)-1)])
            # if prepay is not None:
            #     next_p[next_p > prepay[i]] = prepay[i]
            pays = next_p + cf_tree[i-1, :i]
    return tmp[0], -delta_r / tmp[0]


def ho_lee_price_poio(int_tree, cf_tree, po_tree, io_tree, prepay=None):
    """ Function to get price of asset using Ho and Lee interest tree.

        Args:
            int_tree: np.array, interest tree with Ho and Lee model.
            cf_tree: np.array, cash flow tree for payments.
            po_tree: np.array, cash flow tree for principal only payments.
            io_tree: np.array, cash flow tree for interest only payments.
            prepay: np.array or None, if not None it gives prepayment at each period.

        Returns:
            tmp_po[0]: float, price of asset (principal only).
            tmp_io[0]: float, price of asset (interest only).
            -delta_po / tmp_po[0]: float, spot rate duration of asset (principal only).
            -delta_io / tmp_io[0]: float, spot rate duration of asset (interest only).
    """
    pays = cf_tree[-1]
    pays_po, pays_io = po_tree[-1], io_tree[-1]

    for i in range(len(int_tree) - 1, -1, -1):
        tmp = pays / (1 + int_tree[i, :i + 1])
        tmp_po = pays_po / (1 + int_tree[i, :i + 1])
        tmp_io = pays_io / (1 + int_tree[i, :i + 1])
        if prepay is not None:
            mask = tmp > prepay[i]
            tmp[mask] = prepay[i]
            tmp_po[mask] = prepay[i]
            tmp_io[mask] = 0
        if len(tmp_po) == 2:
            delta_po = (tmp_po[0] - tmp_po[1]) / (int_tree[1][0] - int_tree[1][1])
            delta_io = (tmp_io[0] - tmp_io[1]) / (int_tree[1][0] - int_tree[1][1])
        if i != 0:
            next_p = np.array([np.mean(tmp[j:j + 2]) for j in range(len(tmp) - 1)])
            next_po = np.array([np.mean(tmp_po[j:j + 2]) for j in range(len(tmp_po) - 1)])
            next_io = np.array([np.mean(tmp_io[j:j + 2]) for j in range(len(tmp_io) - 1)])

            pays = next_p + cf_tree[i - 1, :i]
            pays_po = next_po + po_tree[i-1, :i]
            pays_io = next_io + io_tree[i-1, :i]
    return tmp_po[0], tmp_io[0], -delta_po / tmp_po[0], -delta_io / tmp_io[0]


def p_to_equal_c(p, r, comp_period, n, adj="end"):
    """ Given principal, compute equal coupon per period. """
    r /= comp_period
    s = 1 if adj == 'end' else 0
    coef = 1 / ((1 + r) ** s) * (1 - (1 / (1 + r)) ** n) / (1 - 1 / (1 + r))
    res = p / coef
    return res


def equal_c_to_p(c, r, comp_period, n, adj='end'):
    """ Given equal coupon per period, compute principal. """
    r /= comp_period
    s = 1 if adj == 'end' else 0
    coef = 1 / ((1 + r) ** s) * (1 - (1 / (1 + r)) ** n) / (1 - 1 / (1 + r))
    res = c * coef
    return res


if __name__ == '__main__':
    bond_yield = [0.05, 0.055, 0.057, 0.059, 0.06, 0.061]
    interest_tree = ho_lee_interest(bond_yield, 0.015)
    print("interest tree")
    print(interest_tree.T)

    # non-prepayable mortgage
    principal = 100
    r = 0.055
    comp_period = 1
    n = 6
    payments = p_to_equal_c(principal, r, comp_period, n, adj="end")
    print("payment at the end of each year")
    print(payments)
    cf_tree = np.tril(np.ones_like(interest_tree) * 20.0179, 0)
    print(cf_tree)
    price, duration = ho_lee_price(interest_tree, cf_tree)
    print(f"price of non-prepayable mortgage: {price}")
    print(f"spot rate duration of non-prapayable mortgage: {duration}")

    # prepayable mortgage
    remain_p = np.array([equal_c_to_p(payments, r, comp_period, i) for i in range(6, 0, -1)])
    print("remaining balance at the begin of each year")
    print(remain_p)

    price, duration = ho_lee_price(interest_tree, cf_tree, prepay=remain_p)
    print(f"price of prepayable mortgage: {price}")
    print(f"spot rate duration of prapayable mortgage: {duration}")

    # principal only and interest only for prepayable mortgage
    pay_p = np.r_[remain_p[:-1] - remain_p[1:], remain_p[-1]]
    pay_i = payments * np.ones(len(pay_p)) - pay_p

    po_tree = np.tril(np.ones_like(interest_tree) * pay_p.reshape(-1, 1), 0)
    io_tree = np.tril(np.ones_like(interest_tree) * pay_i.reshape(-1, 1), 0)

    price_po, price_io, dur_po, dur_io = ho_lee_price_poio(interest_tree, cf_tree, po_tree,
                                                           io_tree, prepay=remain_p)
    print(f"PO price of prepayable mortgage: {price_po}")
    print(f"PO spot duration of prepayable mortgage: {dur_po}")
    print(f"IO price of prepayable mortgage: {price_io}")
    print(f"IO spot duration of prepayable mortgage: {dur_io}")
