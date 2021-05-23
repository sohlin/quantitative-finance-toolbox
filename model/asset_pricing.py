#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import operator as op
from functools import reduce


class OptionPricing(object):

    def __init__(self, S, K, r, sigma, T, theta):
        """
        Args:
            S: int or float, underlying asset's price.
            K: int or float, strike price.
            r: float, risk-free interest rate.
            sigma: float, volatility of the asset.
            T: int or float, time to maturity (represented as a unit-less fraction of one year).
            theta: float, annual dividend yield of the underlying asset.
        """
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.theta = theta

    def black_scholes_price(self, greeks=True, display=True):
        """ Calculate option price using Black Scholes model.

            Args:
                greeks: bool, calculate relevant Greeks or not.
                display: bool, print the price or not.

            Returns:
                res: dict, price information.
        """
        S = self.S
        K = self.K
        sigma = self.sigma
        r = self.r
        T = self.T
        theta = self.theta

        x = (np.log(S / K) + (r - theta + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
        call = S * np.exp(-theta * T) * stats.norm.cdf(x) - K * np.exp(-r * T) * stats.norm.cdf(x - sigma * np.sqrt(T))
        put = K * np.exp(-r * T) * stats.norm.cdf(-(x - sigma * np.sqrt(T))) - S * np.exp(-theta * T) * stats.norm.cdf(
            -x)
        digital = np.exp(-r * T) * stats.norm.cdf(x - sigma * np.sqrt(T))

        res = {
            "Price": (call, put, digital)
        }

        if greeks:
            # Delta V/S
            delta_call = np.exp(-theta * T) * stats.norm.cdf(x)
            delta_put = - np.exp(-theta * T) * stats.norm.cdf(-x)
            # Gamma V^2 / S^2
            gamma = np.exp(-theta * T) * stats.norm.cdf(x) / (S * sigma * np.sqrt(T))
            # Vega V / sigma
            vega = S * np.exp(-theta * T) * stats.norm.cdf(x) * np.sqrt(T)
            # Rho V / r
            rho_call = K * T * np.exp(-r * T) * stats.norm.cdf(x - sigma * np.sqrt(T))
            rho_put = -K * T * np.exp(-r * T) * stats.norm.cdf(-(x - sigma * np.sqrt(T)))
            # Theta -V / T
            theta_call = (-np.exp(-theta * T) * S * stats.norm.cdf(x) * sigma / (2 * np.sqrt(T))
                          - r * K * np.exp(-r * T) * stats.norm.cdf(x - sigma * np.sqrt(T))
                          + theta * S * np.exp(-theta * T) * stats.norm.cdf(x))
            theta_put = (-np.exp(-theta * T) * S * stats.norm.cdf(x) * sigma / (2 * np.sqrt(T))
                         + r * K * np.exp(-r * T) * stats.norm.cdf(-x + sigma * np.sqrt(T))
                         - theta * S * np.exp(-theta * T) * stats.norm.cdf(-x))

            greeks_table = pd.DataFrame({"Call": [delta_call, gamma, vega, rho_call, theta_call],
                                         "Put": [delta_put, gamma, vega, rho_put, theta_put]},
                                        index=['Delta', 'Gamma', 'Vega', 'Rho', 'Theta'])
            res['Delta'] = (delta_call, delta_put)
            res['Gamma'] = (gamma, gamma)
            res['Vega'] = (vega, vega)
            res['Rho'] = (rho_call, rho_put)
            res['Theta'] = (theta_call, theta_put)

        if display:
            print(f"European Call price: {call}\n")
            print(f"European Put price: {put}\n")
            print(f"Digital price: {digital}\n")
            if greeks:
                print(greeks_table)

        return res

    def binomial_tree_price(self, N=50, early_ex=False, display=True):
        """ Calculate option price using binomial tree.

        Args:
            N: int, depth of the tree.
            early_ex: bool, whether the option can be early exercised or not.
            display: bool, print the price or not.

        Returns:
            (call_pay, put_pay): tuple, price of the call and put.
        """

        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt))
        q0 = (np.exp(self.r * dt) - d) / (u - d)
        q1 = 1 - q0

        if not early_ex:
            call_payoff = np.zeros(N + 1)
            put_payoff = np.zeros(N + 1)
            prob = np.zeros(N + 1)
            S_price = np.zeros(N + 1)

            for i in range(N + 1):
                j = N - i
                prob[i] = (self.ncr(N, i)) * np.power(q0, j) * np.power(q1, i)
                S_price[i] = self.S * np.power(u, j) * np.power(d, i)
                if S_price[i] >= self.K:
                    call_payoff[i] = prob[i] * (S_price[i] - self.K)
                else:
                    put_payoff[i] = prob[i] * (self.K - S_price[i])
            call_pay = np.sum(call_payoff) * np.exp(-self.r * self.T)
            put_pay = np.sum(put_payoff) * np.exp(-self.r * self.T)
        else:
            call_payoff = [0] * N
            put_payoff = [0] * N

            # get initial value at T
            for i in range(N):
                spot = self.S * np.power(u, 2 * i - N)
                call_payoff[i] = max(self.S - self.K, 0)
                put_payoff[i] = max(self.K - self.S, 0)

            t = self.T
            # move to earlier times
            for j in range(N-1, 0, -1):
                t -= dt
                for i in range(j):
                    spot = self.S * np.power(u, 2 * i - j)
                    call_tmp = spot - self.K
                    put_tmp = self.K - spot
                    call_payoff[i] = (q0 * call_payoff[i + 1] + q1 * call_payoff[i]) * np.exp(-self.r * dt)
                    put_payoff[i] = (q0 * put_payoff[i+1] + q1 * put_payoff[i]) * np.exp(-self.r * dt)
                    call_payoff[i] = call_tmp if call_tmp > call_payoff[i] else call_payoff[i]
                    put_payoff[i] = put_tmp if put_tmp > put_payoff[i] else put_payoff[i]

            call_pay = call_payoff[0]
            put_pay = put_payoff[0]

        if display:
            print(f"Binomial Tree price for call option: {call_pay}")
            print(f"Binomial Tree price for put option: {put_pay}")

        return call_pay, put_pay

    @staticmethod
    def ncr(n, r):
        """ Compute combination number. """
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    def mc_path_simulation(self, N, symmetric=True):
        """ Path for brownian motion. """
        dt = self.T / N
        brownian = np.random.randn(N)
        vol = brownian * self.sigma * np.sqrt(dt)
        drift = (self.r - self.sigma * self.sigma / 2) * dt
        ret = np.exp(drift + vol)
        ret = np.cumprod(ret)

        if symmetric:
            brownian_2 = -1 * brownian
            vol2 = brownian_2 * self.sigma * np.sqrt(dt)
            drift2 = (self.r - self.sigma * self.sigma / 2) * dt
            ret2 = np.exp(drift2 + vol2)
            ret2 = np.cumprod(ret2)
            return ret, ret2

        return ret

    def monte_carlo_price(self, depth, paths, display=True, plot=False):
        """ Option pricing with Monte Carlo simulation (antithetic variates).

        The method of antithetic variates attempts to reduce variance by
        introducing negative correlation between pairs of observations.

        Args:
            depth: int, number of step in one simulation path.
            paths: int, number of simulation path.
            plot: bool, whether to plot the simulation results.

        Returns:
            (call, put): tuple, price of calls and puts.
        """
        calls = []
        puts = []
        for M in range(1, paths):
            final_s = []
            for i in range(M):
                res, res2 = self.mc_path_simulation(depth, symmetric=True)
                final_s.append(res[-1] * self.S)
                final_s.append(res2[-1] * self.S)
            # collect values from one path in one call option array
            call = [i - self.K for i in final_s if i > self.K]
            put = [self.K - i for i in final_s if i < self.K]
            call_res = np.sum(call) / (2 * M) * np.exp(-self.r * self.T)
            put_res = np.sum(put) / (2 * M) * np.exp(-self.r * self.T)
            calls.append(call_res)
            puts.append(put_res)
        
        if display:
            print(f"Binomial Tree price for call option: {calls[-1]}")
            print(f"Binomial Tree price for put option: {puts[-1]}")
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))
            axes[0].plot(calls, 'r')
            axes[0].set_title("European Call price from Monte Carlo simulation")
            axes[0].set_xlabel("Number of paths")
            axes[1].plot(puts, 'g')
            axes[1].set_title("European Put price from Monte Carlo simulation")
            axes[1].set_xlabel("Number of paths")

        return calls[-1], puts[-1]






