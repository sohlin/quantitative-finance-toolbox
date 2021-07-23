#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
import statsmodels.api as sm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from sympy.solvers import solve
from sympy import Symbol
import math
import holidays
from scipy.interpolate import interp1d


class DateFunc:
    @classmethod
    def third_wednesday_of_month(cls, year, month):
        c = calendar.Calendar(firstweekday=calendar.SUNDAY)

        year = year
        month = month

        monthcal = c.monthdatescalendar(year, month)
        third_wednesday = [day for week in monthcal for day in week if \
                           day.weekday() == calendar.WEDNESDAY and \
                           day.month == month][2]
        return dt.datetime(third_wednesday.year, third_wednesday.month, third_wednesday.day)

    @classmethod
    def third_monday_of_month(cls, year, month):
        c = calendar.Calendar(firstweekday=calendar.SUNDAY)

        year = year
        month = month

        monthcal = c.monthdatescalendar(year, month)
        third_monday = [day for week in monthcal for day in week if \
                        day.weekday() == calendar.MONDAY and \
                        day.month == month][2]
        return dt.datetime(third_monday.year, third_monday.month, third_monday.day)

    @classmethod
    def is_business_day(cls, date):
        return bool(len(pd.bdate_range(date, date)))

    @classmethod
    def is_holiday(cls, date):
        us_holidays = holidays.UnitedStates(years=date.year)
        return date in [dt.datetime(holiday[0].year, holiday[0].month, holiday[0].day) for holiday in
                        us_holidays.items()]

    @classmethod
    def get_most_recent_business_day(cls, date):
        while not DateFunc.is_business_day(date) or DateFunc.is_holiday(date):
            date += dt.timedelta(days=1)

        return date

    @classmethod
    def delta_month(cls, cur_date, n):
        cur_month = cur_date.month
        next_month = cur_month + n
        next_year = cur_date.year
        if next_month <= 0:
            next_year = cur_date.year - (-next_month // 12 + 1)
            next_month = 12 - (-next_month % 12)
        elif next_month > 12:
            next_year = cur_date.year + next_month // 12
            next_month = next_month % 12
        next_date = dt.datetime(next_year, next_month, cur_date.day)
        return next_date

    @classmethod
    def is_EOM(cls, date):
        next_date = date + dt.timedelta(days=1)
        return next_date.month != date.month

    @classmethod
    def calc_30I(cls, d1, d2):
        D1, M1, Y1 = d1.day, d1.month, d1.year
        D2, M2, Y2 = d2.day, d2.month, d2.year

        if DateFunc.is_EOM(d1) and DateFunc.is_EOM(d2) and M1 == 2 and M2 == 2:
            D2 = 30
        elif DateFunc.is_EOM(d1) and M1 == 2:
            D1 = 30
        if D2 == 31 and (D1 == 30 or D1 == 31):
            D2 = 30
        if D1 == 31:
            D1 = 30

        return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)

    # TODO: adjust to pandas format
    @classmethod
    def forward_date_parser(cls, date_int):
        year = date_int // 10000
        month = (date_int % 10000) // 100
        day = date_int % 100
        date = dt.datetime(year, month, day)

        return date


class ConversionFunc:
    @classmethod
    def simple_to_df(cls, simple_rate, t0, t1):
        act_num_days = (t1 - t0).days
        df = 1 / (1 + simple_rate * act_num_days / 360)
        return df

    @classmethod
    def forward_to_df(cls, forward_rate, t1, yield_curve):
        forward_start_month = t1.month - 3
        forward_start_year = t1.year
        if forward_start_month <= 0:
            forward_start_year -= 1
            forward_start_month += 12
        forward_start_day = t1.day

        forward_start_date = DateFunc.third_wednesday_of_month(forward_start_year, forward_start_month)
        act_num_days = (t1 - forward_start_date).days
        forward_start_df = yield_curve.get_discount_factor(forward_start_date)
        df = 1 / (1 + forward_rate * act_num_days / 360) * forward_start_df

        return df

    @classmethod
    def df_to_simple(cls, df, t0, t1):
        act_num_days = (t1 - t0).days
        return (1 / df - 1) / (act_num_days / 360)

    @classmethod
    def df_to_forward(cls, df, t1, middle_date, yield_curve):
        act_num_days = (t1 - middle_date).days
        middle_df = yield_curve.get_discount_factor(middle_date)
        # print((middle_df / df - 1) / (act_num_days / 360))
        return (middle_df / df - 1) / (act_num_days / 360)

    @classmethod
    def df_to_zero(cls, df, t0, t1):
        if t0 == t1:
            return 0
        thirty_I = DateFunc.calc_30I(t0, t1)
        if thirty_I == 0:
            return 0
        zero_rate = (df ** (-360 / (2 * thirty_I)) - 1) * 2
        return zero_rate


class DateDiscount:
    '''
    A class that represents the discount factor from t0 to t1
    Supports construction from simple rates, forward rates, and swap rates
    Also supports conversion to simple rates, forward rates, and swap rates
    '''

    def __init__(self, *args):
        self.t0 = args[0]

        if args[1] == 'Simple Rate':
            self.simple_to_df_construct(args[2:])
        elif args[1] == 'Forward Rate':
            self.forward_to_df_construct(args[2:])
        elif args[1] == 'Swap Rate':
            self.swap_to_df_construct(args[2:])
        else:
            print('Invalid Arguments')

    # constructions
    def simple_to_df_construct(self, args):
        self.t1 = args[0]
        simple_rate = args[1]

        self.df = ConversionFunc.simple_to_df(simple_rate, self.t0, self.t1)

    def forward_to_df_construct(self, args):
        self.t1 = args[0]
        forward_rate = args[1]
        yield_curve = args[2]

        self.df = ConversionFunc.forward_to_df(forward_rate, self.t1, yield_curve)

    def swap_to_df_construct(self, args):
        swap_rate = args[0]
        swap_years = args[1]
        yield_curve = args[2]

        total_months = swap_years * 12
        # fetch all payment dates
        payment_dates = [DateFunc.get_most_recent_business_day(DateFunc.delta_month(self.t0, i)) for i in
                         range(6, total_months + 6, 6)]

        self.t1 = payment_dates[-1]
        # get the current range of the extrapolated yield curve
        yield_start, yield_end = yield_curve.get_date_range()

        unknown_start_index = 0
        # find the first payment date outside the range of the current yield curve
        for i in range(len(payment_dates)):
            if payment_dates[i] <= yield_end:
                unknown_start_index += 1

        # if the first payment date ouside the range of the current yield curve is not the last payment
        if unknown_start_index < len(payment_dates) - 1:
            swap_payments_pv = 0
            prev_date = self.t0

            for date in payment_dates[:unknown_start_index]:
                swap_payments_pv += DateFunc.calc_30I(prev_date,
                                                       date) / 360 * swap_rate * yield_curve.get_discount_factor(date)
                prev_date = date
            # set the discount factor of the last date as x
            x = Symbol('x')
            # convert to simple rate
            last_simple_rate = ConversionFunc.df_to_simple(x, self.t0, payment_dates[-1])
            # fetch the last known simple rate from the curve
            last_known_simple_rate = yield_curve.get_date_discount(payment_dates[unknown_start_index - 1]).to_simple()

            unknown_pv = 0
            # iterate over all unknown payments and calculate their pv in terms of x
            for j in range(unknown_start_index, len(payment_dates) - 1):
                unknown_date = payment_dates[j]
                unknown_simple_rate = last_known_simple_rate + (last_simple_rate - last_known_simple_rate) * (
                            unknown_date - payment_dates[unknown_start_index - 1]).days / (
                                                  payment_dates[-1] - payment_dates[unknown_start_index - 1]).days
                unknown_discount_factor = ConversionFunc.simple_to_df(unknown_simple_rate, self.t0, unknown_date)
                unknown_payment_pv = DateFunc.calc_30I(payment_dates[j - 1],
                                                        unknown_date) / 360 * swap_rate * unknown_discount_factor
                unknown_pv += unknown_payment_pv

            # express the discount factor for the last date in terms of previous cashflow pv
            last_discount_factor = (1 - swap_payments_pv - unknown_pv) / (
                        1 + DateFunc.calc_30I(payment_dates[-2], payment_dates[-1]) * swap_rate / 360)
            # equate that with x and solve
            last_discount_factor_soln = solve(last_discount_factor - x, x)[-1]
            self.df = abs(last_discount_factor_soln)

        else:
            # all swap payment dates are within the current yield curve range
            swap_payments_pv = 0
            prev_date = self.t0
            for date in payment_dates[:-1]:
                swap_payments_pv += DateFunc.calc_30I(prev_date,
                                                       date) / 360 * swap_rate * yield_curve.get_discount_factor(date)
                prev_date = date
            self.df = (1 - swap_payments_pv) / (
                        1 + DateFunc.calc_30I(payment_dates[-2], payment_dates[-1]) * swap_rate / 360)

    # conversion functions
    def to_simple(self):
        return ConversionFunc.df_to_simple(self.df, self.t0, self.t1)

    def to_forward(self, middle_date, yield_curve):
        return ConversionFunc.df_to_forward(self.df, self.t1, middle_date, yield_curve)

    def to_zero(self):
        return ConversionFunc.df_to_zero(self.df, self.t0, self.t1)

    # frequently used getter
    def get_mat_date(self):
        return self.t1

    def get_df(self):
        return self.df


class ExtrapolatedYieldCurve:
    '''
    A class that represents the extrapolated yield curve that
    is implemented as a list of date_discount objects
    date_discount_list: list of date_discount objects
    '''

    def __init__(self, start_date):
        self.start_date = start_date
        self.date_discount_list = []

    def add_date_discount_list(self, new_date_discount):
        """ Function to take a new date discount and add to the list """
        new_date = new_date_discount.get_mat_date()
        insert_loc = self.get_date_index(new_date)
        if len(self.date_discount_list) > 0 and new_date > self.date_discount_list[-1].get_mat_date():
            insert_loc += 1
        if len(self.date_discount_list) > insert_loc and self.date_discount_list[insert_loc].get_mat_date() == new_date:
            print('Date already in curve')
        self.date_discount_list.insert(insert_loc, new_date_discount)

    def get_date_range(self):
        """ Function to get the current date range of the yield curve """
        if len(self.date_discount_list) == 0:
            return -1
        return self.date_discount_list[0].get_mat_date(), self.date_discount_list[-1].get_mat_date()

    def get_date_index(self, new_date):
        """ helper: find the location for new_date in the current yield curve """
        date_index = 0

        for i in range(0, len(self.date_discount_list)):
            date_index = i
            cur_date = self.date_discount_list[i].get_mat_date()
            if new_date <= cur_date:
                break

        return date_index

    def get_date_discount(self, date):
        """ Function to use piecewise linear to get a DateDiscount object """
        date_index = self.get_date_index(date)
        index_date_discount = self.date_discount_list[date_index]
        index_date = index_date_discount.get_mat_date()

        if date == index_date:
            return index_date_discount
        elif (date < index_date and date_index == 0) or (date > index_date and date_index == len(self) - 1):
            new_date_discount = DateDiscount(self.start_date, 'Simple Rate', date, index_date_discount.to_simple())
            return new_date_discount
        else:
            index_simple_rate = index_date_discount.to_simple()
            prev_index_date_discount = self.date_discount_list[date_index - 1]
            prev_index_date = prev_index_date_discount.get_mat_date()
            prev_index_simple_rate = prev_index_date_discount.to_simple()

            new_simple_rate = prev_index_simple_rate + (index_simple_rate - prev_index_simple_rate) * (
                        date - prev_index_date).days / (index_date - prev_index_date).days
            new_date_discount = DateDiscount(self.start_date, 'Simple Rate', date, new_simple_rate)

            return new_date_discount

    def get_discount_factor(self, date):
        """ Function to get the discount factor for date """
        date_discount = self.get_date_discount(date)
        return date_discount.get_df()

    def get_discount_factor_list(self):
        return [dd.get_df() for dd in self.date_discount_list]

    def get_start_date(self):
        return self.start_date

    def __len__(self):
        return len(self.date_discount_list)


class NelsonSeigelCurve(object):
    def __init__(self, rate, t, start_date, rate_type):
        self.params = None

        self.rate_type = rate_type
        if rate_type == 'Zero Rate':
            self.t2m = self.get_t2m(start_date, t)
            self.dcf = self.zero_to_dcf(rate, self.t2m)
            self.r_cc = self.dcf_to_rcc(self.dcf, self.t2m)
        elif rate_type == 'Forward Rate':
            pass

    def fit(self):

        res = optimize.minimize(
            fun=self.get_ns_loss,
            x0=np.array([1, 1]),
            tol=1e-16
        )
        self.params = [res.x[0], res.x[1]]
        self.fitted, tmp = self.nelson_seigel_approx(self.params, np.array(self.t2m), np.array(self.r_cc))
        self.params += tmp.flatten().tolist()

    def predict(self, t=None):
        if t is None:
            dcf = self.rcc_to_dcf(self.fitted.flatten(), self.t2m)
            zero = self.dcf_to_zero(dcf, np.array(self.t2m))
            return zero, dcf

    def zero_to_dcf(self, r, t):
        dcf = [1 / (1 + i / 2) ** (2 * j) for i, j in zip(r, t)]
        return dcf

    def dcf_to_zero(self, dcf, t):
        return 2 * (np.power(1 / dcf, 1 / (2 * t)) - 1)

    def get_t2m(self, start_date, t):
        def helper(start, end):
            days = 360 * (end.year - start.year) + 30 * (end.month - start.month)
            day1 = 30 if start.is_month_end else start.day
            day2 = 30 if (end.is_month_end and end.month != 2) else end.day
            days += day2 - day1
            return days / 360

        return list(map(lambda x: helper(start_date, x), t))

    def dcf_to_rcc(self, dcf, t2m):
        return [-np.log(i) / j if j != 0 else 0 for i, j in zip(dcf, t2m)]

    def rcc_to_dcf(self, r, t2m):
        return np.exp(-np.array(r) * np.array(t2m))

    def nelson_seigel_approx(self, params, m, r):
        tau1, tau2 = params
        if m.ndim == 1:
            m = m[:, np.newaxis]
        if r.ndim == 1:
            r = r[:, np.newaxis]
        X = np.c_[(1 - np.exp(-tau1 * m)) / (tau1 * m), (1 - np.exp(-tau2 * m)) / (tau2 * m) - np.exp(-tau2 * m)]
        X = sm.add_constant(X)
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(r)

        return X.dot(beta), beta

    def get_ns_loss(self, params):
        approx, beta = self.nelson_seigel_approx(params, np.array(self.t2m), np.array(self.r_cc))
        loss = np.mean(np.power(np.array(self.r_cc) - approx.flatten(), 2))
        return loss


if __name__ == '__main__':
    data = pd.read_excel("../notebooks/ICVS_23_02282019.xlsx", sheet_name=[0, 1, 2, 3], header=0)
    input_rates = data[0]
    output_rates = data[1]

    start = pd.Timestamp('2019-03-04')
    output_rates['Date'] = pd.to_datetime(output_rates['Date'], format="%m/%d/%Y")

    model = NelsonSeigelCurve((output_rates['Zero Rate'] / 100).tolist()[1:],
                              output_rates['Date'].tolist()[1:],
                              start,
                              "Zero Rate")
    model.fit()
    zero, dcf = model.predict()
    plt.plot(np.r_[0, zero] * 100)
    plt.plot(output_rates['Zero Rate'].values)
    plt.show()

