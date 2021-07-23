#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/14 11:28
# @Author  : Zefu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import combinations_with_replacement, combinations
from collections import defaultdict

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class OLS(object):
    """ Costumed Ordinary Least Squared regression class.

    Implemented functions:
        fit: OLS regression (with ols standard error & white error).
        predict: return the fitted value.
        scatter plot: plot the scatter points of independent and dependent variables.
        heteroscedasticity test: test whether heteroscedasticity assumption is needed.
        rolling test: test whether coefs are stationary.
        qq plot: test whether residual is normal-like.

    """

    def __init__(self, intercept=True):
        self.n_samples = None
        self.n_features = None
        self.feature_names = None
        self.beta_hat = None
        self.pred = None
        self.residual = None
        self.skew_res = None
        self.kurt_res = None
        self.f_stat = None
        self.f_p_value = None
        self.r2 = None
        self.r2_adj = None
        self.V = None
        self.st_error = None
        self.df = None
        self.t_stat = None
        self.p_value = None
        self.ci = None
        self.log_likelihood = None
        self.AIC = None
        self.BIC = None
        self.HQIC = None
        self.dw_test = None
        self.bg_test = None
        self.jb_test = None
        self.jb_test_p = None
        self.cond = None
        self.intercept = intercept

    def fit(self, X, y, std_error='white', feature_names=None):
        """ Full fit of the model.

        Args:
            X: np.array, shape=[n_samples, n_features], independent variable.
            y: np.array, shape=[n_samples, 1] or [n_samples,], dependent variable.
            std_error: str, assumption for standard error of coefs.
            feature_names: list or None, variable names.

        Returns:
            self: fitted model.
        """
        # check the shape of X and y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.ndim != 2:
            raise Exception("The dimension of X must be 2.")
        if not feature_names:
            feature_names = ['X' + str(i) for i in range(len(X[0]))]
        # check the constant column
        if np.any(X[:, 0] != 1) and self.intercept:
            X = np.column_stack((np.ones(len(X)), X))
            feature_names = ['const'] + feature_names

        self.feature_names = feature_names
        # get the degree of freedom
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features - 1
        self.df = len(X) - len(X[0])

        # get the OLS estimate
        self.beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        pred = X.dot(self.beta_hat)
        self.pred = pred

        # get the residual
        self.residual = y - pred
        self.skew_res = stats.skew(self.residual)
        self.kurt_res = stats.kurtosis(self.residual, fisher=False)

        # compute R squared
        self.r2 = np.sum((pred - np.mean(y)) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # compute adjusted R squared
        self.r2_adj = 1 - (1 - self.r2) * (len(X) - 1) / (self.df - 1)

        # get F test
        self.f_stat, self.f_p_value = self._f_test(SSM=np.sum((pred - np.mean(y)) ** 2),
                                                   SSE=np.sum(self.residual ** 2),
                                                   df_m=n_features - 1,
                                                   df_e=self.df)

        if std_error == 'ols':
            # compute OLS standard error (unbiased)
            sigma2 = np.sum(self.residual ** 2) / self.df
            self.V = np.linalg.inv(X.T.dot(X)) * sigma2
            self.st_error = np.sqrt(np.diagonal(self.V))
        elif std_error == 'white':
            # compute the White standard error
            self.V = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(
                np.diag((self.residual ** 2).flatten())).dot(X).dot(np.linalg.inv(X.T.dot(X)))
            self.st_error = np.sqrt(np.diagonal(self.V))
        else:
            raise Exception("No existing mode for " + std_error)

        # hypothesis test
        self.t_stat, self.p_value = self._t_test(self.beta_hat,
                                                 self.st_error.reshape(-1, 1),
                                                 self.df,
                                                 null=0)
        # get the confidence interval
        self.ci = self._get_conf_int()
        # get the information criterion
        log_ml = -len(X) / 2 * (np.log(2 * np.pi) + 1) \
                 - len(X) / 2 * np.log(np.mean(self.residual ** 2))
        self.log_likelihood = log_ml
        self.AIC, self.BIC, self.HQIC = self._get_info_crit(log_ml, n_features, n_samples)

        # get DW test and Breusch-Godfrey test statistic
        self.dw_test = self._DW_test()
        self.bg_test = self._Breusch_Godfrey_test(lags=12)
        # get JB test statistic and p value
        self.jb_test, self.jb_test_p = self._JB_test()
        # get condition number of X
        self.cond = np.linalg.cond(X)

        return self

    def predict(self, X):
        """ Return linear predicted values from a design matrix.

        Args:
            X: np.array, shape=[n_samples, n_features], independent variable.

        Returns:
            pred: np.array, shape=[n_samples, 1], predicted value.
        """
        # check the constant column
        if np.any(X[:, 0] != 1) and self.intercept:
            X = np.column_stack((np.ones(len(X)), X))
        if X.shape[1] != self.beta_hat.shape[0]:
            raise Exception("The cols of X do not match with betas.")
        pred = X.dot(self.beta_hat)

        return pred

    def scatter_plot(self, x, y, fit=True, xlabel=None, ylabel=None):
        """ Scatter plot of two variables a and b.

        Args:
            x: np.array, shape=[n_samples, n_features], variable a.
            y: np.array, shape=[n_samples,], variable b.
            fit: bool, if True, plot the fitted (regression line).
            xlabel: str or None, label for x axis.
            ylabel: str or None, label for y axis.

        """
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        if fit:
            ax.plot(x, self.pred.flatten(), '-r')
        ax.set_title("Scatter plot")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.show()

    def _t_test(self, beta_hat, std_error, df, null=0):
        """ Built-in method for standard t test.

        Args:
            beta_hat: np.array, shape=[n_samples, 1], fitted coefs.
            std_error: np.array, shape=[n_samples, 1], standard error for coefs.
            df: int, degree of freedom for the t distribution.
            null: int, null hypothesis, Default=0.

        Returns:
            t_stat: float, test statistic.
            p_value: float, computed Prob(t test).

        """
        t_stat = (beta_hat - null) / std_error
        p_value = np.minimum(stats.t.sf(t_stat, df), 1-stats.t.sf(t_stat, df)) * 2
        return t_stat, p_value

    def _f_test(self, SSM, SSE, df_m, df_e):
        """ Built-in method for F test.

        Args:
            SSM: float, sum of squares of model.
            SSE: float, sum of squares of error.
            df_m: int, degree of freedom for model (corrected).
            df_e: int, degree of freedom for residual (corrected).

        Returns:
            f_stat: float, test statistic for F test.
            p_value: float, computed Prob(F test).

        """
        f_stat = SSM / df_m / (SSE / df_e)
        p_value = stats.f.sf(f_stat, df_m, df_e)
        return f_stat, p_value

    def hetero_std_test(self, X, y):
        """ Heteroscedasticity test for residuals.

        Use white test to test whether error depends on X;
        use ARCH F test to test whether error depends on lagged term.

        Args:
            X: np.array, shape=[n_samples, n_features], independent variable.
            y: np.array, shape=[n_samples, 1] or [n_samples,], dependent variable.

        Returns:
            white_p_value: float, p-value associated with white test.
            f_p_value: float, p-value associated with ARCH F test.

        """
        if self.residual is None:
            raise Exception("The model has not been fitted.")

        # white test for sigma^2 depends on x
        n_samples, n_feature = X.shape

        if np.any(X[:, 0] != 1) and self.intercept:
            X, m = self._poly_features(X, n_feature)
        elif np.all(X[:, 0] == 1) and not self.intercept:
            X, m = self._poly_features(X[:, 1:], n_feature - 1, include_bias=False)
        elif np.any(X[:, 0] != 1) and not self.intercept:
            X, m = self._poly_features(X, n_feature, include_bias=False)

        res2 = np.power(self.residual, 2)
        plt.plot(res2)
        plt.title("squared residual")
        plt.show()
        # get the OLS estimate for the auxiliary regression
        beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(res2)
        pred = X.dot(beta_hat)
        # compute R squared for the auxiliary regression
        r2 = np.sum((pred - np.mean(res2)) ** 2) / np.sum((res2 - np.mean(res2)) ** 2)

        chi2_stat = n_samples * r2
        white_p_value = stats.chi2.sf(chi2_stat, m)
        print("White test p-value: " + str(white_p_value))

        # ARCH F-test for sigma^2 depend on lagged term
        lag_res2_1 = np.roll(res2, 1)[3:]
        lag_res2_2 = np.roll(res2, 2)[3:]
        lag_res2_3 = np.roll(res2, 3)[3:]
        res2 = res2[3:]
        X = np.column_stack((np.ones(len(lag_res2_1)), lag_res2_1,
                             lag_res2_2, lag_res2_3))

        beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(res2)
        pred = X.dot(beta_hat)
        residual = res2 - pred

        f_stat, f_p_value = self._f_test(SSM=np.sum((pred - np.mean(res2)) ** 2),
                                         SSE=np.sum(residual ** 2),
                                         df_m=len(X[0]) - 1,
                                         df_e=len(res2) - len(X[0]))
        print("ARCH F test p-value: " + str(f_p_value))

        if white_p_value <= 0.05:
            print("White test: Reject null that sigma^2 "
                  "does not depend on X.")
        if f_p_value <= 0.05:
            print("ARCH F test: Reject null that sigma^2 "
                  "does not depend on lagged term.")
        if white_p_value > 0.05 and f_p_value > 0.05:
            print("Not reject null. No significance evidence.")

        return white_p_value, f_p_value

    def _poly_features(self, X, degree, interaction_only=False, include_bias=True):
        """ Built-in method to generate polynomial features.

        Args:
            X: np.array, shape=[n_samples, n_features], independent variable.
            degree: int, designed degree of the polynomial.
            interaction_only: bool, whether polynomial only contain interaction term.
            include_bias: bool, whether polynomial include intercept term or not.

        Returns:
            features: np.array, commputed polynomial features.
            m: int, number of features of polynomial.

        """
        features = X.copy()
        comb = []
        if interaction_only:
            for d in range(2, degree + 1):
                comb += list(combinations(list(range(X.shape[1])), d))
        else:
            for d in range(2, degree + 1):
                comb += list(combinations_with_replacement(list(range(X.shape[1])), d))

        for c in comb:
            tmp = np.ones((len(X), 1))
            for i in c:
                tmp *= X[:, i:i + 1]
            features = np.column_stack((features, tmp))

        if include_bias:
            features = np.column_stack((np.ones(len(X)), features))

        return features, len(X[0])

    def _get_conf_int(self):
        """ Built-in method to compute confidence interval.
        """
        ci = {}
        for i in [0.005, 0.025, 0.05]:
            thresh = stats.t.ppf(1 - i, self.df)
            tmp = [(self.beta_hat[j, 0] - thresh * self.st_error[j],
                    self.beta_hat[j, 0] + thresh * self.st_error[j])
                   for j in range(len(self.beta_hat))]
            ci['ci_' + str(int((1 - 2 * i) * 100))] = tmp
        return ci

    def _get_info_crit(self, log_ml, n_features, n_samples):
        """ Built-in method to get information criterion. (for model selection)

        Compute the three information criterion as below:
            AIC, BIC, Hannah-Quinn Criterion.

        Args:
            log_ml: float, the optimal (maximized) log-likelihood value.
            n_features: int, number of features of the model.
            n_samples: int, number of samples of the data.

        Returns:
            aic: float, AIC.
            bic: float, BIC.
            hqic: float, Hannah-Quinn Criterion.

        """
        aic = -2 * log_ml + 2 * n_features
        bic = -2 * log_ml + n_features * np.log(n_samples)
        hqic = -2 * log_ml + 2 * n_features * np.log(np.log(n_samples))
        return aic, bic, hqic

    def _DW_test(self):
        """ Built-in method for Durbin-Watson test. (for Auto Correlation)
        """
        statistic = np.sum(np.power(self.residual[1:] - self.residual[:-1], 2)) \
                    / np.sum(np.power(self.residual[1:], 2))
        return statistic

    def _Breusch_Godfrey_test(self, lags=12):
        """ Built-in method for Breusch-Godfrey test. (for auto correlation)

        Args:
            lags: int, number of lagged terms.

        Returns:
            res: DataFrame, test statistic and p-value for the test.

        """
        res = defaultdict(list)
        for lag in range(1, lags + 1):
            X = np.column_stack([np.roll(self.residual, i)[lag:] for i in range(1, lag + 1)])
            y = self.residual[lag:]
            beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            pred = X.dot(beta_hat)
            residual = y - pred
            r2 = np.sum((pred - np.mean(y)) ** 2) / np.sum((y - np.mean(y)) ** 2)
            n_samples, n_features = X.shape
            statsistic = (n_samples - n_features) * r2
            res['BG statistic'].append(statsistic)
            res['p-value'].append(stats.chi2.sf(statsistic, n_features))
        return pd.DataFrame(res, index=list(range(1, lags + 1)))

    def qq_plot(self, x, y='norm'):
        """ Quantile-Quantile plot for variable and theoretical distribution.

        Args:
            x: np.array, observed variable.
            y: str, theoretical distribution, e.g. "norm".

        """
        stats.probplot(x, dist=y, plot=plt)
        plt.show()

    def _JB_test(self):
        """ Built-in method for Jarque-Bera test (for Normality)
        """
        JB = len(self.residual) / 6 * ((self.skew_res ** 2)
                                       + ((self.kurt_res - 3)
                                          ** 2) / 4)
        p_value = stats.chi2.sf(JB, 2)
        return JB, p_value

    def rolling_test(self, X, y, intercept=True, window_lens=60,
                     std_error='white', ci_size=0.025, plot=None, labels=None):
        """ Rolling regression for X and y to test stationary result.

        Args:
            X: np.array, shape=[n_samples, n_features], independent variable.
            y: np.array, shape=[n_samples, 1] or [n_samples,], dependent variable.
            intercept: bool, whether to include intercept term, Default=True.
            window_lens: int, window length for rolling, Default=60.
            std_error: str, assumption for standard error of coefs, Default='white'.
            ci_size: float, size of confidence interval, Default=0.025.
            plot: list or None, if not None, then plot the desired coefs.
            labels: list or None, if not None, then label the plot.

        Returns:
            betas: np.array, rolling coefs.
            cis: np.array, rolling confidence interval (exclude the mean (i.e. betas)).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.ndim != 2:
            raise Exception("The dimension of X must be 2.")
        n_samples = len(X)
        if n_samples < window_lens:
            raise Exception("Sample size is limited!")

        # check the constant column
        if np.any(X[:, 0] != 1) and intercept:
            X = np.column_stack((np.ones(len(X)), X))
            labels = ['const'] + labels
        df = n_samples - len(X[0])
        if ci_size:
            thresh = stats.t.ppf(1 - ci_size, df)

        for i in range(n_samples - window_lens):
            X_roll = X[i:i + window_lens, :]
            y_roll = y[i:i + window_lens]
            beta_hat = np.linalg.inv(X_roll.T.dot(X_roll)).dot(X_roll.T).dot(y_roll)
            pred = X_roll.dot(beta_hat)

            # get the residual
            residual = y_roll - pred
            if std_error == 'ols':
                # compute OLS standard error (unbiased)
                sigma2 = np.sum(residual ** 2) / df
                V_ols = np.linalg.inv(X_roll.T.dot(X_roll)) * sigma2
                st_error = np.sqrt(np.diagonal(V_ols))
            elif std_error == 'white':
                # compute the White standard error
                V_white = np.linalg.inv(X_roll.T.dot(X_roll)).dot(X_roll.T).dot(
                    np.diag((residual ** 2).flatten())).dot(X_roll).dot(
                    np.linalg.inv(X_roll.T.dot(X_roll)))
                st_error = np.sqrt(np.diagonal(V_white))

            if not ci_size:
                pass
            else:
                ci = thresh * st_error
            if i == 0:
                betas = beta_hat.flatten()
                cis = ci.flatten()
            else:
                betas = np.vstack([betas, beta_hat.flatten()])
                cis = np.vstack([cis, ci.flatten()])

        if not plot:
            pass
        elif intercept:
            fig, ax = plt.subplots()
            for idx in plot:
                ax.plot(np.arange(len(betas)), betas[:, idx + 1], label=labels[idx + 1])
                ax.fill_between(np.arange(len(betas)), (betas[:, idx + 1] - ci[idx + 1]),
                                (betas[:, idx + 1] + ci[idx + 1]), color='b', alpha=.2)
            ax.set_title('Rolling regression with window length: ' + str(window_lens))
            ax.set_ylabel('Coefficient')
            ax.legend()
            plt.show()
        elif not intercept:
            fig, ax = plt.subplots()
            for idx in plot:
                ax.plot(np.arange(len(betas)), betas[:, idx], label=labels[idx])
                ax.fill_between(np.arange(len(betas)), (betas[:, idx] - ci[:, idx]),
                                (betas[:, idx] + ci[:, idx]), color='b', alpha=.2)
            ax.set_title('Rolling regression with window length ' + str(window_lens))
            ax.set_ylabel('Coefficient')
            ax.legend()
            plt.show()
        return betas, cis

    def _round(self, df):
        """ Built-in method to round the output.

        Args:
            df: DataFrame, data output.

        """

        def get_round(x):
            if isinstance(x, tuple):
                return tuple(map(lambda y: round(y, 3), x))
            elif isinstance(x, str):
                return x
            else:
                return round(x, 3)

        return df.applymap(get_round)

    def summary(self):
        """ Presentation of the regression results.
        """
        if self.beta_hat is None:
            raise Exception("The model has not been fitted.")

        des_df = pd.DataFrame({
            "Model:": "OLS",
            "Method:": "Least Squares",
            "No. Observations:": self.n_samples,
            "Df Residuals:": self.df,
            "Df Model:": self.n_features,
            "R-squared:": self.r2,
            "Adj. R-squared:": self.r2_adj,
            "F-statistic:": self.f_stat,
            "Prob(F-statistic):": self.f_p_value,
            "Log-likelihood:": self.log_likelihood,
            "AIC:": self.AIC,
            "BIC:": self.BIC,
            "Hannah-Quinn IC:": self.HQIC
        }, index=[" "]).T

        des_df = des_df.round(3)
        des_df = self._round(des_df)

        coef_df = pd.DataFrame({
            "coef": self.beta_hat.flatten(),
            "std err": self.st_error.flatten(),
            "t": self.t_stat.flatten(),
            "P>|t|": self.p_value.flatten(),
            "[99% Conf. Int.]": self.ci['ci_99'],
            "[95% Conf. Int.]": self.ci['ci_95'],
            "[90% Conf. Int.]": self.ci['ci_90']
        }, index=self.feature_names)
        coef_df = coef_df.round(3)
        coef_df = self._round(coef_df)

        plus_df = pd.DataFrame({
            "Skew:": self.skew_res,
            "Kurtosis:": self.kurt_res,
            "Durbin-Watson:": self.dw_test,
            "Jarque-Bera(JB):": self.jb_test,
            "Prob(JB):": self.jb_test_p,
            "Cond. No.:": self.cond
        }, index=[" "]).T
        plus_df = plus_df.round(3)

        # BG_test_df = self.bg_test
        print("                                OLS Regression Results"
              "                               ")
        print('='*90)
        print(des_df)
        print('-'*90)
        print(coef_df)
        print('-'*90)
        print(plus_df)
        print('-'*90)
        print("Breusch-Godfrey test")
        print(self.bg_test.round(3))
        print('='*90)

        return None


if __name__ == '__main__':
    # load data
    data = pd.read_csv('./Data_PG_UN.csv', header=0, index_col=0)
    X = (data['MKT'] - data['RF']).values.reshape(-1, 1)
    y = data['UN'].values

    reg = OLS()
    reg.fit(X, y, std_error="white", feature_names=['MKT'])
    reg.summary()
    # reg.scatter_plot(X, y, xlabel="MKT", ylabel="UN")

    # reg.hetero_std_test(X, y)
    # reg.qq_plot(reg.residual.flatten(), y='norm')
    # y_pred = reg.predict(X)
    # X = np.column_stack((X, np.power(y_pred, 2), np.power(y_pred, 3)))

    # reg = OLS()
    # reg.fit(X, y, std_error="white", feature_names=['MKT', 'FITTEDsq', 'FITTEDcb'])
    # reg.summary()
    reg.rolling_test(X, y, plot=[0], labels=['beta'])
