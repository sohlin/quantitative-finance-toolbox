import numpy as np
import pandas as pd
import statsmodels.api as sm


class KalmanFilter(sm.tsa.statespace.MLEModel):
    """ KalmanFilter implemented through statsmodels. """
    def __init__(self, y_t, exog):
        super(KalmanFilter, self).__init__(
            endog=y_t, exog=exog, k_states=exog.shape[1],
            initialization='approximate_diffuse')

        # Since the design matrix is time-varying, it must be
        # shaped k_endog x k_states x nobs
        # Notice that exog.T is shaped k_states x nobs, so we
        # just need to add a new first axis with shape 1
        self.ssm['design'] = exog.T[np.newaxis, :, :]  # shaped 1 x 2 x nobs
        self.ssm['selection'] = np.eye(self.k_states)
        self.ssm['transition'] = np.eye(self.k_states)

        # number of parameters need to be positive
        self.positive_parameters = slice(1, self.k_states+2)

    @property
    def param_names(self):
        return ['intercept', 'var.e'] + ['var.x' + str(i) + '.coeff' for i in range(1, self.k_states+1)]

    @property
    def start_params(self):
        """
        Defines the starting values for the parameters
        The linear regression gives us reasonable starting values
        for the variance of the epsilon error
        """
        exog = sm.add_constant(self.exog)
        res = sm.OLS(self.endog, exog).fit(cov_type='HC0')
        params = np.r_[0, res.scale, [0.001] * self.k_states]
        return params

    def transform_params(self, unconstrained):
        """
        We constraint the last three parameters
        ('var.e', 'var.x.coeff', 'var.w.coeff') to be positive,
        because they are variances
        """
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = constrained[self.positive_parameters] ** 2
        return constrained

    def untransform_params(self, constrained):
        """
        Need to unstransform all the parameters you transformed
        in the `transform_params` function
        """
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = unconstrained[self.positive_parameters] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        params = super(KalmanFilter, self).update(params, **kwargs)

        self['obs_intercept', 0, 0] = params[0]
        self['obs_cov', 0, 0] = params[1]
        self['state_cov'] = np.diag(params[2:(self.k_states+2)])


if __name__ == '__main__':
    y = np.random.normal(size=(1000,))
    X = sm.add_constant(np.random.normal(size=(1000,)))
    mod = KalmanFilter(y, X)
    res = mod.fit()