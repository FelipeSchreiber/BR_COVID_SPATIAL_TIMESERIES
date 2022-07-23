from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
import pandas as pd
import numpy as np

class compartmentalModel:
    def __init__(self,infections,deaths,\
                    initN,initE=1000,initI=47,initR=0,\
                    beta=1.08,sigma=0.2,gamma=0.2) -> None:
        self.df = pd.DataFrame.\
                    from_dict({"infected":infections,"total_recovered_or_dead":deaths})
        self.initN = initN
        self.initE = initE
        self.initI = initI
        self.initR = initR
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        params = Parameters()
        params.add('beta', value=self.beta, min=0, max=10)
        params.add('sigma', value=self.sigma, min=0, max=10)
        params.add('gamma', value=self.gamma, min=0, max=10)
        self.params = params

    def ode_model(self,z, t, beta, sigma, gamma):
        """
        Reference https://www.idmod.org/docs/hiv/model-seir.html
        """
        S, E, I, R = z
        N = S + E + I + R
        dSdt = -beta*S*I/N
        dEdt = beta*S*I/N - sigma*E
        dIdt = sigma*E - gamma*I
        dRdt = gamma*I
        return [dSdt, dEdt, dIdt, dRdt]

    def ode_solver(self,t, initial_conditions, params):
        initE, initI, initR, initN = initial_conditions
        beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
        initS = initN - (initE + initI + initR)
        res = odeint(self.ode_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
        return res

    def error(self,params, initial_conditions, tspan, data):
        sol = self.ode_solver(tspan, initial_conditions, params)
        return (sol[:, 2:4] - data).ravel()
    
    def fit_predict(self):
        days = self.df.shape[0]
        tspan = np.arange(0, days, 1)
        data_sir = self.df.loc[0:(days-1), ['infected', 'total_recovered_or_dead']].values
        initial_conditions = [self.initE, self.initI, self.initR, self.initN]
        result = minimize(self.error, self.params, args=(initial_conditions, tspan, data_sir), method='leastsq')
        final = data_sir + result.residual.reshape(data_sir.shape)
        return final