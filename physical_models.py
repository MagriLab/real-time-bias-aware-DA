from scipy.interpolate import splrep, splev
import pylab as plt

import bias_models
from Util import Cheb, RK4
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
from functools import partial
from copy import deepcopy

# os.environ["OMP_NUM_THREADS"] = '1'
num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)

rng = np.random.default_rng(6)

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general model properties and methods definitions.
    """
    attr_model: dict = dict(t=0., dt=1e-4, precision_t=6,
                            psi0=np.empty(1), alpha0=np.empty(1), psi=None, alpha=None,
                            ensemble=False, filename='', governing_eqns_params=dict())
    attr_ens: dict = dict(filter='EnKF',
                          constrained_filter=False,
                          regularization_factor=1.,
                          m=10,
                          dt_obs=None,
                          est_a=False,
                          est_s=True,
                          est_b=False,
                          biasType=bias_models.NoBias,
                          inflation=1.002,
                          reject_inflation=1.002,
                          std_psi=0.001,
                          std_a=0.001,
                          alpha_distr='normal',
                          ensure_mean=False,
                          num_DA_blind=0,
                          num_SE_only=0,
                          start_ensemble_forecast=0.,
                          get_cost=False,
                          Na=0
                          )
    __slots__ = list(attr_model.keys()) + list(attr_ens.keys()) + ['hist', 'hist_t', 'hist_J', '_pool', '_M', '_Ma']

    def __init__(self, **model_dict):
        # ================= INITIALISE THERMOACOUSTIC MODEL ================== ##
        for key, val in Model.attr_model.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)
        for key, val in Model.attr_ens.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])

        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()
        self.psi = np.array([self.psi0]).T
        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi])
        self.hist_t = np.array([self.t])
        self.hist_J = []
        # ========================== DEFINE LENGTHS ========================== ##
        self.precision_t = int(-np.log10(self.dt)) + 2

    @property
    def Nphi(self):
        return len(self.psi0)

    @property
    def Nq(self):
        return np.shape(self.getObservables())[0]

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    def copy(self):
        return deepcopy(self)

    def reshape_ensemble(self, m=None, reset=True):
        model = self.copy()
        if m is None:
            m = model.m
        psi = model.psi
        if m == 1:
            psi = np.mean(psi, -1, keepdims=True)
            model.ensemble = False
        else:
            psi = model.addUncertainty(np.mean(psi, -1, keepdims=True), np.std(psi, -1, keepdims=True), m)
        model.updateHistory(psi=psi, t=0., reset=reset)
        print(self.psi.shape, m)
        return model

    def getObservableHist(self, Nt=0, **kwargs):
        return self.getObservables(Nt, **kwargs)

    def print_model_parameters(self):
        print('\n ------------------ {} Model Parameters ------------------ '.format(self.name))
        for k in self.attr.keys():
            print('\t {} = {}'.format(k, getattr(self, k)))

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##
    @property
    def M(self):
        if not hasattr(self, '_M'):
            setattr(self, '_M', np.hstack((np.zeros([self.Nq, self.Na+self.Nphi]),
                                           np.eye(self.Nq))))
        return self._M

    @property
    def Ma(self):
        if not hasattr(self, '_Ma'):
            setattr(self, '_Ma', np.hstack((np.zeros([self.Na, self.Nphi]),
                                            np.eye(self.Na),
                                            np.zeros([self.Na, self.Nq]))))
        return self._Ma

    # ------------------------- Functions for update/initialise the model --------------------------- #
    @staticmethod
    def addUncertainty(mean, std, m, method='normal', param_names=None, ensure_mean=False):
        if method == 'normal':
            if isinstance(std, float):
                cov = np.diag((mean * std) ** 2)
            else:
                raise TypeError('std in normal distribution must be float not {}'.format(type(std)))
            ens = rng.multivariate_normal(mean, cov, m).T
        elif method == 'uniform':
            ens = np.zeros((len(mean), m))
            if isinstance(std, float):
                for ii, pp in enumerate(mean):
                    if abs(std) <= .5:
                        ens[ii, :] = pp * (1. + rng.uniform(-std, std, m))
                    else:
                        ens[ii, :] = rng.uniform(pp - std, pp + std, m)
            elif isinstance(std, dict):
                if param_names is not None:
                    for ii, key in enumerate(param_names):
                        ens[ii, :] = rng.uniform(std[key][0], std[key][1], m)
                else:
                    for ii, _ in enumerate(mean):
                        ens[ii, :] = rng.uniform(std[ii][0], std[ii][1], m)
            else:
                raise TypeError('std in normal distribution must be float or dict')
        else:
            raise ValueError('Parameter distribution {} not recognised'.format(method))

        if ensure_mean:
            ens[:, 0] = mean
        return ens

    # def getOutputs(self):
    #     out = dict(name=self.name,
    #                hist_y=self.getObservableHist(),
    #                y_lbls=self.obsLabels,
    #                bias=self.bias.getOutputs(),
    #                hist_t=self.hist_t,
    #                hist=self.hist,
    #                hist_J=self.hist_J,
    #                alpha0=self.alpha0
    #                )
    #     for key in self.attr.keys():
    #         out[key] = getattr(self, key)
    #     if self.ensemble:
    #         for key in self.attr_ens.keys():
    #             out[key] = getattr(self, key)
    #     return out

    def initEnsemble(self, **DAdict):
        DAdict = DAdict.copy()
        self.ensemble = True

        for key, val in Model.attr_ens.items():
            if key in DAdict.keys():
                setattr(self, key, DAdict[key])
            else:
                setattr(self, key, val)
        self.filename += '{}_ensemble_m{}'.format(self.name, self.m)
        if hasattr(self, 'modify_settings'):
            self.modify_settings()
        # --------------- RESET INITIAL CONDITION AND HISTORY --------------- ##
        # Note: if est_a and est_b psi = [psi; alpha; biasWeights]
        ensure_mean = self.ensure_mean
        self.psi0 = np.mean(self.psi, -1)

        mean_psi = self.psi0 * rng.uniform(0.9, 1.1, len(self.psi0))
        new_psi0 = self.addUncertainty(mean_psi, self.std_psi, self.m,
                                       method='normal', ensure_mean=ensure_mean)

        if self.est_a:  # Augment ensemble with estimated parameters
            mean_a = np.array([getattr(self, pp) for pp in self.est_a])
            new_alpha0 = self.addUncertainty(mean_a, self.std_a, self.m, method=self.alpha_distr,
                                             param_names=self.est_a, ensure_mean=ensure_mean)
            new_psi0 = np.vstack((new_psi0, new_alpha0))
            self.Na = len(self.est_a)

        # RESET ENSEMBLE HISTORY
        self.updateHistory(psi=new_psi0, t=0., reset=True)

    def initBias(self, **Bdict):

        if 'biasType' in Bdict.keys():
            self.biasType = Bdict['biasType']

        # Assign some required items
        for key, default_value in zip(['t_val', 't_train', 't_test'],
                                      [self.t_CR, self.t_transient, self.t_CR]):
            if key not in Bdict.keys():
                Bdict[key] = default_value

        # Initialise bias. Note: self.bias is now an instance of the bias class
        self.bias = self.biasType(y=self.getObservables(),
                                  t=self.t, dt=self.dt, **Bdict)
        # Create bias history
        b = self.bias.getBias
        self.bias.updateHistory(b, self.t, reset=True)

    def updateHistory(self, psi=None, t=None, reset=False):
        if not reset:
            self.hist = np.concatenate((self.hist, psi), axis=0)
            self.hist_t = np.hstack((self.hist_t, t))
        else:
            if psi is None:
                psi = np.array([self.psi0]).T
            if t is None:
                t = self.t
            self.hist = np.array([psi])
            self.hist_t = np.array([t])

        self.psi = self.hist[-1]
        self.t = self.hist_t[-1]

    def is_not_physical(self, print_=False):
        if not hasattr(self, '_physical'):
            self._physical = 0
        if print_:
            print('Number of non-physical analysis = ', self._physical)
        else:
            self._physical += 1

    # -------------- Functions required for the forecasting ------------------- #
    @property
    def pool(self):
        if not hasattr(self, '_pool'):
            self._pool = mp.Pool()
        return self._pool

    def close(self):
        if hasattr(self, '_pool'):
            self.pool.close()
            self.pool.join()
            delattr(self, "_pool")
        else:
            pass

    def getAlpha(self, psi=None):
        alpha = []
        if psi is None:
            psi = self.psi
        for mi in range(psi.shape[-1]):
            ii = -self.Na
            alph = self.alpha0.copy()
            for param in self.est_a:
                alph[param] = psi[ii, mi]
                ii += 1
            alpha.append(alph)
        return alpha

    @staticmethod
    def forecast(y0, fun, t, params):
        # SOLVE IVP ========================================
        assert len(t) > 1

        part_fun = partial(fun, **params)

        out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45')
        psi = out.y.T

        # ODEINT =========================================== THIS WORKS AS IF HARD CODED
        # psi = odeint(fun, y0, t_interp, (params,))
        #
        # HARD CODED RUGGE KUTTA 4TH ========================
        # psi = RK4(t_interp, y0, fun, params)
        return psi

    def timeIntegrate(self, Nt=100, averaged=False, alpha=None):
        """
            Integrator of the model. If the model is forcast as an ensemble, it uses parallel computation.
            Args:
                Nt: number of forecast steps
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        t = np.round(self.t + np.arange(0, Nt+1) * self.dt, self.precision_t)
        args = self.governing_eqns_params

        if not self.ensemble:

            psi = [Model.forecast(y0=self.psi[:, 0], fun=self.timeDerivative, t=t, params={**self.alpha0, **args})]

        else:
            if not averaged:
                alpha = self.getAlpha()
                forecast_part = partial(Model.forecast, fun=self.timeDerivative, t=t)
                sol = [self.pool.apply_async(forecast_part,
                                             kwds={'y0': self.psi[:, mi].T, 'params':{**args, **alpha[mi]}})
                       for mi in range(self.m)]
                psi = [s.get() for s in sol]
            else:
                psi_mean0 = np.mean(self.psi, 1, keepdims=True)
                psi_deviation = self.psi - psi_mean0

                if alpha is None:
                    alpha = self.getAlpha(psi_mean0)[0]
                psi_mean = Model.forecast(y0=psi_mean0[:, 0], fun=self.timeDerivative, t=t,
                                          params={**alpha, **args})

                if np.mean(np.std(self.psi[:len(self.psi0)]/np.array([self.psi0]).T, axis=0)) < 2.:
                    psi_deviation /= psi_mean0
                    psi = [psi_mean * (1 + psi_deviation[:, ii]) for ii in range(self.m)]
                else:
                    psi = [psi_mean + psi_deviation[:, ii] for ii in range(self.m)]

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        psi = np.array(psi).transpose((1, 2, 0))
        return psi[1:], t[1:]


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
    attr: dict = dict(dt=1.0E-4, t_transient=1.5, t_CR=0.04,
                      omega=2 * np.pi * 120., law='tan',
                      zeta=60., beta=70., kappa=4.0, gamma=1.7)  # beta, zeta [rad/s]

    params: list = ['beta', 'zeta', 'kappa']  # ,'omega', 'gamma']
    param_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')

    fixed_params = ['law', 'omega']

    # __________________________ Init method ___________________________ #
    def __init__(self, **TAdict):
        model_dict = dict(TAdict)
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)

        super().__init__(**model_dict)
        if 'psi0' not in model_dict.keys():
            self.psi0 = [0.1, 0.1]  # initialise eta and mu
            self.updateHistory(reset=True)

        # _________________________ Add fixed parameters  ________________________ #
        self.set_fixed_params()

    def set_fixed_params(self):
        for key in VdP.fixed_params:
            self.governing_eqns_params[key] = getattr(self, key)

    # _______________ VdP specific properties and methods ________________ #
    @property
    def param_lims(self):
        return dict(zeta=(5, 120),
                    kappa=(0.1, 20),
                    beta=(5, 120),
                    gamma=(0., 5.)
                    )

    @property
    def obsLabels(self):
        return "$\\eta$"

    def getObservables(self, Nt=1):
        if Nt == 1:  # required to reduce from 3 to 2 dimensions
            return self.hist[-1, :1, :]
        else:
            return self.hist[-Nt:, :1, :]

    @staticmethod
    def timeDerivative(t, psi, beta, zeta, kappa, law, omega):
        eta, mu = psi[:2]
        dmu_dt = - omega ** 2 * eta + mu * (beta - zeta)
        # Add nonlinear term
        if law == 'cubic':  # Cubic law
            dmu_dt -= mu * kappa * eta ** 2
        elif law == 'tan':  # arc tan model
            dmu_dt -= mu * (kappa * eta ** 2) / (1. + kappa / beta * eta ** 2)

        return (mu, dmu_dt) + (0,) * (len(psi) - 2)


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretisation and gain-delay sqrt heat release law.
        Args:
            TAdict: dictionary with the model parameters. If not defined, the default value is used.
                > Nm - Number of Galerkin modes
                > Nc - Number of Chebyshev modes
                > beta - Heat source strength [-]
                > tau - Time delay [s]
                > C1 - First damping constant [-]
                > C2 - Second damping constant [-]
                > xf - Flame location [m]
                > L - Tube length [m]
    """

    name: str = 'Rijke'
    attr: dict = dict(dt=1E-4, t_transient=1., t_CR=0.02,
                      Nm=10, Nc=10, Nmic=6,
                      beta=4.0, tau=1.5E-3, C1=.05, C2=.01, kappa=1E5,
                      xf=0.2, L=1., law='sqrt')
    params: list = ['beta', 'tau', 'C1', 'C2', 'kappa']
    param_labels = dict(beta='$\\beta$', tau='$\\tau$', C1='$C_1$', C2='$C_2$', kappa='$\\kappa$')

    fixed_params = ['cosomjxf', 'Dc',  'gc', 'jpiL', 'L', 'law', 'meanFlow', 'Na', 'Nm', 'tau_adv', 'sinomjxf']

    def __init__(self, **TAdict):

        model_dict = dict(TAdict)
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)

        super().__init__(**model_dict)

        self.tau_adv = self.tau
        if 'psi0' not in TAdict.keys():
            self.psi0 = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])
            # self.resetInitialConditions()
            self.updateHistory(reset=True)
        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

        # ------------------------------------------------------------------------------------- #

        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nmic + 1)[:-1]

        # Define modes frequency of each mode and sin cos etc
        jj = np.arange(1, self.Nm + 1)
        self.jpiL = jj * np.pi / self.L
        self.sinomjxf = np.sin(self.jpiL * self.xf)
        self.cosomjxf = np.cos(self.jpiL * self.xf)

        # Mean Flow Properties
        def weight_avg(y1, y2):
            return self.xf / self.L * y1 + (1. - self.xf / self.L) * y2

        self.meanFlow = dict(u=weight_avg(10, 11.1643),
                             p=101300.,
                             gamma=1.4,
                             T=weight_avg(300, 446.5282),
                             R=287.1
                             )
        self.meanFlow['rho'] = self.meanFlow['p'] / (self.meanFlow['R'] * self.meanFlow['T'])
        self.meanFlow['c'] = np.sqrt(self.meanFlow['gamma'] * self.meanFlow['R'] * self.meanFlow['T'])

        self.set_fixed_params()

        # Wave parameters ############################################################################################
        # c1: 347.2492    p1: 1.0131e+05      rho1: 1.1762    u1: 10          M1: 0.0288          T1: 300
        # c2: 423.6479    p2: 101300          rho2: 0.7902    u2: 11.1643     M2: 0.0264          T2: 446.5282
        # Tau: 0.0320     Td: 0.0038          Tu: 0.0012      R_in: -0.9970   R_out: -0.9970      Su: 0.9000
        # Qbar: 5000      R_gas: 287.1000     gamma: 1.4000
        ##############################################################################################################


    def modify_settings(self):
        if self.est_a and 'tau' in self.est_a:
            extra_Nc = self.Nc - 50
            self.tau_adv, self.Nc = 1E-2, 50
            self.psi0 = np.hstack([self.psi0, np.zeros(extra_Nc)])
            # self.resetInitialConditions()
            self.updateHistory(reset=True)
            self.set_fixed_params()

    # _________________________ Governing equations ________________________ #
    def set_fixed_params(self):
        for key in Rijke.fixed_params:
            self.governing_eqns_params[key] = getattr(self, key)

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def param_lims(self):
        return dict(beta=(0.01, 5),
                    tau=(1E-6, self.tau_adv),
                    C1=(0., 1.),
                    C2=(0., 1.),
                    kappa=(1E3, 1E8)
                    )

    @property
    def obsLabels(self, loc=None, velocity=False):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        if not velocity:
            return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]
        else:
            return [["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()],
                    ["$u'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]]

    def getObservables(self, Nt=1, loc=None):
        if loc is None:
            loc = self.x_mic
        loc = np.expand_dims(loc, axis=1)
        om = np.array([self.jpiL])
        mu = self.hist[-Nt:, self.Nm:2 * self.Nm, :]

        # Compute acoustic pressure and velocity at locations
        p_mic = -np.dot(np.sin(np.dot(loc, om)), mu)
        p_mic = p_mic.transpose(1, 0, 2)
        if Nt == 1:
            p_mic = p_mic[0]
        return p_mic


    @staticmethod
    def timeDerivative(t, psi,
                       C1, C2, beta, kappa, tau,  # Possibly-inferred parameters
                       cosomjxf, Dc,  gc, jpiL, L, law, meanFlow, Na, Nm, tau_adv, sinomjxf  # fixed_params
                       ):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
            Returns:
                concatenation of the state vector time derivative
        """

        eta = psi[:Nm]
        mu = psi[Nm:2 *Nm]
        v = psi[2 * Nm:len(psi)-Na]

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, cosomjxf), v))

        # Evaluate u(t_interp-tau) i.e. velocity at the flame at t_interp - tau
        x_tau = tau / tau_adv
        if x_tau < 1:
            f = splrep(gc, v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            raise Exception("tau = {} can't_interp be larger than tau_adv = {}".format(tau, tau_adv))

        # Compute damping and heat release law
        zeta = C1 * (jpiL * L / np.pi) ** 2 + C2 * (jpiL * L / np.pi) ** .5

        MF = meanFlow.copy()  # Physical properties
        if law == 'sqrt':
            q_dot = MF['p'] * MF['u'] * beta * (np.sqrt(abs(1./3 + u_tau / MF['u'])) - np.sqrt(1./3))  # [W/m2]=[m/s3]
        elif law == 'tan':
            q_dot = beta * np.sqrt(beta / kappa) * np.arctan(np.sqrt(beta / kappa) * u_tau)  # [m / s3]
        else:
            raise ValueError('Law "{}" not defined'.format(law))
        q_dot *= -2. * (MF['gamma'] - 1.) / L * sinomjxf  # [Pa/s]

        # governing equations
        deta_dt = jpiL/ MF['rho'] * mu
        dmu_dt = - jpiL * MF['gamma'] * MF['p'] * eta - MF['c'] / L * zeta * mu + q_dot
        dv_dt = - 2. / tau_adv * np.dot(Dc, v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(Na)))


# %% =================================== LORENZ 63 MODEL ============================================== %% #
class Lorenz63(Model):
    """ Lorenz 63 Class
    """

    name: str = 'Lorenz63'
    attr: dict = dict(rho=28., sigma=10., beta=8. / 3., dt=0.02, t_CR=5.)

    params: list = ['rho', 'sigma', 'beta']
    param_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')

    fixed_params = []

    # __________________________ Init method ___________________________ #
    def __init__(self, **TAdict):

        model_dict = dict(TAdict)
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)

        super().__init__(**model_dict)

        if 'psi0' not in TAdict.keys():
            self.psi0 = [1.0, 1.0, 1.0]  # initialise x, y, z
            self.updateHistory(reset=True)

        # set limits for the parameters
        self.param_lims = dict(rho=(None, None), beta=(None, None), sigma=(None, None))

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obsLabels(self):
        return ["$x$", '$y$', '$z$']

    def getObservables(self, Nt=1):
        if Nt == 1:
            return self.hist[-1, :, :]
        else:
            return self.hist[-Nt:, :, :]

    @staticmethod
    def timeDerivative(t, psi, sigma, rho, beta):
        x1, x2, x3 = psi[:3]
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)


# %% =================================== 2X VAN DER POL MODEL ============================================== %% #
class Annular(Model):
    """ Annular combustor model, wich consists of two coupled oscillators
    """

    name: str = 'Annular'
    attr: dict = dict(dt=1 / 51.2E3, t_transient=0.5, t_CR=0.03,
                      n=1., theta_b=0.63, theta_e=0.66, omega=1090., epsilon=0.0023,
                      nu=17., beta_c2=17., kappa=1.2E-4)  # values in Fig.4

    # attr['nu'], attr['beta_c2'] = 30., 5.  # spin
    # attr['nu'], attr['beta_c2'] = 1., 25.  # stand
    attr['nu'], attr['beta_c2'] = 20., 18.  # mix

    params: list = ['omega', 'nu', 'beta_c2', 'kappa', 'epsilon', 'theta_b', 'theta_e']

    param_labels = dict(omega='$\\omega$', nu='$\\nu$', beta_c2='$c_2\\beta $', kappa='$\\kappa$',
                        epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')

    fixed_params = []

    # __________________________ Init method ___________________________ #
    def __init__(self, **TAdict):

        model_dict = dict(TAdict)
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)

        super().__init__(**TAdict)

        self.theta_mic = np.radians([0, 60, 120, 240])

        if 'psi0' not in TAdict.keys():
            self.psi0 = [100, -10, -100, 10]  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}
            # self.resetInitialConditions()
            self.updateHistory(reset=True)

    # set limits for the parameters
    @property
    def param_lims(self):
        return dict(omega=(1000, 1300),
                    nu=(-40., 60.),
                    beta_c2=(1., 60.),
                    kappa=(None, None),
                    epsilon=(None, None),
                    theta_b=(0, 2 * np.pi),
                    theta_e=(0, 2 * np.pi)
                    )

    # _______________  Specific properties and methods ________________ #
    # @property
    # def obsLabels(self):
    #     return ["\\eta_1", '\\eta_2']

    def getObservables(self, Nt=1, loc=None, modes=False):
        """
        :return: pressure measurements at theta = [0º, 60º, 120º, 240º`]
        p(θ, t) = η1(t) * cos(nθ) + η2(t) * sin(nθ).
        """
        if loc is None:
            loc = self.theta_mic

        if modes:
            self.obsLabels = ["$\\eta_1$", '$\\eta_2$']
            return self.hist[-Nt:, [0, 2], :]

        self.obsLabels = ["$p(\\theta={})$".format(int(np.round(np.degrees(th)))) for th in loc]

        loc = np.array(loc)
        eta1, eta2 = self.hist[-Nt:, 0, :], self.hist[-Nt:, 2, :]

        if max(loc) > 2 * np.pi:
            raise ValueError('Theta must be in radians')

        p_mics = np.array([eta1 * np.cos(th) + eta2 * np.sin(th) for th in loc]).transpose(1, 0, 2)

        if Nt == 1:
            return p_mics.squeeze(axis=0)
        else:
            return p_mics

    @staticmethod
    def timeDerivative(t, psi, nu, kappa, beta_c2, theta_b, omega, epsilon, theta_e):
        y_a, z_a, y_b, z_b = psi[:4]  # y = η, and z = dη/dt

        def k1(y1, y2, sign):
            return 2 * nu - 3. / 4 * kappa * (3 * y1 ** 2 + y2 ** 2) + sign / 2. * beta_c2 * np.cos(2. * theta_b)

        k2 = 0.5 * beta_c2 * np.sin(2. * theta_b) - 3. / 2 * kappa * y_a * y_b

        def k3(y1, y2, sign):
            return omega ** 2 * (y1 + epsilon / 2. * (sign * y1 * np.cos(2. * theta_e) + y2 * np.sin(2. * theta_e)))

        dz_a = z_a * k1(y_a, y_b, 1) + z_b * k2 - k3(y_a, y_b, 1)
        dz_b = z_b * k1(y_b, y_a, -1) + z_a * k2 - k3(y_b, y_a, -1)

        return (z_a, dz_a, z_b, dz_b) + (0,) * (len(psi) - 4)


if __name__ == '__main__':

    t0 = time.time()
    # Ensemble case =============================
    case = VdP(beta=80)
    DA_params = dict(m=10, est_a=['beta'], std_psi=0.1, std_a=dict(beta=(70, 90)), alpha_distr='uniform')
    state, t_ = case.timeIntegrate(int(case.t_transient / case.dt))
    case.updateHistory(state, t_)

    case.initEnsemble(**DA_params)

    t1 = time.time()
    for _ in range(5):
        state, t_ = case.timeIntegrate(int(1. / case.dt))
        case.updateHistory(state, t_)
    # for _ in range(5):
    #     state, t_ = case.timeIntegrate(int(.1 / case.dt), averaged=True)
    #     case.updateHistory(state, t_)


    print('Elapsed time = ', str(time.time() - t1))

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    _, ax = plt.subplots(1, 3, figsize=[15, 5])
    plt.suptitle('Ensemble case')
    # State evolution
    y, lbl = case.getObservableHist(), case.obsLabels

    ax[0].plot(t_h, y[:, 0], color='blue', label=lbl)
    i, j = [0, 1]
    ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='blue')

    ax[0].set(xlabel='t', ylabel=lbl, xlim=[t_h[0], t_h[-1]])
    ax[1].set(xlabel='t', xlim=[t_h[-t_zoom], t_h[-1]])

    # Params

    ai = -case.Na
    max_p, min_p = -1000, 1000
    c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    mean = np.mean(case.hist, -1, keepdims=True)
    for p in case.est_a:
        # reference_p = truth['true_params']
        reference_p = case.alpha0

        mean_p = mean[:, ai].squeeze() / reference_p[p]
        std = np.std(case.hist[:, ai] / reference_p[p], axis=1)

        max_p = max(max_p, max(mean_p))
        min_p = min(min_p, min(mean_p))

        ax[2].plot(t_h, mean_p, color=c[-ai], label='$\\{0}/\\{0}^{1}$'.format(p, '\\mathrm{init}'))
        ax[2].fill_between(t_h, mean_p + std, mean_p - std, alpha=0.2, color=c[-ai])
        ax[2].plot(t_h, case.hist[:, ai] / reference_p[p], lw=.5, color=c[-ai])

        ax[2].set(xlabel='$t$', xlim=[t_h[0], t_h[-1]])
        ai += 1
    ax[2].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
    ax[2].plot(t_h[1:], t_h[1:] / t_h[1:], '-', color='k', linewidth=.5)
    ax[2].set(ylim=[min_p - 0.1, max_p + 0.1])

    plt.tight_layout()

    # import pympler.asizeof


    # print('time={}, size={}'.format(time.time() - t0, pympler.asizeof.asizeof(case)))

    plt.show()