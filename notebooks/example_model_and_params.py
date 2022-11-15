"""Model 
"""
import random
import numpy as np
from neural.basemodel import Model
from neural.network import Network
from neural.network.operator import Operator, Repeat, Add
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as garray
import skcuda
import skcuda.misc

class PostLNSyn(Model):
    Default_States = dict(
        x1=(0.0, 0.0, 1.0), x2=(0.0, 0.0, 1.0), x3=(1.0, 0.0, 1.0), I=0.0
    )
    Default_Params = dict(
        a1=1.0,
        b1=1.0,
        k=10.0,
        a2=1.0,
        b2=1.0,
        a3=1.0,
        b3=1.0,
        gmax=1.0,
    )

    def ode(self, u=0.0):
        self.d_x1 = self.a1 * self.x2 * (1.0 - self.x1) - self.b1 * self.x1- self.k * self.x1 * self.x3
        self.d_x2 = self.a2 * u * (1.0 - self.x2) - self.b2 * self.x2
        self.d_x3 = self.a3 * u * (1.0 - self.x3) - self.b3 * self.x3
        self.I = self.x1 * self.gmax


class PreLN(Operator):
    """BlockMean +  Repeat"""

    def __init__(self, operation="mean", block_size=1, **kwargs):
        self.operation = operation
        self.block_size = block_size
        mean_output_size = int(kwargs["size"] // block_size)
        super().__init__(**kwargs)
        self.meaned = garray.empty(mean_output_size, dtype=self.dtype)

        self._repeat_kernel = ElementwiseKernel(
            "{0} *output, {0} *input".format(self.dtype),
            "output[i] = input[i / {}]".format(self.block_size),
            "Repeat",
        )

    def update(self, input):
        if np.isscalar(input):
            input = np.array([input])
        _input = input.reshape(-1, self.block_size)
        if self._backend == "cuda":
            if self.operation == "mean":
                skcuda.misc.mean(_input, out=self.meaned, axis=1)
            elif self.operation == "sum":
                skcuda.misc.sum(_input, out=self.meaned, axis=1)
            elif "norm" in self.operation:
                degree = float(self.operation.split("norm-")[-1])
                skcuda.misc.sum((_input ** degree), out=self.meaned, axis=1)
                self.meaned = self.meaned ** (1./degree)
            else:
                raise TypeError
            self._repeat_kernel(self.output, self.meaned)
        else:
            raise NotImplementedError


class dDNP(Model):
    Default_States = dict(
        x1=(0.0, 0.0, 1.0),
        I=0.
    )
    Default_Params = dict(
        a1=10.0,
        b1=1.0,
        k=10.0,
        gmax=1.
    )

    def ode(self, u=0.0, l=0.0):
        self.d_x1 = self.a1 * u * (1.0 - self.x1) - self.b1 * self.x1 - self.k * self.x1 * l
        self.I = self.gmax * self.x1


class PoissonCSN0(Model):
    Default_States = dict(
        spike_rate=(0.0, 0.0, 350.0), x=(0.0, 0.0, 1.0), r=0.0, spike=0.0, cx=0.0
    )
    Default_Params = dict(
        x7=7.06672200e-12,
        x6=-4.41125703e-09,
        x5=1.14814892e-06,
        x4=-1.61863859e-04,
        x3=1.34591675e-02,
        x2=-6.79691186e-01,
        x1=2.25636389e01,
        x0=-1.36299580e02,
    )

    def ode(self, I=0.0):
        self.x = 0.0
        Ip = 1.0
        self.d_x = Ip * self.x0
        Ip = Ip * I
        self.d_x += Ip * self.x1
        Ip = Ip * I
        self.d_x += Ip * self.x2
        Ip = Ip * I
        self.d_x += Ip * self.x3
        Ip = Ip * I
        self.d_x += Ip * self.x4
        Ip = Ip * I
        self.d_x += Ip * self.x5
        Ip = Ip * I
        self.d_x += Ip * self.x6
        Ip = Ip * I
        self.d_x += Ip * self.x7
        self.r = random.uniform(0.0, 1.0)

    def post(self):
        self.spike = self.r < self.x
        self.cx += self.x


class PoissonCSN(Model):
    Default_States = dict(x=0.0, y=0.0, r=0.0, spike=0.0, cx=0.0)
    Default_Params = dict(
        x6=2.79621009e-09,
        x5=-9.55636291e-07,
        x4=1.25880567e-04,
        x3=-7.79496241e-03,
        x2=1.94672932e-01,
        x1=3.44246777,
        x0=5.11085315,
    )

    def ode(self, I=0.0):

        self.x = 0.0
        Ip = 1.0
        self.d_x = Ip * self.x0
        Ip = Ip * I
        self.d_x += Ip * self.x1
        Ip = Ip * I
        self.d_x += Ip * self.x2
        Ip = Ip * I
        self.d_x += Ip * self.x3
        Ip = Ip * I
        self.d_x += Ip * self.x4
        Ip = Ip * I
        self.d_x += Ip * self.x5
        Ip = Ip * I
        self.d_x += Ip * self.x6
        self.y = self.d_x

        self.r = random.uniform(0.0, 1.0)

    def post(self):
        self.spike = self.r < self.x
        self.cx += self.x

class PoissonCSN1(Model):
    Default_States = dict(
        x=(0.0, 0., 1.), 
        r=0.0, 
        spike=0.0, 
        cx=0.0
    )            
    Default_Params = dict(
        x8=  2.62834699e-12,
        x7= -1.18820316e-09,
        x6=  2.24914051e-07,
        x5= -2.30971695e-05,
        x4=  1.38994273e-03,
        x3= -4.88554093e-02,
        x2=  8.87442880e-01,
        x1= -6.90178752e-01,
        x0=  8.95839969e-01,
    )

    def ode(self, I=0.0):
        self.x = 0.0
        Ip = 1.0
        self.d_x = Ip * self.x0
        Ip = Ip * I
        self.d_x += Ip * self.x1
        Ip = Ip * I
        self.d_x += Ip * self.x2
        Ip = Ip * I
        self.d_x += Ip * self.x3
        Ip = Ip * I
        self.d_x += Ip * self.x4
        Ip = Ip * I
        self.d_x += Ip * self.x5
        Ip = Ip * I
        self.d_x += Ip * self.x6
        Ip = Ip * I
        self.d_x += Ip * self.x7
        Ip = Ip * I
        self.d_x += Ip * self.x8
        self.r = random.uniform(0.0, 1.0)

    def post(self):
        self.spike = self.r < self.x
        self.cx += self.x

class OTP(Model):
    Default_States = dict(
        v=(0.0, 0, 1e9),
        I=0.0,
        uh=(0.0, 0.0, 1e9),
        duh=0.0,
        x1=(0.0, 0.0, 1.0),
        x2=(0.0, 0.0, 1.0),
        x3=(0.0, 0.0, 1000.0),
    )
    Default_Params = dict(
        br=1.0,
        dr=1.0,
        gamma=0.215,
        b1=0.8,
        a1=45.0,
        a2=146.1,
        b2=117.2,
        a3=2.539,
        b3=0.9096,
        kappa=8841,
        p=1.0,
        c=0.06546,
        Imax=85.0,
    )

    def ode(self, stimulus=0.0):
        self.d_x1 = self.br * self.v * (1.0 - self.x1) - self.dr * self.x1
        f = np.cbrt(self.x2 * self.x2) * np.cbrt(self.x3 * self.x3)
        self.d_x2 = self.a2 * self.x1 * (1.0 - self.x2) - self.b2 * self.x2 - self.kappa * f
        self.d_x3 = self.a3 * self.x2 - self.b3 * self.x3

        self.I = self.Imax * self.x2 / (self.x2 + self.c)

        self.d_uh = self.duh
        self.d_duh = -2 * self.a1 * self.b1 * self.duh + self.a1 * self.a1 * (stimulus - self.uh)
        self.v = self.uh + self.gamma * self.duh


class IonSyn(Model):
    Default_States = dict(
        x1=(0.0, 0.0, 1.0),
        I=0.0,
    )
    Default_Params = dict(
        a1=10.0,
        b1=1.0,
        gmax=10.0,
    )

    def ode(self, u=0.0):
        self.d_x1 = self.a1 * u * (1.0 - self.x1) - self.b1 * self.x1
        self.I = self.x1 * self.gmax
        
class NoisyConnorStevens(Model):
    """
    Connor-Stevens Model
    """

    Time_Scale = 1e3  # s to ms
    Default_States = dict(
        v=(-60, -80, 80),
        n=(0.0, 0.0, 1.0),
        m=(0.0, 0.0, 1.0),
        h=(1.0, 0.0, 1.0),
        a=(1.0, 0.0, 1.0),
        b=(1.0, 0.0, 1.0),
        spike=0,
        v1=-60.0,
        v2=-60.0,
        refactory=0.0,
    )
    Default_Params = dict(
        ms=-5.3,
        ns=-4.3,
        hs=-12.0,
        gNa=120.0,
        gK=20.0,
        gL=0.3,
        ga=47.7,
        ENa=55.0,
        EK=-72.0,
        EL=-17.0,
        Ea=-75.0,
        sigma=2.05,
        refperiod=1.5,
    )

    def ode(self, I=0.0):

        alpha = np.exp(-(self.v + 50.0 + self.ns) / 10.0) - 1.0
        if abs(alpha) <= 1e-7:
            alpha = 0.1
        else:
            alpha = -0.01 * (self.v + 50.0 + self.ns) / alpha
        beta = 0.125 * np.exp(-(self.v + 60.0 + self.ns) / 80.0)
        n_inf = alpha / (alpha + beta)
        tau_n = 2.0 / (3.8 * (alpha + beta))

        alpha = np.exp(-(self.v + 35.0 + self.ms) / 10.0) - 1.0
        if abs(alpha) <= 1e-7:
            alpha = 1.0
        else:
            alpha = -0.1 * (self.v + 35.0 + self.ms) / alpha
        beta = 4.0 * np.exp(-(self.v + 60.0 + self.ms) / 18.0)
        m_inf = alpha / (alpha + beta)
        tau_m = 1.0 / (3.8 * (alpha + beta))

        alpha = 0.07 * np.exp(-(self.v + 60.0 + self.hs) / 20.0)
        beta = 1.0 / (1.0 + np.exp(-(self.v + 30.0 + self.hs) / 10.0))
        h_inf = alpha / (alpha + beta)
        tau_h = 1.0 / (3.8 * (alpha + beta))

        a_inf = np.cbrt(0.0761 * np.exp((self.v + 94.22) / 31.84) / (1.0 + np.exp((self.v + 1.17) / 28.93)))
        tau_a = 0.3632 + 1.158 / (1.0 + np.exp((self.v + 55.96) / 20.12))
        b_inf = np.power(1 / (1 + np.exp((self.v + 53.3) / 14.54)), 4.0)
        tau_b = 1.24 + 2.678 / (1 + np.exp((self.v + 50) / 16.027))

        i_na = self.gNa * np.power(self.m, 3) * self.h * (self.v - self.ENa)
        i_k = self.gK * np.power(self.n, 4) * (self.v - self.EK)
        i_l = self.gL * (self.v - self.EL)
        i_a = self.ga * np.power(self.a, 3) * self.b * (self.v - self.Ea)

        self.d_v = I - i_na - i_k - i_l - i_a
        self.d_n = (n_inf - self.n) / tau_n + random.gauss(0.0, self.sigma)
        self.d_m = (m_inf - self.m) / tau_m + random.gauss(0.0, self.sigma)
        self.d_h = (h_inf - self.h) / tau_h + random.gauss(0.0, self.sigma)
        self.d_a = (a_inf - self.a) / tau_a + random.gauss(0.0, self.sigma)
        self.d_b = (b_inf - self.b) / tau_b + random.gauss(0.0, self.sigma)

        self.d_refactory = self.refactory < 0

    def post(self):
        self.spike = (self.v1 < self.v2) * (self.v < self.v2) * (self.v > -30.0)
        self.v1 = self.v2
        self.v2 = self.v
        self.spike = (self.spike > 0.0) * (self.refactory >= 0)
        self.refactory -= (self.spike > 0.0) * self.refperiod

DEFAULT_PARAMS = {
    "osn_axt": {
        "a1": 101.6,
        "b1": 3.12,
        "k": 1e5
    },
    "osn-to-preln": {
        "a1": 1.06,
        "b1": 15.5,
        "gmax": 3232.6,
    },
    "osn-to-posteln": {
        'a1': 2.858053089117711,
        'b1': 1.4540529550164887,
        'k': 323039.26691518247,
        'a2': 1.5028414134309191,
        'b2': 98.82274646620965,
        'a3': 8.583048572622092,
        'b3': 5.9266873782378555,
        'gmax': 6.791877e+07
        # 'a1': 1.6715276978509699,
        # 'b1': 14.531396405401471,
        # 'k': 246899.115294319,
        # 'a2': 8.790437137355001,
        # 'b2': 37.217816956396454,
        # 'a3': 71.89118189388185,
        # 'b3': 2.8568639066323342,
        # 'gmax': 91451893.88946372
    },
    "osn-to-postiln": {
        # 'a1': 12.537946628371419,
        # 'b1': 3.5533912946105644,
        # 'k': 850654.7752613979,
        # 'a2': 64.39457099988905,
        # 'b2': 14.7847156711102,
        # 'a3': 4.946413745118113,
        # 'b3': 23.05951238421643,
        # 'gmax': 2.463306e+04
        'a1': 9.038009011043401,
        'b1': 81.30204837646579,
        'k': 384165.1012001436,
        'a2': 2.094005529791446,
        'b2': 3.657942885692797,
        'a3': 81.35091649546338,
        'b3': 85.33405338413093,
        'gmax': 516779.02440091287
    },
    "osn_axt-to-pn": {
        'a1':1.,
        'b1':100.,
        'gmax':1.273e5 * 9.63512603e-01
    },
    "posteln-to-pn": {
        'a1':1.,
        'b1':100.,
        'gmax':20.85 * 1.03949467e+04
    },
    "postiln-to-pn": {
        'a1':1.,
        'b1':100.,
        'gmax':89.5 * -2.87216219e+02
    },
    "osn_bsg": {
        "sigma": 0.0025
    },
    "pn_bsg": {
        "sigma": 0.0014
    },
    "preln_bsg": {
        "sigma": 0.
    },
    "posteln_bsg": {
        "sigma": 0.
    },
    "postiln_bsg": {
        "sigma": 0.
    }
}