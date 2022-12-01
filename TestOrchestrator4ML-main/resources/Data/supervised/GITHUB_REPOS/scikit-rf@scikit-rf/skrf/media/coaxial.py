# -*- coding: utf-8 -*-
'''
.. module:: skrf.media.coaxial

============================================================
coaxial (:mod:`skrf.media.coaxial`)
============================================================

A coaxial transmission line defined from its electrical or geometrical/physical properties

.. autosummary::
   :toctree: generated/

   Coaxial

'''

#from copy import deepcopy
from scipy.constants import  epsilon_0, mu_0, pi, c
from numpy import sqrt, log, real, imag, exp, size
from ..tlineFunctions import surface_resistivity
from .distributedCircuit import DistributedCircuit
from .media import Media, DefinedGammaZ0
from ..constants import INF
from ..mathFunctions import feet_2_meter, db_per_100feet_2_db_per_100meter, db_2_np


class Coaxial( DistributedCircuit,Media ):
    '''
    A coaxial transmission line defined in terms of its inner/outer
    diameters and permittivity



    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object

    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if z0 is None then will default to Z0
    Dint : number, or array-like
        inner conductor diameter, in m
    Dout : number, or array-like
        outer conductor diameter, in m
    epsilon_r=1 : number, or array-like
        relative permittivity of the dielectric medium
    tan_delta=0 : number, or array-like
        loss tangent of the dielectric medium
    sigma=infinity : number, or array-like
        conductors electrical conductivity, in S/m


    TODO : different conductivity in case of different conductor kind

    Notes
    ----------
    Dint, Dout, epsilon_r, tan_delta, sigma can all be vectors as long
    as they are the same length

    References
    ---------
    .. [#] Pozar, D.M.; , "Microwave Engineering", Wiley India Pvt. Limited, 1 sept. 2009



        '''
    ## CONSTRUCTOR
    def __init__(self, frequency=None,  z0=None, Dint=.81e-3,
                 Dout=5e-3, epsilon_r=1, tan_delta=0, sigma=INF,
                 *args, **kwargs):



        Media.__init__(self, frequency=frequency,z0=z0)

        self.Dint, self.Dout = Dint,Dout
        self.epsilon_r, self.tan_delta, self.sigma = epsilon_r, tan_delta, sigma
        self.epsilon_prime = epsilon_0*self.epsilon_r
        self.epsilon_second = epsilon_0*self.epsilon_r*self.tan_delta

    @classmethod
    def from_attenuation_VF(cls, frequency=None, z0=None, Z0=50,
                         att=0, unit='dB/m', VF=1):
        """
        Init from electrical properties of the line: attenuation and velocity factor.

        Attenuation can be expressed in dB/m, dB/100m, dB/ft, dB/100ft, Neper/m or Neper/ft.
        Default unit is dB/m. A different unit is set by the `unit` parameter.

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency` object

        z0 : number, array-like, or None
            the port impedance for media. Only needed if its different
            from the characterisitc impedance of the transmission
            line. if z0 is None then will default to Z0
        Z0 : number
            desired characteristic impedance
        att : number, or array-like. optional
            Attenuation of the coaxial line. The default is 0.
            If passed as an array, should be of same size than the frequency.
        unit : string, optional
            Unit of the attenuation. Can be: 'dB/m', dB/100m'', 'dB/ft', 'dB/100ft',
            'Neper/m' or 'Neper/ft' (or 'Np/m', 'Np/ft'). The default is 'dB/m'.
        VF : number, or array-like. optional
            Velocity Factor VF [VF]_. The default is 1.
            If passed as an array, should be of same size than the frequency.

        References
        ----------
        .. [VF] : https://www.microwaves101.com/encyclopedias/light-phase-and-group-velocities

        """
        # test size of parameters
        if size(att) not in (1, size(frequency)):
            raise ValueError('Attenuation should be scalar or of same size that the frequency.')
    
        # create gamma
        if unit in ('dB/m', 'db/m'):
            alpha = db_2_np(att)
        elif unit in ('dB/100m', 'db/100m'):
            alpha = db_2_np(att/100)
        elif unit in ('dB/ft', 'dB/feet'):
            alpha = db_2_np(att/feet_2_meter())
        elif unit in ('dB/100ft', 'dB/100feet'):
            alpha = db_2_np(db_per_100feet_2_db_per_100meter(att)/100)
        elif unit in ('Np/m', 'np/m', 'n/m'):
            alpha = att
        elif unit in ('Neper/feet', 'Neper/ft',
                      'Np/feet', 'Np/ft',
                      'np/feet', 'np/ft',
                      'N/feet', 'N/ft'):
            alpha = att/feet_2_meter()
        else:
            raise ValueError('Incorrect attenuation unit. Please see documentation. ', unit)

        beta = 2 * pi * frequency.f / c / VF

        gamma = alpha + 1j*beta

        # return media object from z0 and gamma
        return DefinedGammaZ0(frequency=frequency, gamma=gamma,
                                    z0=z0, Z0=Z0)

    @classmethod
    def from_Z0_Dout(cls, frequency=None, z0=None,Z0=50,  epsilon_r=1,
                     Dout=5e-3, **kw):
        '''
        Init from characteristic impedance and outer diameter

        Parameters
        ----------
        frequency : :class:`~skrf.frequency.Frequency` object

        z0 : number, array-like, or None
            the port impedance for media. Only needed if  its different
            from the characterisitc impedance of the transmission
            line. if z0 is None then will default to Z0
        Z0 : number
            desired characteristic impedance
        Dout : number, or array-like
            outer conductor diameter, in m
        epsilon_r=1 : number, or array-like
            relative permittivity of the dielectric medium
        **kw :
            passed to __init__
        '''
        ep= epsilon_0*epsilon_r

        if imag(Z0) !=0:
            raise NotImplementedError()

        b = Dout/2.
        b_over_a = exp(2*pi*Z0*sqrt(ep/mu_0))
        a = b/b_over_a
        Dint = 2*a
        return cls(frequency=frequency, z0 = z0, Dint=Dint, Dout=Dout,
                    epsilon_r=epsilon_r, **kw)


    @property
    def Rs(self):
        '''
        Surface resistivity in Ohm/area

        Returns
        -------
        Rs : number or array
            surface resistivity

        '''
        f  = self.frequency.f
        rho = 1./self.sigma
        mu_r =1
        return surface_resistivity(f=f,rho=rho, mu_r=mu_r)

    @property
    def a(self):
        '''
        Inner radius of the coaxial line

        Returns
        -------
        a : float
            Inner radius

        '''
        return self.Dint/2.

    @property
    def b(self):
        '''
        Outer radius of the coaxial line
        
        Returns
        -------
        b : float
            Outer radius
        '''
        return self.Dout/2.


    # derivation of distributed circuit parameters
    @property
    def R(self):
        '''
        Distributed resistance R, in Ohm/m

        Returns
        -------
        R : number, or array-like
            distributed resistance, in Ohm/m

        '''
        return self.Rs/(2.*pi)*(1./self.a + 1./self.b)

    @property
    def L(self):
        '''
        Distributed inductance L, in H/m

        Returns
        -------
        L : number, or array-like
            distributed inductance, in  H/m

        '''
        return mu_0/(2.*pi)*log(self.b/self.a)

    @property
    def C(self):
        '''
        Distributed capacitance C, in F/m

        Returns
        -------
        C : number, or array-like
            distributed capacitance, in F/m

        '''
        return 2.*pi*self.epsilon_prime/log(self.b/self.a)

    @property
    def G(self):
        '''
        Distributed conductance G, in S/m

        Returns
        -------        
        G : number, or array-like
            distributed conductance, in S/m

        '''
        f =  self.frequency.f
        return f*self.epsilon_second/log(self.b/self.a)

    def __str__(self):
        f=self.frequency
        try:
            output =  \
                'Coaxial Transmission Line.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nDint= %.2f mm, Dout= %.2f mm '% \
                (self.Dint*1e3, self.Dout*1e3) +\
                '\nCharacteristic Impedance=(%.1f,%.1fj)-(%.1f,%.1fj) Ohm'% \
                (real(self.Z0[0]), imag(self.Z0[0]), real(self.Z0[-1]), imag(self.Z0[-1])) +\
                '\nPort impedance Z0=(%.1f,%.1fj)-(%.1f,%.1fj) Ohm'% \
                (real(self.z0[0]), imag(self.z0[0]), real(self.z0[-1]), imag(self.z0[-1]))
        except(TypeError):
            output =  \
                'Coaxial Transmission Line.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\nDint= %.2f mm, Dout= %.2f mm '% \
                (self.Dint[0]*1e3, self.Dout[0]*1e3) +\
                '\nCharacteristic Impedance=(%.1f,%.1fj)-(%.1f,%.1fj) Ohm'% \
                (real(self.Z0[0]), imag(self.Z0[0]), real(self.Z0[-1]), imag(self.Z0[-1])) +\
                '\nPort impedance Z0=(%.1f,%.1fj)-(%.1f,%.1fj) Ohm'% \
                (real(self.z0[0]), imag(self.z0[0]), real(self.z0[-1]), imag(self.z0[-1]))
        return output
