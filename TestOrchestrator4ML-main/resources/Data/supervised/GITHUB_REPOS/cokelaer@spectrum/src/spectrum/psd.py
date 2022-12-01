"""This module provides the Base class for PSDs"""
import logging

import numpy

from spectrum.tools import nextpow2
from spectrum import errors
from spectrum.window import window_names
from spectrum import tools as stools


__all__ = ["Spectrum", "FourierSpectrum", "ParametricSpectrum", "Range"]


class Range(object):
    """A class to ease the creation of frequency ranges.

    Given the length :attr:`N` of a data sample and a sampling frequency
    :attr:`sampling`, this class provides methods to generate frequency
    ranges

        * :meth:`centerdc`: frequency range from -sampling/2 up to sampling/2 (excluded),
        * :meth:`twosided`: frequency range from 0 up to sampling (excluded),
        * :meth:`onesided`: frequency range from 0 up to sampling (included or
          excluded depending on evenness of the data). If NFFT is even, PSD 
          has length NFFT/2 + 1 over the interval [0,pi]. If NFFT is odd, 
          the length of PSD is (NFFT+1)/2 and the interval is [0, pi)


    Each method has a generator version:

    .. doctest::
        :options: +SKIP

        >>> r = Range(10, sampling=1)
        >>> list(r.onesided_gen())
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        >>> r.onesided()
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    The frequency range length is :math:`N/2+1` for the *onesided* case
    :math:`((N+1)/2` if :math:`N` is odd), and :math:`N` for the *twosided* and
    *centerdc* cases:

    .. doctest::
        :options: +SKIP

        >>> r.twosided()
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> len(r.twosided())
        10
        >>> len(r.centerdc())
        10
        >>> len(r.onesided())
        5
    """

    def __init__(self, N, sampling=1.):
        """.. rubric:: **Constructor**

        :param int N: the data length
        :param float sampling: sampling frequency of the input :attr:`data`.


        .. rubric:: Attributes:

        From the input parameters, read/write attributes are set:

            * :attr:`N`, the data length,
            * :attr:`sampling`, the sampling frequency.

        Additionally, the following read-only attribute is available:

            * :attr:`df`, the frequency step computed from :attr:`N` and
              :attr:`sampling`.

        """
        self.__N = N
        self.__sampling = sampling
        self.__df = None
        self._setN(N)
        self._setsampling(sampling)

    def _getdf(self):
        return self.__df
    df = property(fget=_getdf, doc="""Getter to access the frequency step,
        computed from :attr:`N` and :attr:`sampling`.""")

    def _getN(self):
        return self.__N
    def _setN(self,N):
        self.__N = N
        self.__df = self.__sampling/float(self.__N)
    N = property(fget=_getN, fset=_setN, doc="""Getter/Setter of the data length.
        If changed, :attr:`df` is updated.""")

    def _getsampling(self):
        return self.__sampling
    def _setsampling(self, sampling):
        self.__sampling = sampling
        self.__df = self.__sampling/float(self.__N)
    sampling = property(fget=_getsampling, fset=_setsampling,
        doc="""Getter/Setter of the sampling frequency. If changed, :attr:`df` is updated.""")

    def centerdc_gen(self):
        """Return the centered frequency range as a generator.

        ::

            >>> print(list(Range(8).centerdc_gen()))
            [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375]

        """
        for a in range(0, self.N):
            yield (a-self.N/2) * self.df

    def twosided_gen(self):
        """Returns the twosided frequency range as a generator

        ::

            >>> print(list(Range(8).centerdc_gen()))
            [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        """
        for a in range(0, self.N):
            yield a * self.df

    def onesided_gen(self):
        """Return the one-sided frequency range as a generator.

        If :attr:`N` is even, the length is N/2 + 1.
        If :attr:`N` is odd, the length is (N+1)/2.

        ::

            >>> print(list(Range(8).onesided()))
            [0.0, 0.125, 0.25, 0.375, 0.5]
            >>> print(list(Range(9).onesided()))
            [0.0, 0.1111, 0.2222, 0.3333, 0.4444]

        """
        if self.N % 2 == 0:
            for n in range(0, self.N//2 + 1):
                yield n * self.df
        else:
            for n in range(0, (self.N+1)//2):
                yield n * self.df

    def onesided(self):
        """Return the one-sided frequency range as a list (see
        :meth:`onesided_gen` for details).
        """
        return list(self.onesided_gen())

    def twosided(self):
        """Return the two-sided frequency range as a list (see
        :meth:`twosided_gen` for details)."""
        return list(self.twosided_gen())

    def centerdc(self):
        """Return the two-sided frequency range as a list (see
        :meth:`centerdc_gen` for details)."""
        return list(self.centerdc_gen())

    def __str__(self):
        msg = 'Range object information\n'
        msg += '    N=%s\n' % self.__N
        msg += '    sampling=%s\n' % self.__sampling
        msg += '    df=%s\n' % self.__df
        return msg


class Spectrum(object):
    """Base class for all Spectrum classes

    All PSD classes should inherits from this class to store common attributes
    such as the input data or sampling frequency. An instance is created
    as follows::

        >>> p = Spectrum(data, sampling=1024)
        >>> p.data
        >>> p.sampling

    The input parameters are:

    :param array data: input data (list or numpy.array)
    :param array data_y: input data required to perform cross-PSD only.
    :param float sampling: sampling frequency of the input :attr:`data`
    :param str detrend: detrend method ([None,'mean']) to apply on the input data before
        computing the PSD. See :attr:`detrend`.
    :param bool scale_by_freq: divide the final PSD by :math:`2*\pi/df`
    :param int NFFT: total length of the final data sets (padded with 
        zero if needed; default is 4096)

    The input parameters are available as attributes. Additional
    attributes such as :attr:`N` (the data length), :attr:`df` (the frequency
    step are set (see constructor documentation for a complete list).

    .. warning:: :class:`Spectrum` does not compute the PSD estimate.

    You can populate manually the :attr:`psd` attribute but you should
    respect the following convention:

        * if the input data is real, the PSD is assumed to be one-sided (odd
          length)
        * if the input data is complex, the PSD is assumed to be two-sided
          (even length).

    When :attr:`psd` is set, :attr:`sides` is reset to its default value,
    :attr:`NFFT` and :attr:`df` are updated.

    Spectrum instances have plotting utilities like :meth:`plot`
    that take care of plotting the PSD versus the appropriate frequency
    range (based on :attr:`sampling`, :attr:`NFFT` and :attr:`sides`)

    .. note:: the modification of some attributes (e.g., NFFT), makes the PSD
     obsolete. In such cases, the PSD must be re-computed before using :meth:`plot` again.

    At any time, you can get general information about the Spectrum instance::

        >>> p = Spectrum(marple_data)
        >>> print(p)
        Spectrum summary
            Data length is 64
            PSD not yet computed
            Sampling 1.0
            freq resolution 0.015625
            datatype is complex
            sides is twosided
            scal_by_freq is True

    """
    _detrend_choices = [None, 'mean']
    _sides_choices = ['onesided','twosided', 'centerdc', 'default']

    def __init__(self, data, data_y=None, sampling=1.,
                 detrend=None, scale_by_freq=True, NFFT=None):
        """**Constructor**

        .. rubric:: Attributes:

        From the input parameters, the following attributes are set:

          * :attr:`data` (updates :attr:`N`, :attr:`df`, :attr:`datatype`)
          * :attr:`data_y` used for cross PSD only (correlogram)
          * :attr:`detrend`
          * :attr:`sampling` (updates :attr:`df`)
          * :attr:`scale_by_freq`
          * :attr:`NFFT` (reset :attr:`sides`, :attr:`df`)

        The following read-only attributes are set during the initialisation:

          * :attr:`datatype`
          * :attr:`df`
          * :attr:`N`

        And finally, additional read-write attributes are available:

          * :attr:`psd`: used to store the PSD data array, which size depends
            on :attr:`sides` i.e., one-sided for real data and two-sided for
            the complex data.
          * :attr:`sides`: if set, changed the :attr:`psd`.

        """
        # user attributes. No need to bother about initialisation here
        self.__data = None
        self.__data_y = None
        self.__sampling = None
        self.__detrend = None
        self.__scale_by_freq = None
        # other attributes
        self.__sides = None
        self.__N = None
        self.__NFFT = None
        self.__df = None
        self.__datatype = None
        self.__psd = None
        self.__method = None

        # Real initialise is made here
        self.data = data
        if data_y is not None:
            self.data_y = data_y
        self.sampling = sampling
        self.sides = 'default'
        # self._range.N is the NFFT value. By default, it is the size of the data

        self._range = Range(self.__data.size, sampling) # can be private.
        self.modified = True
        self.sampling = sampling
        self.scale_by_freq = scale_by_freq

        self.NFFT = NFFT
        self.method = self.__class__  # alias to the class name

    def _getMethod(self):
        return self.__method
    def _setMethod(self, method):
        self.__method = method
    method = property(fget=_getMethod, fset=_setMethod)

    """def __call__(self, *args, **kargs):
        # To be use with care. THis function is there just to help, it
        #    does not populate the proper attribute except psd.
        if self.method is not None:
            res = self.method(self.data, *args, **kargs)
            self.psd = res[0]
        #return res
    """
    def run(self, *args, **kargs):
        # Kept for back compatibility
        # e.g. used in this external notebook:
        # http://nbviewer.jupyter.org/gist/juhasch/5182528
        self()

    def _getDetrend(self):
        return self.__detrend
    def _setDetrend(self, detrend):
        if detrend == self.__detrend:
            return
        if detrend not in self._detrend_choices:
            raise errors.SpectrumChoiceError(detrend, self._detrend_choices)
        self.__detrend = detrend
        self.modified = True
    detrend = property(fget=_getDetrend, fset=_setDetrend,
                       doc="""Getter/Setter to detrend:

                    * None: do not perform any detrend.
                    * 'mean': remove the mean value of each segment from each segment of the data.
                    * 'long-mean': remove the mean value from the data before splitting it into segments.
                    * 'linear': remove linear trend from each segment.""")

    def _get_range(self):
        return self._range
    range = property(fget=_get_range,
            doc="""Read only attribute to a :class:`Range` object.""")

    def _getScale(self):
        return self.__scale_by_freq
    def _setScale(self, scale):
        if scale == self.__scale_by_freq: 
            return
        assert scale in [True, False]
        self.__scale_by_freq = scale
        self.modified = True
    scale_by_freq = property(fget=_getScale, fset=_setScale,
                            doc="scale the PSD by :math:`2*\pi/df`")

    def _getNFFT(self):
        return self.__NFFT
    def _setNFFT(self, NFFT):#if NFFT is changed, we need to redo the padding
        if self.__NFFT == NFFT and self.__NFFT is not None:
            logging.debug('NFFT is the same, nothing to do')
            return
        new_nfft = None
        if NFFT == 'nextpow2':
            n = nextpow2(self.data.size)
            new_nfft = int(pow(2, n))
            logging.debug('NFFT is based on nextpow2: %s' % new_nfft)
        elif NFFT is None:
            logging.debug('NFFT set to data length')
            new_nfft = self.N
        elif isinstance(NFFT, int):
            logging.debug('NNFT set  manually to {}'.format(NFFT))
            assert NFFT > 0, 'NFFT must be a positive integer'
            new_nfft = NFFT
        else:
            raise ValueError("NFFT must be either None, positive integer or 'nextpow2'")

        #print new_nfft
        if self.__NFFT != new_nfft:
            self.__NFFT = new_nfft
            # Now that the NFFT has changed, we need to update the range
            self._range.N = self.__NFFT
            self.__sides = self._default_sides()
            self.modified = True
    NFFT = property(fget=_getNFFT, fset=_setNFFT, doc=
                    """Getter/Setter to the NFFT attribute.

                    :param NFFT: a user choice for setting :attr:`NFFT`.

                        * if None, the NFFT is set to :attr:`N`, the data length.
                        * if 'nextpow2', computes the next power of 2 greater
                          than or equal to the data length.
                        * if a integer is provided, it must be positive

                    If NFFT is changed, :attr:`sides` is reset and :attr:`df` as well.
                    """)

    def _default_sides(self):
        if self.datatype == 'real':
            sides = 'onesided'
        else:
            sides = 'twosided'
        return sides

    def _getSides(self):
        return self.__sides
    def _setSides(self, sides):
        # check validity of sides
        if sides not in self._sides_choices:
            raise errors.SpectrumChoiceError(sides, self._sides_choices)

        # default value
        if sides == 'default':
            sides = self._default_sides()

        # check validity of sides
        if self.datatype == 'complex':
            assert sides != ['onesided'], "complex data cannot be onesided (%s provided)" % sides

        # If sides is indeed different, update the psd
        if self.__psd is not None:
            newpsd = self.get_converted_psd(sides)
            self.__psd = newpsd
        self.__sides = sides
        logging.debug('------------> %s %s' % (self.__sides, sides))
        # we set the PSD by hand, so we can consider that PSD is up-to-date
        self.modified = False
    _doc_sides = """Getter/Setter to the :attr:`sides` attributes.

    It can be 'onesided', 'twosided', 'centerdc'. This setter changes
    :attr:`psd` to reflect the user argument choice.

    If the datatype is complex, sides cannot be one-sided.
    """
    sides = property(fget=_getSides, fset=_setSides, doc=_doc_sides)

    def _get_data_y(self):
        return self.__data_y
    def _set_data_y(self, data):
        self.__data_y = data
        self.modified = True
    data_y = property(fget=_get_data_y, fset=_set_data_y,
                      doc="""Getter/Setter to the Y-data""")

    def _getData(self):
        return self.__data
    def _setData(self, data):
        if type(data) == list:
            from numpy import array
            self.__data = array(data)
        else:
            self.__data = data.copy()
        self.__N = self.data.size # N has no setter, so we use the private version
        self.modified = True

        if numpy.isrealobj(self.__data):
            self.__datatype = 'real'
        else:
            self.__datatype = 'complex'
    data = property(fget=_getData, fset=_setData, doc="""Getter/Setter for the
    input data. If input is a list, it is cast into a numpy.array. Then, :attr:`N`,
    :attr:`df` and  :attr:`datatype` are updated.""")

    def _getPSD(self):
        if self.__psd is None or self.modified is True:
            logging.debug('Computing PSD.')
            self()
            self.modified = False
        return self.__psd
    def _setPSD(self, psd):
        # Reset the sides attribute depending on the datatype
        # if a user sets the PSD manually, there is an ambiguity because
        # input data may be odd or even length for the one-sided case
        # A twosided PSD must be even though.
        # note dec 2018: onesided can be even. e.g. if data length is N=14,
        # then psd length is 8 (N/2+1). We should introduce a more robust way
        # of setting PSD or forbid it.
        if self.datatype == 'real':
            # if psd is set we assume that the original data was even
            # so that length original data is 2 + 2*(len(psd)-1)
            #assert len(psd) % 2 == 1, 'odd data, so PSD must be one-sided'
            self.__sides = 'onesided'
            # we cannot know the NFFT used by the user when creating the 
            # PSD because depends on even or odd data. Same issue with range
            #self.__NFFT = 2 + (len(psd)-2) * 2
            self.__psd = numpy.array(psd)
            #self._range.N = self.__NFFT
        else:
            #assert len(psd) % 2 == 0, 'even  data, so PSD should be in two-sided format'
            self.__sides = 'twosided'
            self.__NFFT = len(psd)
            self.__psd = numpy.array(psd)
            self._range.N = self.__NFFT
        # we set the psd manually, so :attr:modified is reset
        self.modified = False
    psd = property(fget=_getPSD, fset=_setPSD, doc="""Getter/Setter to :attr:`psd`

    :param array psd: the array must be in agreement with the onesided/twosided
        convention: if the data in real, the psd must be onesided. If the
        data is complex, the psd must be twosided.

    When you set this attribute, several attributes are set:

        * :attr:`sides` is set to onesided if datatype is real and twosided
          if datatype is complex.
        * :attr:`NFFT` is set to len(psd)*2 if sides=onesided and (len(psd)-1)*2
          if sides=twosided.
        * :attr:`range`.N is set to NFFT, which update :attr:`df`.

    """)

    # simple getter/setter for the N attribute. Updates df
    def _getN(self):
        return self.__N
    N = property(fget=_getN,
                 doc="""Getter to the original data size. :attr:`N` is automatically
                 updated when changing the data only.""")

    #simple getter/setter for the sampling attribute. Updates sampling and df
    def _getSampling(self):
        return self.__sampling
    def _setSampling(self, sampling):
        if sampling == self.__sampling: return
        self.__sampling = sampling
        self.__df = self.__sampling / float(self.__N)
        self.modified = True
    sampling = property(fget=_getSampling, fset=_setSampling,
        doc="""Getter/Setter to sampling frequency. Updates the :attr:`df` automatically.""")

    #the frequency resolution. Read only.
    def _getdf(self):
        return self._range.df
    df = property(fget=_getdf, doc="""Getter to step frequency. This attribute
 is updated as soon as :attr:`data` or :attr:`sampling` is changed""")

    def _getdatatype(self):
        return self.__datatype
    datatype = property(fget=_getdatatype,
                    doc="""Getter to the datatype ('real' or 'complex').
                    :attr:`datatype` is automatically updated when changing
                    the data.""")

    def scale(self):
        if self.scale_by_freq is True:
            self.psd *= 2*numpy.pi/self.df

    def frequencies(self, sides=None):

        """Return the frequency vector according to :attr:`sides`"""
        # use the attribute sides except if a valid sides argument is provided
        if sides is None:
            sides = self.sides
        if sides not in self._sides_choices:
            raise errors.SpectrumChoiceError(sides, self._sides_choices)

        if sides == 'onesided':
            return self._range.onesided()
        if sides == 'twosided':
            return self._range.twosided()
        if sides == 'centerdc':
            return self._range.centerdc()

    def get_converted_psd(self, sides):
        """This function returns the PSD in the **sides** format

        :param str sides: the PSD format in ['onesided', 'twosided', 'centerdc']
        :return: the expected PSD.

        .. doctest::

            from spectrum import *
            p = pcovar(marple_data, 15)
            centerdc_psd = p.get_converted_psd('centerdc')

        .. note:: this function does not change the object, in particular, it
            does not change the :attr:`psd` attribute. If you want to change
            the psd on the fly, change the attribute :attr:`sides`.

        """
        if sides == self.sides:
            #nothing to be done is sides = :attr:`sides
            return self.__psd

        if self.datatype == 'complex':
            assert sides != 'onesided', \
                "complex datatype so sides cannot be onesided."

        if self.sides == 'onesided':
            logging.debug('Current sides is onesided')
            if sides == 'twosided':
                logging.debug('--->Converting to twosided')
                # here we divide everything by 2 to get the twosided version
                #N = self.NFFT
                newpsd = numpy.concatenate((self.psd[0:-1]/2., list(reversed(self.psd[0:-1]/2.))))
                # so we need to multiply by 2 the 0 and FS/2 frequencies
                newpsd[-1] = self.psd[-1]
                newpsd[0] *= 2.
            elif sides == 'centerdc':
                # FIXME. this assumes data is even so PSD is stored as
                # P0 X1 X2 X3 P1
                logging.debug('--->Converting to centerdc')
                P0 = self.psd[0]
                P1 = self.psd[-1]
                newpsd = numpy.concatenate((self.psd[-1:0:-1]/2., self.psd[0:-1]/2.))
                # so we need to multiply by 2 the 0 and F2/2 frequencies
                #newpsd[-1] = P0 / 2
                newpsd[0] = P1
        elif self.sides == 'twosided':
            logging.debug('Current sides is twosided')
            if sides == 'onesided':
                # we assume that data is stored as X0,X1,X2,X3,XN
                # that is original data is even.
                logging.debug('Converting to onesided assuming ori data is even')
                midN = (len(self.psd)-2) / 2
                newpsd = numpy.array(self.psd[0:int(midN)+2]*2)
                newpsd[0] /= 2
                newpsd[-1] = self.psd[-1]
            elif sides == 'centerdc':
                newpsd = stools.twosided_2_centerdc(self.psd)
        elif self.sides == 'centerdc': # same as twosided to onesided
            logging.debug('Current sides is centerdc')
            if sides == 'onesided':
                logging.debug('--->Converting to onesided')
                midN = int(len(self.psd) / 2)
                P1 = self.psd[0]
                newpsd = numpy.append(self.psd[midN:]*2, P1)
            elif sides == 'twosided':
                newpsd = stools.centerdc_2_twosided(self.psd)
        else:
            raise ValueError("sides must be set to 'onesided', 'twosided' or 'centerdc'")

        return newpsd

    def plot(self, filename=None, norm=False, ylim=None,
              sides=None,  **kargs):
        """a simple plotting routine to plot the PSD versus frequency.

        :param str filename: save the figure into a file
        :param norm: False by default. If True, the PSD is normalised.
        :param ylim: readjust the y range .
        :param sides: if not provided, :attr:`sides` is used. See :attr:`sides`
            for details.
        :param kargs: any optional argument accepted by :func:`pylab.plot`.

        .. plot::
            :width: 80%
            :include-source:

            from spectrum import *
            p = Periodogram(marple_data)
            p.plot(norm=True, marker='o')

        """
        import pylab
        from pylab import ylim as plt_ylim
        #First, check that psd attribute is up-to-date
        # just to get the PSD to be recomputed if needed
        _ = self.psd


        # check that the input sides parameter is correct if provided
        if sides is not None:
            if sides not in self._sides_choices:
                raise errors.SpectrumChoiceError(sides, self._sides_choices)

        # if sides is provided but identical to the current psd, nothing to do.
        # if sides not provided, let us use self.sides
        if sides is None or sides == self.sides:
            frequencies = self.frequencies()
            psd = self.psd
            sides = self.sides
        elif sides is not None:
            # if sides argument is different from the attribute, we need to
            # create a new PSD/Freq ; indeed we do not want to change the
            # attribute itself

            # if data is complex, one-sided is wrong in any case.
            if self.datatype == 'complex':
                if sides == 'onesided':
                    raise ValueError("sides cannot be one-sided with complex data")

            logging.debug("sides is different from the one provided. Converting PSD")
            frequencies = self.frequencies(sides=sides)
            psd = self.get_converted_psd(sides)

        if len(psd) != len(frequencies):
            raise ValueError("PSD length is %s and freq length is %s" % (len(psd), len(frequencies)))

        if 'ax' in list(kargs.keys()):
            save_ax = pylab.gca()
            pylab.sca(kargs['ax'])
            rollback = True
            del kargs['ax']
        else:
            rollback = False

        if norm:
            pylab.plot(frequencies, 10 * stools.log10(psd/max(psd)),  **kargs)
        else:
            pylab.plot(frequencies, 10 * stools.log10(psd),**kargs)

        pylab.xlabel('Frequency')
        pylab.ylabel('Power (dB)')
        pylab.grid(True)

        if ylim:
            plt_ylim(ylim)

        if sides == 'onesided':
            pylab.xlim(0, self.sampling/2.)
        elif sides == 'twosided':
            pylab.xlim(0, self.sampling)
        elif sides == 'centerdc':
            pylab.xlim(-self.sampling/2., self.sampling/2.)

        if filename:
            pylab.savefig(filename)

        if rollback:
            pylab.sca(save_ax)

        del psd, frequencies #is it needed?

    def power(self):
        r"""Return the power contained in the PSD

        if scale_by_freq is False, the power is:

        .. math:: P = N \sum_{k=1}^{N} P_{xx}(k)

        else, it is

        .. math:: P =  \sum_{k=1}^{N} P_{xx}(k) \frac{df}{2\pi}

        .. todo:: check these equations


        """
        if self.scale_by_freq == False:
            return sum(self.psd) * len(self.psd)
        else:
            return sum(self.psd) * self.df/(2.*numpy.pi)

    def _str_title(self):
        return "Spectrum summary\n"

    def __str__(self):
        msg = self._str_title()
        msg += "    Data length is %s\n" % self.data.size
        try:
            msg += "    PSD length is %s\n" % self.psd.size
        except:
            msg += "    PSD not yet computed\n"
        msg += "    Sampling %s\n" % self.sampling
        msg += "    freq resolution %s\n" % self.df
        msg += "    datatype is %s\n" % self.datatype
        msg += "    sides is %s\n" % self.sides
        msg += "    scal_by_freq is %s\n" % self.scale_by_freq
        return msg


class ParametricSpectrum(Spectrum):
    """Spectrum based on Fourier transform.

    This class inherits attributes and methods from
    :class:`Spectrum`. It is used by children class :class:`~spectrum.periodogram.Periodogram`,
    :class:`~spectrum.correlog.pcorrelogram` and :class:`Welch` PSD estimates.
    The parameters are those used by :class:`Spectrum`.

    :param array data:     Input data (list or numpy.array)
    :param float sampling: sampling frequency of the input :attr:`data`
    :param str detrend:    detrend method ([None,'mean']) to apply on the input
        data before computing the PSD. See :attr:`detrend`.
    :param bool scale_by_freq: Divide the final PSD by :math:`2*\pi/df`

    In addition you need specific parameters such as:

    :param str window:  a tapering window. See :class:`~spectrum.window.Window`.
    :param int lag:     to be used by the :class:`~spectrum.correlog.pcorrelogram` methods only.
    :param int NFFT:    Total length of the data given to the FFT

    This class has dedicated PSDs methods such as :meth:`periodogram`, which
    are equivalent to children class such as :class:`~spectrum.periodogram.Periodogram`.

    .. plot::
        :width: 80%
        :include-source:

        from spectrum import datasets
        from spectrum import ParametricSpectrum
        data = datasets.data_cosine(N=1024)
        s = ParametricSpectrum(data, ar_order=4, ma_order=4, sampling=1024, NFFT=512, lag=10)
        #s.parma()
        #s.plot(sides='onesided')
        #s.plot(sides='twosided')

    """
    def __init__(self, data, sampling=1., ar_order=None, ma_order=None,
            lag=-1, NFFT=None, detrend=None, scale_by_freq=True):
        """**Constructor**

        See the class documentation for the parameters.

        .. rubric:: Additional attributes to those inherited
            from :class:`Spectrum`:

        * :attr:`ar_order`, the ar order of the PSD estimates
        * :attr:`ma_order`, the ar order of the PSD estimates

        """
        super(ParametricSpectrum, self).__init__(data, sampling=sampling,
                                                 NFFT=NFFT,
                                                 scale_by_freq=scale_by_freq,
                                                 detrend=detrend)
        if ar_order is None and ma_order is None:
            raise errors.SpectrumARMAError
        #new user attributes
        self.__ar_order = ar_order
        self.__ma_order = ma_order
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.lag = lag
        # will be populated when running an ARMA PSD estimate
        self.__ar = None
        self.__ma = None
        self.__reflection = None
        self.__rho = None

    def _set_ar_order(self, ar):
        if ar is not None:
            if ar < 0:
                raise errors.SpectrumARError
            self.__ar_order = ar
    def _get_ar_order(self):
        return self.__ar_order
    ar_order = property(fget=_get_ar_order, fset=_set_ar_order, doc="")

    def _set_ma_order(self, ma):
        if ma is not None:
            if ma < 0:
                raise errors.SpectrumMAError
            self.__ma_order = ma
        else:
            self.__ma_order = None
    def _get_ma_order(self):
        return self.__ma_order
    ma_order = property(fget=_get_ma_order, fset=_set_ma_order, doc="")

    def _set_ma(self, ma):
        self.__ma = ma
    def _get_ma(self):
        return self.__ma
    ma = property(fget=_get_ma, fset=_set_ma, doc="")

    def _set_ar(self, ar):
        self.__ar = ar
    def _get_ar(self):
        return self.__ar
    ar = property(fget=_get_ar, fset=_set_ar, doc="")

    def _set_rho(self, rho):
        self.__rho = rho
    def _get_rho(self):
        return self.__rho
    rho = property(fget=_get_rho, fset=_set_rho)

    def _set_ref(self, ref):
        self.__reflection = ref
    def _get_ref(self):
        return self.__reflection
    reflection = property(fget=_get_ref, fset=_set_ref)

    """ self.reflection = None
        elif method == 'aryule':
            from spectrum import aryule
            ar, v, coeff = aryule(self.data, self.ar_order)
            self.ar = ar
            self.rho = v
            self.reflection = coeff
        """

    def plot_reflection(self):
        from pylab import stem, title, xlabel, ylabel
        if self.reflection is not None:
            stem(list(range(0, len(self.reflection))), abs(self.reflection))
            title('Reflection coefficient evolution')
            xlabel('Order')
            ylabel('Reflection Coefficient absolute values')
        else:
            logging.warning("Reflection coefficients not available or not yet computed.")

    def _str_title(self):
        return "ParametricSpectrum summary\n"

    def __str__(self):
        return super(ParametricSpectrum, self).__str__()



class FourierSpectrum(Spectrum):
    """Spectrum based on Fourier transform.

    This class inherits attributes and methods from  :class:`Spectrum`. It is
    used by children class :class:`~spectrum.periodogram.Periodogram`,
    :class:`~spectrum.correlog.pcorrelogram` and :class:`Welch` PSD estimates.

    The parameters are those used by :class:`Spectrum`

    :param array data:     Input data (list or numpy.array)
    :param data_y: input data required to perform cross-PSD only.
    :param float sampling: sampling frequency of the input :attr:`data`
    :param str detrend:    detrend method ([None,'mean']) to apply on the input
        data before computing the PSD. See :attr:`detrend`.
    :param bool scale_by_freq: Divide the final PSD by :math:`2*\pi/df`
    :param int NFFT: total length of the data given to the FFT

    In addition you need specific parameters such as:

    :param str window:  a tapering window. See :class:`~spectrum.window.Window`.
    :param int lag:     to be used by the :class:`~spectrum.correlog.pcorrelogram` methods only.


    This class has dedicated PSDs methods such as :meth:`speriodogram`, which
    are equivalent to children class such as :class:`~spectrum.periodogram.Periodogram`.

    .. plot::
        :width: 80%
        :include-source:

        from spectrum import datasets
        from spectrum import FourierSpectrum
        s = FourierSpectrum(datasets.data_cosine(), lag=32, sampling=1024, NFFT=512)
        s.periodogram()
        s.plot(label='periodogram')
        #s.correlogram()
        s.plot(label='correlogram')
        from pylab import legend, xlim
        legend()
        xlim(0,512)

    """
    _window = window_names

    def __init__(self, data, sampling=1.,
                 window='hanning', NFFT=None, detrend=None,
                 scale_by_freq=True, lag=-1):
        """**Constructor**

        See the class documentation for the parameters.

        .. rubric:: Additional attributes to those inherited from

        :class:`Spectrum` are:

        * :attr:`lag`, a lag used to compute the autocorrelation
        * :attr:`window`, the tapering window to be used

        """
        super(FourierSpectrum, self).__init__(data,
                sampling=sampling, detrend=detrend,
                scale_by_freq=scale_by_freq, NFFT=NFFT)
        self.__window = None
        self.__lag = None
        #self.P0 = None
        self.lag = lag
        self.window = window

    def _set_window(self, window):
        if window == self.__window:
            return
        if window not in self._window:
            raise errors.SpectrumChoiceError(window, self._window)
        self.__window = window
        self.modified = True
    def _get_window(self):
        return self.__window
    window = property(fget=_get_window, fset=_set_window,
                      doc="""Tapering window to be applied""")

    def _set_lag(self, lag):
        if lag == self.__lag:
            return
        self.__lag = lag
        self.modified = True
    def _get_lag(self):
        return self.__lag
    lag = property(fget=_get_lag, fset=_set_lag, doc="""Getter/Setter used by the correlogram when
        autocorrelation estimates are required.""")

    def _str_title(self):
        return "FourierSpectrum summary\n"

    def periodogram(self):
        """An alias to :class:`~spectrum.periodogram.Periodogram`

        The parameters are extracted from the attributes. Relevant attributes
        ares :attr:`window`, attr:`sampling`, attr:`NFFT`, attr:`scale_by_freq`,
        :attr:`detrend`.

        .. plot::
            :width: 80%
            :include-source:

            from spectrum import datasets
            from spectrum import FourierSpectrum
            s = FourierSpectrum(datasets.data_cosine(), sampling=1024, NFFT=512)
            s.periodogram()
            s.plot()
        """
        from .periodogram import speriodogram
        psd = speriodogram(self.data, window=self.window, sampling=self.sampling,
                             NFFT=self.NFFT, scale_by_freq=self.scale_by_freq,
                             detrend=self.detrend)
        self.psd = psd


