''' bootstrap.py

Source: Mastering Python for Finance source code
'''
#import math
import numpy as np
class BootstrapYieldCurve(object):

    def __init__(self):
        self.zero_rates = dict()  # Map each T to a zero rate
        self.instruments = dict()  # Map each T to an instrument

    def add_instrument(self, par, T, coup, price,
                       compounding_freq=2):
        """  Save instrument info by maturity """
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_zero_rates(self):
        """  Calculate a list of available zero rates """
        self.__bootstrap_zero_coupons__()
        self.__get_bond_spot_rates__()
        return [self.zero_rates[T] for T in self.get_maturities()]

    def get_maturities(self):
        """ Return sorted maturities from added instruments. """
        return sorted(self.instruments.keys())

    def __bootstrap_zero_coupons__(self):
        """ Get zero rates from zero coupon bonds """
        for T in self.instruments.keys():
            (par, coup, price, freq) = self.instruments[T]
            if coup == 0:
                self.zero_rates[T] = \
                    self.zero_coupon_spot_rate(par, price, T)

    def __get_bond_spot_rates__(self):
        """ Get spot rates for every marurity available """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument

            if coup != 0:
                self.zero_rates[T] = \
                    self.__calculate_bond_spot_rate__(
                        T, instrument)

    def __calculate_bond_spot_rate__(self, T, instrument):
        """ Get spot rate of a bond by bootstrapping """
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq  # Number of coupon payments
            value = price
            per_coupon = coup / freq  # Coupon per period

            for i in range(int(periods)-1):
                t = (i+1)/float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * \
                                    np.exp(-spot_rate*t)
                value -= discounted_coupon

            # Derive spot rate for a particular maturity
            last_period = int(periods)/float(freq)
            spot_rate = -np.log(value /
                                  (par+per_coupon))/last_period
            return spot_rate

        except:
            print("Error: spot rate not found for T=%s" % t)

    def zero_coupon_spot_rate(self, par, price, T):
        """ Get zero rate of a zero coupon bond """
        spot_rate = np.log(par/price)/T
        return spot_rate
