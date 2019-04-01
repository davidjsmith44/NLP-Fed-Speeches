''' Time Value of Money Classes '''


from math import pow, floor, ceil, log
from quant.optimization import newton

class TVM:
    bgn, end = 0, 1
    def __str__(self):
        return "n=%f, r=%f, pv=%f, pmt=%f, fv=%f" % (
            self.n, self.r, self.pv, self.pmt, self.fv)
    def __init__(self, n=0.0, r=0.0, pv=0.0, pmt=0.0, fv=0.0, mode=end):
        self.n = float(n)
        self.r = float(r)
        self.pv = float(pv)
        self.pmt = float(pmt)
        self.fv = float(fv)
        self.mode = mode
    def calc_pv(self):
        z = pow(1+self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        return -(self.fv*z + (1-z)*pva)
    def calc_fv(self):
        z = pow(1+self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        return -(self.pv + (1-z) * pva)/z
    def calc_pmt(self):
        z = pow(1+self.r, -self.n)
        if self.mode==TVM.bgn:
            return (self.pv + self.fv*z) * self.r / (z-1) / (1+self.r)
        else:
            return (self.pv + self.fv*z) * self.r / (z-1)
    def calc_n(self):
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        z = (-pva-self.pv) / (self.fv-pva)
        return -log(z) / log(1+self.r)
    def calc_r(self):
        def function_fv(r, self):
            z = pow(1+r, -self.n)
            pva = self.pmt / r
            if (self.mode==TVM.bgn): pva += self.pmt
            return -(self.pv + (1-z) * pva)/z
        return newton(f=function_fv, fArg=self, x0=.05,
            y=self.fv, maxIter=1000, minError=0.0001)

# Newton-Raphson Method:
from math import pow, floor, ceil, log
from quant.optimization import newton
class TVM:
    bgn, end = 0, 1
    def __str__(self):
        return "n=%f, r=%f, pv=%f, pmt=%f, fv=%f" % (
            self.n, self.r, self.pv, self.pmt, self.fv)
    def __init__(self, n=0.0, r=0.0, pv=0.0, pmt=0.0, fv=0.0, mode=end):
        self.n = float(n)
        self.r = float(r)
        self.pv = float(pv)
        self.pmt = float(pmt)
        self.fv = float(fv)
        self.mode = mode
    def calc_pv(self):
        z = pow(1+self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        return -(self.fv*z + (1-z)*pva)
    def calc_fv(self):
        z = pow(1+self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        return -(self.pv + (1-z) * pva)/z
    def calc_pmt(self):
        z = pow(1+self.r, -self.n)
        if self.mode==TVM.bgn:
            return (self.pv + self.fv*z) * self.r / (z-1) / (1+self.r)
        else:
            return (self.pv + self.fv*z) * self.r / (z-1)
    def calc_n(self):
        pva = self.pmt / self.r
        if (self.mode==TVM.bgn): pva += self.pmt
        z = (-pva-self.pv) / (self.fv-pva)
        return -log(z) / log(1+self.r)
    def calc_r(self):
        def function_fv(r, self):
            z = pow(1+r, -self.n)
            pva = self.pmt / r
            if (self.mode==TVM.bgn): pva += self.pmt
            return -(self.pv + (1-z) * pva)/z
        return newton(f=function_fv, fArg=self, x0=.05,
            y=self.fv, maxIter=1000, minError=0.0001)

## example #1 Mortgage payments

from quant.tvm import TVM
pmt = TVM(n=25*12, r=.04/12, pv=500000, fv=0).calc_pmt()
print("Payment = %f" % pmt)

## Example #2 Yield to Maturity
r = 2*TVM(n=10*2, pmt=6/2, pv=-80, fv=100).calc_r()
print("Interest Rate = %f" % r)

### Example 3: Arbitrage-free Bond Pricing
pv = TVM(r=.06, n=8, pmt=5, fv=100).calc_pv()
print("Present Value = %f" % pv)

## Example of boostrapping
'''
epic, description,          coupon, maturity,  bid,    ask
TR13, Uk Gilt Treasury Stk, 4.5,    07-Mar-13, 101.92, 102.07
T813, Uk Gilt Treasury Stk, 8,      27-Sep-13, 107.86, 107.98
TR14, Uk Gilt Treasury Stk, 2.25,   07-Mar-14, 102.90, 103.05
'''

tr = [] # list of raw (not interpolated) times to maturity
yr = [] # list of raw (not interpolated) yields
for b in bonds:
    ttm = (b.maturity - localtime).days / 360
    price = (b.bid+b.ask)/2
    ytm = TVM(n=ttm*b.freq, pv=-price, pmt=b.couponRate/b.freq, fv=1).calc_r() * b.freq
    tr.append(ttm)
    yr.append(ytm)

# the TTM is time to matuiruty and YTM is yield to maturity


''' Interpolation - creates ytm for different bond terms '''
t = list(i for i in range(1,41)) # interpolating in range 1..40 years
y = []
interp = scipy.interpolate.interp1d(tr, yr, bounds_error=False, fill_value=scipy.nan)
for i in t:
    value = float(interp(i))
    if not scipy.isnan(value): # Don't include out-of-range values
        y.append(value)

''' Boostrapping of spot rates '''
s = [] # output array for spot rates
for i in range(0, len(t)): #calculate i-th spot rate
    sum = 0
    for j in range(0, i): #by iterating through 0..i
        sum += y[i] / (1 + s[j])**t[j]
    value = ((1+y[i]) / (1-sum))**(1/t[i]) - 1
    s.append(value)



