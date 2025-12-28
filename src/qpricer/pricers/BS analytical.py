import math          # good for one-off scalar math (simpler, tiny overhead)
import numpy as np   # use for vectors/matrices: faster via vectorization + broadcasting
# Using NumPy lets this class accept arrays (S, K, sigma, t) and compute in one go
# when building an IV surface—no Python loops needed.
from scipy.stats import norm

# what the return type will look like -> If you give it one number, it should give you one float back.
# If you give it a list/array of numbers, it should give you a NumPy array of floats back.

class BS:
    def __init__(self,spot,strike,ttm,rate,dividend,vol):    
        #base variables
        self.S = spot 
        self.K = strike
        self.t = ttm        # time to maturity 
        self.r = rate       # continuos domestic rate 
        self.q = dividend   # contintuous foreign int rate or dividend yield
        self.vol = vol 

        #common terms used freqeuntly 
        self.sqrt_t = np.sqrt(self.t)
        self.df_r = np.exp(-self.r * self.t)
        self.df_q = np.exp(-self.q * self.t )

        #d1 and d2 parameters
        self.d1, self.d2 = self.calculate_d1d2()

        #common terms used freqeuntly
        norm_d1 = norm.cdf(self.d1)
        norm_d2 = norm.cdf(self.d2)
        norm_neg_d1 = norm.cdf(-self.d1)
        norm_neg_d2 = norm.cdf(-self.d2)

        #option prices 
        self.call_price = self.C_price()
        self.put_price = self.P_price()

        #commonly used terms for greeks computation 
        self.norm_d1_pdf = norm.pdf(self.d1)    #N(d1) is cdf of d1, its derivative is just pdf of d1

        #greeks 
        self.call_delta = norm_d1
        self.put_delta = -norm_neg_d1
        self.gamma = self.calculate_gamma()
        self.vega = self.calculate_vega()
        self.call_theta = self.calculate_c_theta()
        self.put_theta = self.calculate_p_theta()
        self.call_rho = self.calculate_c_rho()
        self.put_rho = self.calculate_p_rho()

    #d1 and d2 computation
    def calculate_d1d2(self): 
        temp = np.log( self.S/self.t ) + ( self.r - self.q + ( self.vol ** 2 / 2 ) ) * self.t
        d1 = temp * ( 1 / self.sqrt_t )
        d2 = d1 - ( self.vol * self.sqrt_t)
        return d1,d2
    
    #option price computation
    def C_price(self): 
        temp1 = self.S * self.norm_d1
        temp2 = self.K * self.df * self.norm_d2
        c_price = temp1 - temp2
        return c_price
    
    def P_price(self): 
        temp1 = self.K * self.norm_neg_d2 * self.df_r
        temp2 = self.S * self.norm_neg_d1
        p_price = temp1 - temp2 
        return p_price
    
    #greeks computation
    def calculate_gamma(self): 
        return ( self.norm_d1_pdf )/ (self.S * self.vol * self.sqrt_t )
    
    def calculate_vega(self): 
        return ( self.S * self.norm_d1_pdf * self.sqrt_t)
    
    def calculate_call_theta(self): 
        temp1 =  ( -self.S * self.norm_d1_pdf * self.vol ) / (2 * self.sqrt_t) 
        temp2 =  ( -self.rate * self.K * self.df_r * self.norm_d2 )
        call_theta = temp1 + temp2 
        return call_theta

    def calculate_put_theta(self): 
        temp1 = ( -self.S * self.norm_d1_pdf * self.vol ) / (2 * self.sqrt_t) 
        temp2 = ( self.rate * self.K * self.df_r * self.norm_neg_d2 )
        put_theta = temp1 + temp2
        return put_theta

    def calculate_call_rho(self): 
        return self.K * self.t * self.df_r * self.norm_d2
    
    def calculate_put_rho(self): 
        return (-self.K * self.t * self.df_r * self.norm_neg_d2)