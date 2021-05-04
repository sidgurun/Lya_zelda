import numpy as np

def Analytic_f_esc_Wind( V_Arr , logNH_Arr , ta_Arr ):

    NH18 = 10 ** ( logNH_Arr - 18 )

    # New MCMC
    c11 = 10**(0.4852541)
    c12 = 10**(-0.2006394)
    c21 = 10**(-1.912059)
    c22 = -10**(-0.6380347)
    c3 = 10**(0.046314074999999996)
    c4 = 10**(-1.782037)

    C1  = c11 * ( NH18 ** c12 )
    C2  = c21 * np.log10( NH18 )**2 + c22 * np.log10(NH18) #+ c23
    C3  = c3
    C4  = c4

    K1  = C1 * V_Arr ** C2
    K2  = C3 * V_Arr ** C4

    fesc = 1./ np.cosh( np.sqrt( K1 * ta_Arr ** K2 ) )

    return fesc

if __name__ == '__main__':
    pass


