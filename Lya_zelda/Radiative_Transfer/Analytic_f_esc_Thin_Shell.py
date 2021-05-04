import numpy as np


def Analytic_f_esc_Thin_Shell( V_Arr , logNH_Arr , ta_Arr ):

    NH18 = 10 ** ( logNH_Arr - 18 )

    #New MCMC
    c11 = 10**(1.90526)
    c12 = -10**(2.0399)
    c13 = 10**(2.34829)
    c21 = 10**(-3.138837)
    c22 = -10**(-1.92151)
    c23 = 10**(-1.1860205000000001)
    c24 = -10**(-0.1480042)
    c3 = 10**(0.0530715)
    c4 = 10**(-2.743455)

    C1 = ( ( np.log10(NH18) ) ** 2 ) * c11 + np.log10(NH18) * c12 + c13
    y  = np.log10(NH18)
    C2 = c21*y*y*y + c22*y*y + c23*y + c24
    C3 = c3
    C4 = c4

    K1 = C1 * ( V_Arr ** C2 )
    K2 = C3 * ( V_Arr ** C4 )

    fesc =  1. / np.cosh( np.sqrt( K1 * ( ta_Arr ** K2 ) ) )

    return fesc

if __name__ == '__init__':
    pass


