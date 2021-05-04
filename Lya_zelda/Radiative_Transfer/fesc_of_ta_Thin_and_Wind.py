import numpy as np

def fesc_of_ta_Thin_and_Wind( ta , CCC , KKK ):

    f_esc = 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )

    return f_esc

if __name__ == '__main__' :
    pass
    
