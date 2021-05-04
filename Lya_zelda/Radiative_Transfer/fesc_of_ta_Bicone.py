import numpy as np

def fesc_of_ta_Bicone( ta , CCC , KKK , LLL ):

    f_esc = LLL * 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )

    return f_esc

if __name__ == '__main__':
    pass

