import numpy as np

from .Analytic_f_esc_Thin_Shell import Analytic_f_esc_Thin_Shell
from .Analytic_f_esc_Wind       import Analytic_f_esc_Wind

def  RT_f_esc_Analytic( Geometry , V_Arr , logNH_Arr , ta_Arr ):

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  ]
    
    assert Geometry in Geometry_Set , 'The geometry ' + Geometry + ' is nor supported in MODE=Analytic , only Thin_Shell and Galactic_Wind'
    
    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )
    
    if Geometry == 'Thin_Shell' :
        f_esc_Arr = Analytic_f_esc_Thin_Shell( V_Arr , logNH_Arr , ta_Arr )
    if Geometry == 'Galactic_Wind' :
        f_esc_Arr = Analytic_f_esc_Wind( V_Arr , logNH_Arr , ta_Arr )
    
    return f_esc_Arr

if __name__ == '__main__':
    pass


