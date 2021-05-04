import numpy as np

from .load_machine_fesc import load_machine_fesc

def  RT_f_esc_Machine_Values( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm='Tree' ):

        logNH_Arr         = np.atleast_1d( logNH_Arr )
        ta_Arr            = np.atleast_1d(    ta_Arr )
        V_Arr             = np.atleast_1d(     V_Arr )

        Coor_matrix = np.zeros( len(V_Arr) * 3 ).reshape( len(V_Arr) , 3 )

        Coor_matrix[ : , 0 ] = V_Arr
        Coor_matrix[ : , 1 ] = logNH_Arr
        Coor_matrix[ : , 2 ] = np.log10(ta_Arr)

        loaded_model = load_machine_fesc( Machine_Learning_Algorithm , 'f_esc' , Geometry )

        f_esc_Arr = loaded_model.predict( Coor_matrix )

        return f_esc_Arr

if __name__ == '__main__':
    pass

