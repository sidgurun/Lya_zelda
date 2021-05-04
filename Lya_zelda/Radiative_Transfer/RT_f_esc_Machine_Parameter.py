import numpy as np

from .load_machine_fesc import load_machine_fesc

from .fesc_of_ta_Thin_and_Wind import fesc_of_ta_Thin_and_Wind

from .fesc_of_ta_Bicone import fesc_of_ta_Bicone

def  RT_f_esc_Machine_Parameter( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm='Tree' ):

        logNH_Arr         = np.atleast_1d( logNH_Arr )
        ta_Arr            = np.atleast_1d(    ta_Arr )
        V_Arr             = np.atleast_1d(     V_Arr )

        Coor_matrix = np.zeros( len(V_Arr) * 2 ).reshape( len(V_Arr) , 2 )

        Coor_matrix[ : , 0 ] = V_Arr
        Coor_matrix[ : , 1 ] = logNH_Arr

        if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

            CCC_machine = load_machine_fesc( Machine_Learning_Algorithm , 'CCC' , Geometry )
            KKK_machine = load_machine_fesc( Machine_Learning_Algorithm , 'KKK' , Geometry )

            CCC_model_Arr  = CCC_machine.predict( Coor_matrix )
            KKK_model_Arr  = KKK_machine.predict( Coor_matrix )

            f_esc_Arr = fesc_of_ta_Thin_and_Wind( ta_Arr , CCC_model_Arr , KKK_model_Arr )

        if Geometry in [ 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] :

            CCC_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'CCC' , Geometry )
            KKK_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'KKK' , Geometry )
            LLL_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'LLL' , Geometry )

            CCC_model_in_Arr  = CCC_machine_in.predict( Coor_matrix )
            KKK_model_in_Arr  = KKK_machine_in.predict( Coor_matrix )
            LLL_model_in_Arr  = LLL_machine_in.predict( Coor_matrix )

            f_esc_Arr = fesc_of_ta_Bicone( ta_Arr , CCC_model_in_Arr , KKK_model_in_Arr , LLL_model_in_Arr )

        return f_esc_Arr


if __name__ == '__main__':
    pass



