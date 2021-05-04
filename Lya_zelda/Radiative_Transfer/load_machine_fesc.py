import numpy as np

import os

import pickle

def load_machine_fesc( Machine , property_name , Geometry , Data_location ):

    '''
        This functions gives you the trained model that you want to use.
    '''

    Machine_Set = [ 'KN' , 'Grad' , 'Tree' , 'Forest'  ]

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]

    geo_code     = [ 'thin'        , 'wind'           , 'Bicone_X_Slab' , 'Bicone_X_Slab' ]

    Property_Set = [ 'KKK' , 'CCC' , 'LLL' , 'f_esc' ]

    assert property_name in Property_Set , "Houston we've got a problem, Error Code = 23452345.7523"

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    this_dir, this_filename = os.path.split(__file__)

    print( 'HARDCORING PATH TO GRIDS!!!!' )

    filename_root = 'finalized_model_'+ geo_code[index] +'_f_esc_' + Machine + '_' + property_name

    if Geometry == 'Bicone_X_Slab_In':
        filename_root += '_Inside_Bicone_' + str(True)

    if Geometry == 'Bicone_X_Slab_Out':
        filename_root += '_Inside_Bicone_' + str(False)

    filename = filename_root + '.sav'

    filename = os.path.join( Data_location , filename)

    loaded_model = pickle.load(open(filename, 'rb'))

    return  loaded_model

if __name__ == '__main__':
    pass

