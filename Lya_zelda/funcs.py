"""
zELDA es a phantastic code!!
"""
import os


import time

#from pylab import *

import sys
import shutil

#import urllib

import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import pickle

from scipy.optimize import curve_fit

from scipy.ndimage import gaussian_filter1d

import emcee

from sklearn.neural_network import MLPRegressor

from pyswarms.single.global_best import GlobalBestPSO
#====================================================================#
#====================================================================#
#====================================================================#
Data_location = '/global/users/sidgurung/PROMARE/Grids/'
#====================================================================#
#====================================================================#
#====================================================================#
def Check_if_DATA_files_are_found():
    '''
        This function checks if all the data files are in the set directory.

        **Input**:

        None : None

        **Output**:

        Bool_1 : Bool
                    1 if     found.
                    0 if not found.
    '''
    this_dir, this_filename = os.path.split(__file__)

    Bool_1 = True
     
    arxiv_with_file_names = this_dir + '/DATA/List_of_DATA_files'

    with open( arxiv_with_file_names ) as fd:

        for line in fd.readlines():

            arxiv_name = line.strip('\n')

            Bool_1 = Bool_1 * os.path.isfile( Data_location + arxiv_name )

    return Bool_1
#====================================================================#
#====================================================================#
#====================================================================#
def load_machine_fesc( Machine , property_name , Geometry ):
    '''
        This functions gives you the trained model that you want to use.

        **Input**:

        Machine : String
                    Kind of algorithm: 'KN', 'Grad', 'Tree' or 'Forest'
        
        property_name : String
                    The variable to import: 'KKK' , 'CCC' , 'LLL' or 'f_esc' 

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'

        **Output**:

        loaded_model : file with all the necesary to do machine learning
    '''

    Machine_Set = [ 'KN' , 'Grad' , 'Tree' , 'Forest'  ]

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]

    geo_code     = [ 'thin'        , 'wind'           , 'Bicone_X_Slab' , 'Bicone_X_Slab' ]

    Property_Set = [ 'KKK' , 'CCC' , 'LLL' , 'f_esc' ]

    assert property_name in Property_Set , "Houston we've got a problem, Error Code = 23452345.7523"

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    this_dir, this_filename = os.path.split(__file__)

    filename_root = 'finalized_model_'+ geo_code[index] +'_f_esc_' + Machine + '_' + property_name 

    if Geometry == 'Bicone_X_Slab_In':
        filename_root += '_Inside_Bicone_' + str(True)

    if Geometry == 'Bicone_X_Slab_Out':
        filename_root += '_Inside_Bicone_' + str(False)

    filename = filename_root + '.sav'

    filename = os.path.join( Data_location , filename)

    loaded_model = pickle.load(open(filename, 'rb'))

    return  loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
def Analytic_f_esc_Thin_Shell( V_Arr , logNH_Arr , ta_Arr ):

    '''
        Return the escape fraction computed analytically for the Thin Shell
        (Gurung-lopez et al. 2019a)

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Arr : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        **Output**:

        fesc :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

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
#====================================================================#
#====================================================================#
#====================================================================#
def Analytic_f_esc_Wind( V_Arr , logNH_Arr , ta_Arr ):
    '''
        Return the escape fraction computed analytically for the Galactic Wind
        (Gurung-lopez et al. 2019a)

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        **Output**:

        fesc :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

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
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Analytic( Geometry , V_Arr , logNH_Arr , ta_Arr ):
    '''
        Return the escape fraction computed analytically
        (Gurung-lopez et al. 2019a, 2019b)

        **Input**:
        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Arr : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        **Output**:

        fesc :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

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
#====================================================================#
#====================================================================#
#====================================================================#
def fesc_of_ta_Thin_and_Wind( ta , CCC , KKK ):
    '''
        Anallytic expression of the escape fraction as a function of the
        dust optical depth (Gurung-lopez et al. 2019a)

        **Input**:

        ta : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        CCC : float
                    CCC parameter

        KKK : float
                    KKK parameter

        **Output**:

        fesc :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

    f_esc = 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )

    return f_esc
#====================================================================#
#====================================================================#
#====================================================================#
def fesc_of_ta_Bicone( ta , CCC , KKK , LLL ):
    '''
        Anallytic expression of the escape fraction as a function of the
        dust optical depth (Gurung-lopez et al. 2019b)

        **Input**:

        ta : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        CCC : float
                    CCC parameter

        KKK : float
                    KKK parameter

        LLL : float
                    LLL parameter

        **Output**:

        fesc :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''
 
    f_esc = LLL * 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )
 
    return f_esc
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Machine_Parameter( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm='Tree' ):

    '''
        Anallytic expression of the escape fraction as a function of the
        dust optical depth (Gurung-lopez et al. 2019b). This uses the parametric
        expression of the escape fraction.
    
        **Input**:
    
        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'
    
        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 
    
        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]
    
        Machine_Learning_Algorithm : string
                    Machine learning algorithm: 'KN', 'Grad', 'Tree' or 'Forest'
    
        **Output**:
    
        f_esc_Arr :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''
    
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
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Machine_Values( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm='Tree' ):
    '''
        Anallytic expression of the escape fraction as a function of the
        dust optical depth (Gurung-lopez et al. 2019b). This uses the directly
        the escape fraction.

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        Machine_Learning_Algorithm : string
                    Machine learning algorithm: 'KN', 'Grad', 'Tree' or 'Forest'

        **Output**:

        f_esc_Arr :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

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
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Linear_ND_interpolator( N_dim , Coor_props_Matrix , Coor_grid_list , Field_in_grid_Matrix ):

    '''
        Interpolates in an arbitrary dimension space        

        Parameters
        ----------
        N_dim : int
                Number of dimensions.

        Coor_props_Matrix : List of N_dim float values
                Coordenates in the N_dim space to evaluate.
                For example [ X , Y , Z ]
               

        Coor_grid_list : List of N_dim 1-D sequence of floats
                For example, if there is a field evaluated in X_Arr, Y_Arr, Z_Arr
                [ X_Arr , Y_Arr , Z_Arr ]

        Field_in_grid_Matrix : numpy array with the field to interpolate

        Returns
        -------
        Field_at_the_prob_point :
    '''

    dic_index = {}

    for i in range( 0 , N_dim ):

        dic_index['index_' + str(i) ] = np.where( ( Coor_grid_list[i] < Coor_props_Matrix[i] ) )[0][-1]

    dic_prob = {}

    for i in range( 0 , N_dim ):

        INDEX = dic_index['index_' + str(i) ]

        #print( 'INDEX' , INDEX)

        diff_i = Coor_grid_list[i][ INDEX+1 ] - Coor_grid_list[i][ INDEX ]

        min_i = Coor_grid_list[i][ INDEX ]

        dic_prob['prob_' + str(i) ] = ( Coor_props_Matrix[i] - min_i ) * 1. / diff_i

        #print( 'prob' , dic_prob['prob_' + str(i) ] )

    N_points = 2 ** N_dim

    VOLs   = {}
    FIELDs = {}

    for i in range( 0 , N_points ):

        binary_str = '{0:b}'.format( i ).zfill( N_dim )

        VOLs[ 'vol_' + str( i ) ] = 1.

        eval_INDEX = []

        for j in range( 0 , N_dim ):

            CTE = int( binary_str[j] )

            eval_INDEX.append( dic_index['index_' + str(j) ] + CTE )

            if CTE == 0 : size_j = ( 1. - dic_prob[ 'prob_' + str(j) ] )
            if CTE == 1 : size_j = (      dic_prob[ 'prob_' + str(j) ] )

            pre_vol = VOLs[ 'vol_' + str( i ) ]

            VOLs[ 'vol_' + str( i ) ] = pre_vol * size_j

        eval_INDEX = tuple( eval_INDEX )

        FIELDs['field_' + str(i) ] = Field_in_grid_Matrix[ eval_INDEX ]


    Zero_Zero = np.array( eval_INDEX ) * 0

    Zero_Zero = tuple( Zero_Zero.tolist() )

    Field_at_the_prob_point = np.zeros_like( Field_in_grid_Matrix[Zero_Zero] )

    for i in range( 0 , N_points ) :

        #print( VOLs[ 'vol_' + str( i ) ] , FIELDs['field_' + str(i) ] )

        Field_at_the_prob_point += VOLs[ 'vol_' + str( i ) ] * FIELDs['field_' + str(i) ]

    return Field_at_the_prob_point
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def load_Grid_fesc( Geometry , MODE ):
    '''
        This functions gives you grids of the escape fraction

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'
        MODE : String
                    Parametrization of the escape fraction. 'Parameters' or 'values'
        **Output**:

        loaded_model : file the grid of f_esc parameters/values.
    '''

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out']

    geo_code     = [ 'Thin_Shell'  , 'Wind'           , 'Bicone_X_Slab' , 'Bicone_X_Slab' ]

    MODE_Set = [ 'Parameters' , 'values' ]

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    filename_root = 'Dictonary_'+ geo_code[index] +'_Grid_f_esc_' + MODE 

    if Geometry == 'Bicone_X_Slab_In':
        filename_root += '_Inside_Bicone_' + str(True)

    if Geometry == 'Bicone_X_Slab_Out':
        filename_root += '_Inside_Bicone_' + str(False)

    filename = filename_root + '.npy'

    filename = os.path.join( Data_location , filename)

    loaded_model = np.load( filename , allow_pickle=True , encoding='latin1' ).item()

    return loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_f_esc_Arrays_2D_grid( V_Arr , logNH_Arr , ta_Arr , Grid_Dictionary , Geometry ):
    '''
        Computes the escape fraction using the escape fraction grids of parameters

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        Grid_Dictionary : python dictionary
                    Constains the grid to compute the escape fraction. Loaded with
                    load_Grid_fesc().

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'
                    
        **Output**:

        f_esc_Arr :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

    V_Arr_Grid     = Grid_Dictionary[     'V_Arr' ]

    logNH_Arr_Grid = Grid_Dictionary[ 'logNH_Arr' ]

    logta_Arr_Grid = Grid_Dictionary[ 'logta_Arr' ]

    Grid           = Grid_Dictionary[    'Grid'   ]

    N_objects = len( V_Arr )

    CCC_Arr_evaluated = np.zeros( N_objects )
    KKK_Arr_evaluated = np.zeros( N_objects )

    ###################

    Coor_grid_list = [ V_Arr_Grid , logNH_Arr_Grid ]

    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

        for INDEX in range( 0 , N_objects ):

            Coor_props_Matrix = [ V_Arr[INDEX] , logNH_Arr[INDEX] ]

            #CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] = Linear_2D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , Grid )
            CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] = Linear_ND_interpolator( 2 , Coor_props_Matrix , Coor_grid_list , Grid )

        f_esc_Arr = fesc_of_ta_Thin_and_Wind( ta_Arr , CCC_Arr_evaluated , KKK_Arr_evaluated )

    ###################

    if Geometry in [ 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] :
    
        LLL_Arr_evaluated = np.zeros( N_objects )

        for INDEX in range( 0 , N_objects ):

            Coor_props_Matrix = [ V_Arr[INDEX] , logNH_Arr[INDEX] ]

            #CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] , LLL_Arr_evaluated[ INDEX ] = Linear_2D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , Grid )
            CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] , LLL_Arr_evaluated[ INDEX ] = Linear_ND_interpolator( 2 , Coor_props_Matrix , Coor_grid_list , Grid )

        f_esc_Arr = fesc_of_ta_Bicone( ta_Arr , CCC_Arr_evaluated , KKK_Arr_evaluated , LLL_Arr_evaluated )


    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_fesc_Arrays_3D_grid( V_Arr , logNH_Arr , ta_Arr , Grid_Dictionary ):
    '''
        Computes the escape fraction using the escape fraction grids of parameters

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        Grid_Dictionary : python dictionary
                    Constains the grid to compute the escape fraction. Loaded with
                    load_Grid_fesc().

        **Output**:

        f_esc_Arr_evaluated :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

    V_Arr_Grid     = Grid_Dictionary[     'V_Arr' ]

    logNH_Arr_Grid = Grid_Dictionary[ 'logNH_Arr' ]

    logta_Arr_Grid = Grid_Dictionary[ 'logta_Arr' ]

    Grid           = Grid_Dictionary[    'Grid'   ]

    logta_Arr = np.log10( ta_Arr )

    N_objects = len( V_Arr )

    f_esc_Arr_evaluated = np.zeros( N_objects )

    Coor_Arr_list = [ V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid ]

    for INDEX in range( 0 , N_objects ):

        #f_esc_Arr_evaluated[ INDEX ] = Linear_3D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , logta_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid , Grid )
    
        Coor_list = [ V_Arr[INDEX] , logNH_Arr[INDEX] , logta_Arr[INDEX] ]

        f_esc_Arr_evaluated[ INDEX ] = Linear_ND_interpolator( 3 , Coor_list , Coor_Arr_list , Grid )

    return f_esc_Arr_evaluated
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Interpolation_Values( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm=None ):
    '''
        Computes the escape fraction using the escape fraction grids of values

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        Machine_Learning_Algorithm : String
                    Kind of algorithm: 'KN', 'Grad', 'Tree' or 'Forest'

        **Output**:

        f_esc_Arr :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )

    DATA_DICTIONAY = load_Grid_fesc( Geometry , 'values' )

    f_esc_Arr = Interpolate_fesc_Arrays_3D_grid( V_Arr , logNH_Arr , ta_Arr , DATA_DICTIONAY )

    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Interpolation_Parameters( Geometry , V_Arr , logNH_Arr , ta_Arr , Machine_Learning_Algorithm=None ):
    '''
        Computes the escape fraction using the escape fraction grids of parameters

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        Machine_Learning_Algorithm : String
                    Kind of algorithm: 'KN', 'Grad', 'Tree' or 'Forest'

        **Output**:

        f_esc_Arr :  1-D sequence of floats
                    Escape fractions for the input configurations [no dimensions] 
    '''

    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )

    DATA_DICTIONAY = load_Grid_fesc( Geometry , 'Parameters' )

    f_esc_Arr = Interpolate_f_esc_Arrays_2D_grid( V_Arr , logNH_Arr , ta_Arr , DATA_DICTIONAY , Geometry )

    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE ):
    '''
        Checks the inflow/outflow parameters before doing the proper computation.

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        MODE : optional string
               Set the mode in which the escape fraction is computed. It can be:
                    Analytic        : it uses an analytic equation fitted to the output of the RT MC code.
                    Parametrization : it computes the escape fraction using a function that depends on the 
                                      dust optical depts as in Neufeld et al. 1990.
                    Raw             : it uses directly the output of the RT MC code.

                    Default = 'Parametrization'

                    Kind of algorithm: 'KN', 'Grad', 'Tree' or 'Forest'

        **Output**:

        mask_good : 1-D sequence of bool
                    1 if the parameters are good, 0 if they are bad.  
    '''

    V_Arr     = np.atleast_1d(     V_Arr )
    logNH_Arr = np.atleast_1d( logNH_Arr )
    ta_Arr    = np.atleast_1d(    ta_Arr )

    V_Arr     =     V_Arr.astype(float)
    logNH_Arr = logNH_Arr.astype(float)
    ta_Arr    =    ta_Arr.astype(float)

    bool1 = np.isfinite( V_Arr     )
    bool2 = np.isfinite( logNH_Arr )
    bool3 = np.isfinite( ta_Arr    )

    mask_good = bool1 * bool2 * bool3
    
    assert sum( mask_good ) != 0 , 'All the V-logNH-ta combinations are np.nan, -np.inf or np.inf'

    #============================================#
    if Geometry in [ 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] :
        tmp_bool1 = V_Arr < 100.0
        tmp_bool2 = logNH_Arr >= 20.5

        tmp_mask = tmp_bool1 * tmp_bool2   

        mask_good = mask_good * ~tmp_mask
 
    #============================================#

    bool5 = V_Arr >= 10.00 

    bool6 = V_Arr <= 1000
    
    bool7 = logNH_Arr >= 17.0

    bool8 = logNH_Arr <= 22.0

    mask_good = mask_good * ( bool5 * bool6 ) * ( bool7 * bool8 )

    if MODE=='Raw':
        bool9 = ta_Arr >= 10**(-2.5)

        bool10 = ta_Arr <= 10**(-0.25)

        mask_good = mask_good * ( bool9 * bool10 )

    #return V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good
    return mask_good
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE='Parametrization' , Algorithm='Intrepolation' , Machine_Learning_Algorithm='Tree' ):

    '''
        Return the Lyman alpha escape fraction for a given outflow properties.

        **Input**

        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab'.

        V_Arr : 1-D sequence of float
                Array with the expansion velocity of the outflow. The unit
                are km/s. 

        logNH_Arr : 1-D sequence of float
                    Array with the logarithim of the outflow neutral hydrogen
                    column density. The units of the colum density are in c.g.s,
                    i.e, cm**-2.

        ta_Arr : 1-D sequence of float
                 Array with the dust optic depth of the outflow. 

        MODE : optional string
               Set the mode in which the escape fraction is computed. It can be:
                    Analytic        : it uses an analytic equation fitted to the output of the RT MC code.
                    Parametrization : it computes the escape fraction using a function that depends on the 
                                      dust optical depts as in Neufeld et al. 1990.
                    Raw             : it uses directly the output of the RT MC code.

                Default = 'Parametrization'


        Algorithm : optional string
                Set how the escape fraction is computed. If MODE='Analytic' then this varialbe is useless.
                    Intrepolation    : Direct lineal interpolation.
                    Machine_Learning : uses machine learning algorithms
        
                Default = 'Intrepolation'


        Machine_Learning_Algorithm : optial string
                Set the machine learning algorith used. Available:
                    Tree   : decision tree
                    Forest : random forest
                    KN     : KN

                Default = 'Tree'


        **Output**

        lines_Arr : 1-D sequence of float
                    The Lyman alpha escape fraction for V_Arr[i] ,
                    logNH_Arr[i] , ta_Arr[i] , Inside_Bicone_Arr[i].
    '''

    assert MODE in [ 'Parametrization' , 'Raw' , 'Analytic'] , 'The requested mode ' + MODE + ' is not available. The modes supported are : Parametrization , Raw , Analytic' 

    assert Algorithm in [ 'Intrepolation' , 'Machine_Learning' ] , 'The requested algorithm ' + Algorithm + ' is not available. The algorithms supported are : Intrepolation , Machine_Learning' 

    assert Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] , 'The requested geoemtry ' + Geometry + ' is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'
    
    mask_good = pre_treatment_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE ) 

    #print( 'fraction good confs = ' , np.sum(mask_good) * 1. / len(mask_good)  )

    f_esc_Arr = np.zeros( len( mask_good ) ) * np.nan

    if MODE == 'Parametrization'  :

        if Algorithm == 'Intrepolation' :
            funtion_to_use = RT_f_esc_Interpolation_Parameters 

        if Algorithm == 'Machine_Learning':
            funtion_to_use = RT_f_esc_Machine_Parameter

    if MODE == 'Raw'  :

        if Algorithm == 'Intrepolation' :
            funtion_to_use = RT_f_esc_Interpolation_Values

        if Algorithm == 'Machine_Learning':
            funtion_to_use = RT_f_esc_Machine_Values

    if MODE == 'Analytic' :

        funtion_to_use = RT_f_esc_Analytic        

    f_esc_Arr[ mask_good ] = funtion_to_use( Geometry , V_Arr[ mask_good ] , logNH_Arr[ mask_good ] , ta_Arr[ mask_good ] , Machine_Learning_Algorithm=Machine_Learning_Algorithm )

    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def define_RT_parameters( T4=None ):
    '''
        This function gives the parameters use to compute useful 
        variables when working with radiative transfer.

        **Input**:

        T4 : optional float
            Temperature in units of 10**4 K:
            T4 = T[k] / 10**4

        **Output**:

        nu0 : float
            Lyaman-alpha frequency.

        Dv : float
            
    '''

    if T4 is None :  
       T4 = 1. # = 10000. / 1e4

    nu0 = 2.46777 * 1.e15 #3. * 10.**8 / (1215.67 * (10**(-10)))
    Vth = 12.85 * np.sqrt(T4) # lo he comentado porque sqrt(1) = 1
    Dv = Vth * nu0 *1. / ( 3 * (10**5))
    return nu0 , Dv
#====================================================================#
#====================================================================#
#====================================================================#
def convert_x_into_lamda( x , T4=None ):
    '''
        This function converts from frequency in Doppler
        units to wavelength

        **Input**:

        x : 1-D sequence of float
            Frequency in Doppler units.

        T4 : optional float
            Temperature in units of 10**4 K:
            T4 = T[k] / 10**4

        **Output**:

        w_Arr : 1-D sequence of float
                wavelength.        
    '''

    if T4 is None :  
        T4 = 1. # = 10000. / 1e4

    nu0 = 2.46777 * 1.e15 #3. * 10.**8 / (1215.67 * (10**(-10)))
    Vth = 12.85 * np.sqrt(T4) # lo he comentado porque sqrt(1) = 1
    Dv = Vth * nu0 *1. / ( 3 * (10**5))
    nu0 , Dv = define_RT_parameters( T4 ) 
    w_Arr = 3. * 1.e8 / ( x * Dv + nu0)
    return w_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def convert_lamda_into_x( lamda , T4=None ):
    '''
        This function converts from frequency in Doppler
        units to wavelength

        **Input**:

        lamda : 1-D sequence of float
            wavelength

        T4 : optional float
            Temperature in units of 10**4 K:
            T4 = T[k] / 10**4

        **Output**:

        x_Arr : 1-D sequence of float
                Frequency in Doppler units.
    '''
    nu0 , Dv = define_RT_parameters( T4 ) 
    x_Arr = (( 3. * 1.e8 / lamda) -nu0 ) / Dv
    return x_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def load_Grid_Line( Geometry , MODE='FULL' ):

    '''
        Return the dictionary with all the properties of the grid where the lines were run.

        **Input**

        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab_In', 'Bicone_X_Slab_Out',
                   'Thin_Shell_Cont'.

        MODE : optinal string.
               For the 'Thin_Shell_Cont' ONLY. Defines the grid to be loaded.
               MODE='FULL'  loads a very dense  grid. ~12GB of RAM.
               MODE='LIGHT' loads a more sparse grid. ~ 2GB of RAM.

        **Output**

        loaded_model : Dictionary
                       This dictonary have all the information of the grid.
                       Entries:
                            'V_Arr'     : Array of velocity expansions used.[km/s]
                            'logNH_Arr' : Array of logarithm of the column density. [c.g.s.]
                            'logta_Arr' : Array of logarithm of the dust optical depth.
                            'x_Arr'     : Array of frequency in Doppler  units.
                            'Grid'      : Array with the output of the RT MC code LyaRT:
                                         
                                loaded_model['Grid'][i,j,k,:] has the line profile evaluated in loaded_model['x_Arr']
                                with outflow velocity loaded_model['V_Arr'][i] , logarithm of the neutral hydrogen 
                                column density loaded_model['logNH_Arr'][j] and logarithm of dust optical depth 
                                loaded_model['logta_Arr'][k]  
    '''

    assert Geometry in [ 'Thin_Shell_Cont' , 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] , 'The requested geoemtry ' + Geometry + '  is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]

    geo_code     = [ 'Thin_Shell'  , 'Wind'           , 'Bicone_X_Slab' , 'Bicone_X_Slab' ]

    #this_dir, this_filename = os.path.split(__file__)

    #print( 'HARDCORING PATH TO GRIDS!!!!' )
    #this_dir = '/global/users/sidgurung/PROMARE/Grids/'
    #this_dir = '/global/users/sidgurung/PROMARE/Grids/'

    if not Geometry in [ 'Thin_Shell_Cont' , 'Thin_Shell_Cont_Light' ] :

        index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

        filename_root = 'Dictonary_'+ geo_code[index] +'_Grid_Lines'

        if Geometry == 'Bicone_X_Slab_In':
            filename_root += '_In_Bicone_' + str(True)
        if Geometry == 'Bicone_X_Slab_Out':
            filename_root += '_In_Bicone_' + str(False)

        filename = filename_root + '.npy'

        filename = os.path.join( Data_location , filename)

        loaded_model = np.load( filename , allow_pickle=True , encoding='latin1' ).item()

    if Geometry in [ 'Thin_Shell_Cont' , 'Thin_Shell_Cont_Light' ] :

        NV  = 29
        NNH = 19
        Nta = 9

        NEW = 20
        NWi = 31

        if MODE == 'FULL':
            NEW = 20
            NWi = 31
        
        if MODE == 'LIGHT':
            NEW = 8
            NWi = 9
        
        t_name = '_V_'+str(NV)+'_logNH_'+str(NNH)+'_logta_'+str(Nta)+'_EW_'+str(NEW)+'_Wi_'+str(NWi)+'.npy'

        loaded_model = {}
    
        tmp_1 = np.load( Data_location + '/GRID_info_' + t_name , allow_pickle=True ).item()
        GRID  = np.load( Data_location + '/GRID_data_' + t_name  )

        loaded_model['Grid']      = GRID
        loaded_model['V_Arr']     = tmp_1['V']
        loaded_model['logNH_Arr'] = tmp_1['logNH']
        loaded_model['logta_Arr'] = tmp_1['logta']
        loaded_model['logEW_Arr'] = tmp_1['logEW']
        loaded_model['Wi_Arr']    = tmp_1['Wi']

        loaded_model['x_Arr'] = convert_lamda_into_x( tmp_1['wavelength']*1e-10 )
        loaded_model['w_Arr'] = tmp_1['wavelength']

    return loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_3D_grid( V_Arr , logNH_Arr , logta_Arr , x_Arr , Grid_Dictionary ):
    '''
        Computes the escape fraction using the line profiles grids for many
        configurations

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    Logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        x_Arr : 1-D sequence of floats
                    Frequency in Doppler units where the line prfile will be
                    evaluated.

        Grid_Dictionary : python dictionary
                    All the necessary information for the interpoation. Loaded 
                    with load_Grid_Line().

        **Output**:

        line_Arr :  2-D sequence of floats
                    Flux density in arbitrary units. The first dimension matches
                    the dimension of the input configurations (e.g. V_Arr). The
                    second dimension matches x_Arr.
    '''
    lines_Arr = np.zeros( len(V_Arr) * len( x_Arr ) ).reshape( len(V_Arr) , len( x_Arr ) )

    for i in range( 0 , len( V_Arr ) ):

        lines_Arr[i] = Interpolate_Lines_Arrays_3D_grid_MCMC( V_Arr[i] , logNH_Arr[i] , logta_Arr[i] , x_Arr , Grid_Dictionary )

    return lines_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_3D_grid_MCMC( V_Value , logNH_Value , logta_Value , x_Arr , Grid_Dictionary ):
    '''
        Computes the escape fraction using the line profiles grids for one
        configuration. This is usefull for the Thin_Shell, Galactic_Wind and
        Bicones configurations.

        **Input**:

        V_Value : float
                    Outflow bulk velocity [km/s] 
        
        logNH_Value : float
                    Logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Value : float
                    Dust optical depth [no dimensions]

        x_Arr : 1-D sequence of floats
                    Frequency in Doppler units where the line prfile will be
                    evaluated.

        Grid_Dictionary : python dictionary
                    All the necessary information for the interpoation. Loaded 
                    with load_Grid_Line().

        **Output**:

        axu_line_1 : 1-D sequence of floats
                    Flux density in arbitrary units. 
    '''

    Grid_Line = Grid_Dictionary['Grid']

    V_Arr_Grid = Grid_Dictionary['V_Arr']
    x_Arr_Grid = Grid_Dictionary['x_Arr']

    logNH_Arr_Grid = Grid_Dictionary['logNH_Arr']
    logta_Arr_Grid = Grid_Dictionary['logta_Arr']

    Coor_Arr_list = [ V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid ]
     
    Coor_list = [ V_Value , logNH_Value , logta_Value ]

    aux_line = Linear_ND_interpolator( 3 , Coor_list , Coor_Arr_list , Grid_Line )

    #print( aux_line[0] , right=[-1] )

    axu_line_1 = np.interp( x_Arr , x_Arr_Grid , aux_line , left=aux_line[0] , right=aux_line[-1] )

    Integral = np.trapz( axu_line_1 , x_Arr )

    axu_line_1 = np.absolute( axu_line_1 * 1. / Integral )

    return axu_line_1
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_5D_grid( V_Arr , logNH_Arr , logta_Arr , logEW_Arr , Wi_Arr , x_Arr , Grid_Dictionary ):
    '''
        Computes the escape fraction using the line profiles grids for many
        configurations. This is usefull for the Thin_Shell_Cont

        **Input**:

        V_Arr : 1-D sequence of floats
                    Outflow bulk velocity [km/s] 
        
        logNH_Ar : 1-D sequence of floats
                    Logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Arr : 1-D sequence of floats
                    Dust optical depth [no dimensions]

        logEW_Arr : 1-D sequence of floats
                    Logarithm of the rest frame equivalent width [A]

        Wi_Arr : 1-D sequence of floats
                    Rest frame instrinc line width [A]

        x_Arr : 1-D sequence of floats
                    Frequency in Doppler units where the line prfile will be
                    evaluated.

        Grid_Dictionary : python dictionary
                    All the necessary information for the interpoation. Loaded 
                    with load_Grid_Line().

        **Output**:

        linew_Arr : 2-D sequence of floats
                    Flux density in arbitrary units. The first dimension matches
                    the dimension of the input configurations (e.g. V_Arr). The
                    second dimension matches x_Arr.
                    Flux density in arbitrary units. 
    '''

    lines_Arr = np.zeros( len(V_Arr) * len( x_Arr ) ).reshape( len(V_Arr) , len( x_Arr ) )

    for i in range( 0 , len( V_Arr ) ):

        lines_Arr[i] = Interpolate_Lines_Arrays_5D_grid_MCMC( V_Arr[i] , logNH_Arr[i] , logta_Arr[i] , logEW_Arr[i] , Wi_Arr[i] , x_Arr , Grid_Dictionary )

    return lines_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_5D_grid_MCMC( V_Value , logNH_Value , logta_Value , logEW_Value , Wi_Value , x_Arr , Grid_Dictionary ):
    '''
        Computes the escape fraction using the line profiles grids for many
        configurations. This is usefull for the Thin_Shell_Cont

        **Input**:

        V_Value : float
                    Outflow bulk velocity [km/s] 
        
        logNH_Value : float
                    Logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Value : float
                    Dust optical depth [no dimensions]

        logEW_Value : float
                    Logarithm of the rest frame equivalent width [A]

        Wi_Value : float
                    Rest frame instrinc line width [A]

        x_Arr : 1-D sequence of floats
                    Frequency in Doppler units where the line prfile will be
                    evaluated.

        Grid_Dictionary : python dictionary
                    All the necessary information for the interpoation. Loaded 
                    with load_Grid_Line().

        **Output**:

        axu_line_1 : 1-D sequence of floats
                    Flux density in arbitrary units. 
    '''

    Grid_Line = Grid_Dictionary['Grid']

    V_Arr_Grid = Grid_Dictionary['V_Arr']
    x_Arr_Grid = Grid_Dictionary['x_Arr']

    Wi_Arr_Grid = Grid_Dictionary['Wi_Arr']

    logNH_Arr_Grid = Grid_Dictionary['logNH_Arr']
    logta_Arr_Grid = Grid_Dictionary['logta_Arr']
    logEW_Arr_Grid = Grid_Dictionary['logEW_Arr']

    Coor_Arr_list = [ V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid , logEW_Arr_Grid , Wi_Arr_Grid ]

    Coor_list = [ V_Value , logNH_Value , logta_Value , logEW_Value , Wi_Value ]

    aux_line = Linear_ND_interpolator( 5 , Coor_list , Coor_Arr_list , Grid_Line )

    aux_line = aux_line[::-1]

    x_Arr_Grid = x_Arr_Grid[::-1]

    axu_line_1 = np.interp( x_Arr , x_Arr_Grid , aux_line , left=aux_line[0] , right=aux_line[-1] )

    Integral = np.trapz( axu_line_1 , x_Arr )

    axu_line_1 = np.absolute( axu_line_1 * 1. / Integral )

    return axu_line_1
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_Line_profile_MCMC( Geometry , V_Value , logNH_Value , ta_Value , logEW_Value=None , Wi_Value=None ):
    '''
        Checks the inflow/outflow parameters before doing the proper computation.

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out'
                                         'Thin_Shell_Cont'

        V_Value : float
                    Outflow bulk velocity [km/s] 
        
        logNH_Value : float
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Value : float
                    Dust optical depth [no dimensions]

        logEW_Value : Optinal float
                    Logarithm of rest frame equiavlent width [A]
                    Default = None

        Wi_Value : Optinal float
                    Intrinsic width line in the rest frame [A]
                    Default = None

        **Output**:

        Bool_good : 1-D sequence of bool
                    1 if the parameters are good, 0 if they are bad.  
    '''

    bool1 = np.isfinite(     V_Value )
    bool2 = np.isfinite( logNH_Value )
    bool3 = np.isfinite(    ta_Value )

    Bool_good = bool1 * bool2 * bool3

    if Geometry == 'Thin_Shell_Cont':

        #print('Special treatment for a special geometry!')

        bool4 = np.isfinite( logEW_Value )
        bool5 = np.isfinite(    Wi_Value )

        Bool_good = Bool_good * bool4 * bool5

    if not Bool_good : return Bool_good

    if Geometry in [ 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]:

        if V_Value <= 100.0 and logNH_Value >= 20.5 : Bool_good = False

    if Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]:

        if V_Value <=   10.0 : Bool_good = False 
        if V_Value >= 1000.0 : Bool_good = False 

        if logNH_Value <=   17.0 : Bool_good = False 
        if logNH_Value >=   22.0 : Bool_good = False 

        if ta_Value <=  10**(-3.75 ) : Bool_good = False 
        if ta_Value >=  10**(-0.125) : Bool_good = False 

    if Geometry in [ 'Thin_Shell_Cont' ]:

        if V_Value <=   00.0 : Bool_good = False 
        if V_Value >= 1000.0 : Bool_good = False 

        if logNH_Value <=   17.0 : Bool_good = False 
        if logNH_Value >=   21.5 : Bool_good = False 

        if ta_Value <=  10**(-4.00 ) : Bool_good = False 
        if ta_Value >=  10**( 0.0  ) : Bool_good = False 

        if logEW_Value <= -1. : Bool_good = False
        if logEW_Value >=  3. : Bool_good = False

        if Wi_Value <= 0.01 : Bool_good = False 
        if Wi_Value >= 6.0  : Bool_good = False
 
    return Bool_good
#====================================================================#
#====================================================================#
#====================================================================#
def Compute_Inflow_From_Outflow( w_Arr , f_out_Arr ):

    '''
        Computes the line profile of an inflow from the line profiles of an
        outflow

        **Input**:

        w_Arr : 1-D sequence of floats
                    wavelength where the line profile is evaluated.

        f_out_Arr : float
                    Outflow flux density (line profile)
        
        **Output**:

        f_in_Arr : 1-D sequence of bool
                    Inflow flux density (line profile)
    '''
    w_Lya = 1215.67 * 1e-10

    Delta_Arr = w_Arr - w_Lya

    f_in_Arr = np.interp( Delta_Arr , -1. * Delta_Arr[::-1] , f_out_Arr[::-1] )

    return f_in_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def RT_Line_Profile_MCMC( Geometry , wavelength_Arr , V_Value , logNH_Value , ta_Value , DATA_LyaRT , logEW_Value=None , Wi_Value=None ):

    '''
        Return one and only one Lyman alpha line profile for a given outflow properties.
        This function is especial to run MCMCs or PSO.

        **Input**:

        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab', 'Thin_Shell_Cont'

        wavelength_Arr : 1-D sequence of floats
                         Array with the wavelength vales where the line
                         profile is computed. The units are meters, i.e.,
                         amstrongs * 1.e-10.

        V_Value : float
                  Value of the expansion velocity of the outflow. The unit
                  are km/s. 

        logNH_Value : float
                      Value of the logarithim of the outflow neutral hydrogen
                      column density. The units of the colum density are in c.g.s,
                      i.e, cm**-2. 

        ta_Value : float
                 Value of the dust optic depth of the outflow. 

        DATA_LyaRT : Dictionay
                     This dictonary have all the information of the grid.
                     This dictionary can be loaded with the function : 
                     load_Grid_Line, for example:

                     DATA_LyaRT = load_Grid_Line( 'Thin_Shell' ) 

        **Output**:

        lines_Arr : 1-D sequence of float
                    The Lyman alpha line profile. 
    '''

    Bool_good = pre_treatment_Line_profile_MCMC( Geometry , np.absolute( V_Value ) , logNH_Value , ta_Value , logEW_Value=logEW_Value , Wi_Value=Wi_Value )

    x_Arr = convert_lamda_into_x( wavelength_Arr )

    if Bool_good :
        logta_Value = np.log10( ta_Value )

        if Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out']:

            line_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC( np.absolute( V_Value ) , logNH_Value , logta_Value , x_Arr , DATA_LyaRT )

        if Geometry in [ 'Thin_Shell_Cont' ]:

            line_Arr = Interpolate_Lines_Arrays_5D_grid_MCMC( np.absolute( V_Value )  , logNH_Value , logta_Value , logEW_Value , Wi_Value , x_Arr , DATA_LyaRT )

    if not Bool_good :

        line_Arr = np.ones( len(x_Arr) ) * np.nan

    ###########################################
    CORRECT_FLAT_X = True
    if CORRECT_FLAT_X:

        tmp_line_Arr = line_Arr * wavelength_Arr **2 

        line_Arr = tmp_line_Arr * np.amax( line_Arr ) / np.amax( tmp_line_Arr )
    ###########################################

    if V_Value < 0 :

        line_Arr = Compute_Inflow_From_Outflow( wavelength_Arr , line_Arr )

    return line_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_Line_profile( Geometry , V_Arr , logNH_Arr , ta_Arr , logEW_Arr=None , Wi_Arr=None ):
    '''
        Checks the inflow/outflow parameters before doing the proper computation.

        **Input**:

        Geometry : String
                    Outflow configuration to use: 'Thin_Shell'  , 'Galactic_Wind'  
                                       , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out',
                                         'Thin_Shell_Cont'

        V_Value : 1-D sequence of bool
                    Outflow bulk velocity [km/s] 
        
        logNH_Value : 1-D sequence of bool
                    logarithm of the neutral hydrogen column density [cm**-2] 

        ta_Value : 1-D sequence of bool
                    Dust optical depth [no dimensions]

        logEW_Value : Optional 1-D sequence of bool
                    Logarithm of rest frame equiavlent width [A]
                    Default = None

        Wi_Value : Optional 1-D sequence of bool
                    Intrinsic width line in the rest frame [A]
                    Default = None

        **Output**:

        Bool_good : 1-D sequence of bool
                    1 if the parameters are good, 0 if they are bad.  
    '''

    V_Arr     = np.atleast_1d(     V_Arr )
    logNH_Arr = np.atleast_1d( logNH_Arr )
    ta_Arr    = np.atleast_1d(    ta_Arr )

    V_Arr     =     V_Arr.astype(float)
    logNH_Arr = logNH_Arr.astype(float)
    ta_Arr    =    ta_Arr.astype(float)

    bool1 = np.isfinite( V_Arr     )
    bool2 = np.isfinite( logNH_Arr )
    bool3 = np.isfinite( ta_Arr    )

    mask_good = bool1 * bool2 * bool3

    assert sum( mask_good ) != 0 , 'All the V-logNH-ta combinations are np.nan, -np.inf or np.inf'

    for i in range( 0 , len(V_Arr) ):

        tmp_bool = pre_treatment_Line_profile_MCMC( Geometry , V_Arr[i] , logNH_Arr[i] , ta_Arr[i] , logEW_Value=logEW_Arr[i] , Wi_Value=Wi_Arr[i] )

        mask_good[i] = tmp_bool

    #return V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good
    return mask_good
#====================================================================#
#====================================================================#
#====================================================================#
def RT_Line_Profile( Geometry , wavelength_Arr , V_Arr , logNH_Arr , ta_Arr , logEW_Arr=None , Wi_Arr=None , MODE_CONT='FULL' ):

    '''
        Return the Lyman alpha line profile for a given outflow properties.
        
        **Input**:

        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab', 'Thin_Shell_Cont'
        
        wavelength_Arr : 1-D sequence of floats
                         Array with the wavelength vales where the line 
                         profile is computed. The units are meters, i.e.,
                         amstrongs * 1.e-10.
        
        V_Arr : 1-D sequence of float 
                Array with the expansion velocity of the outflow. The unit
                are km/s. 
        
        logNH_Arr : 1-D sequence of float
                    Array with the logarithim of the outflow neutral hydrogen 
                    column density. The units of the colum density are in c.g.s,
                    i.e, cm**-2. 
        
        ta_Arr : 1-D sequence of float
                 Array with the dust optic depth of the outflow. 

        ta_Value : 1-D sequence of bool
                    Dust optical depth [no dimensions]

        logEW_Value : Optional 1-D sequence of bool
                    Logarithm of rest frame equiavlent width [A]
                    Default = None

        Wi_Value : Optional 1-D sequence of bool
                    Intrinsic width line in the rest frame [A]
                    Default = None

        MODE : optinal string.
               For the 'Thin_Shell_Cont' ONLY. Defines the grid to be loaded.
               MODE_CONT='FULL'  loads a very dense  grid. ~12GB of RAM.
               MODE_CONT='LIGHT' loads a more sparse grid. ~ 2GB of RAM.

        **Output**:

        lines_Arr : 2-D sequence of float
                    The Lyman alpha line profiles. lines_Arr[i] is the line profile 
                    computed at the wavelengths wavelength_Arr for wich V_Arr[i] , 
                    logNH_Arr[i] , ta_Arr[i] , Inside_Bicone_Arr[i].
    '''

    assert Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' , 'Thin_Shell_Cont'] , 'The requested geoemtry ' + Geometry + ' is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'

    V_Arr             = np.atleast_1d(     V_Arr )
    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )

    if Geometry == 'Thin_Shell_Cont':

        assert not logEW_Arr is None , 'logEW_Arr can not be non if Geometry == Thin_Shell_Cont'
        assert not    Wi_Arr is None , 'Wi_Arr    can not be non if Geometry == Thin_Shell_Cont'

        logEW_Arr = np.atleast_1d( logEW_Arr )
        Wi_Arr    = np.atleast_1d(    Wi_Arr )

    x_Arr = convert_lamda_into_x( wavelength_Arr )

    lines_Arr = np.zeros( len(V_Arr) * len( x_Arr ) ).reshape( len(V_Arr) , len( x_Arr ) ) * np.nan
       
    mask_good = pre_treatment_Line_profile( Geometry , V_Arr , logNH_Arr , ta_Arr , logEW_Arr=logEW_Arr , Wi_Arr=Wi_Arr )

    logta_Arr = np.log10( ta_Arr )

    ##############################

    DATA_LyaRT = load_Grid_Line( Geometry , MODE_CONT )

    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ] :

        tmp_lines_Arr = Interpolate_Lines_Arrays_3D_grid( V_Arr[ mask_good ] , logNH_Arr[ mask_good ] , logta_Arr[ mask_good ] , x_Arr , DATA_LyaRT )

    if Geometry in [ 'Thin_Shell_Cont' ] :

        tmp_lines_Arr = Interpolate_Lines_Arrays_5D_grid( V_Arr[ mask_good ] , logNH_Arr[ mask_good ] , logta_Arr[ mask_good ] , logEW_Arr[ mask_good ] , Wi_Arr[ mask_good ] , x_Arr , DATA_LyaRT )

    ##############################

    lines_Arr[ mask_good ] = tmp_lines_Arr

    return lines_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Test_1( ):
    '''
        Script to test if everything is working fine.
    '''

    print( '\n Let us check every different configuration for computing the escape fraction and the line profiles.')
 
    Geometry_set = [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' , 'Thin_Shell_Cont']
    
    ML_codes_set = [ 'Tree' , 'Forest' , 'KN' ]
    
    MODE_set = [ 'Parametrization' , 'Raw' , 'Analytic' ]
    
    Algorithm_set = [ 'Intrepolation' , 'Machine_Learning' ]
    
    # Primero vamos a checkear que funciona las fracciones de escape
    
    N_points = int( 1e4 )
    
    V_Arr     = np.random.rand( N_points ) * 989 + 10.0
    logNH_Arr = np.random.rand( N_points ) *   5 + 17.0
    logta_Arr = np.random.rand( N_points ) *   2 - 2.5  
   
    mask_bad_bicone = ( V_Arr <= 100 ) * ( logNH_Arr >= 20.5 ) 

    V_Arr     =     V_Arr[ ~mask_bad_bicone ]
    logNH_Arr = logNH_Arr[ ~mask_bad_bicone ]
    logta_Arr = logta_Arr[ ~mask_bad_bicone ]
 
    print( '\nComputing', N_points , 'random configurations of escape fraction with each algorithms...\n')
    
    for Geo in Geometry_set:
   
        if Geo == 'Thin_Shell_Cont' : continue
 
        for Mod in MODE_set :
    
            if not Mod in [ 'Analytic' ]:
    
                for Algo in Algorithm_set:
    
                    if Algo in [ 'Intrepolation' , 'Machine_Learning' ]:
    
                        if Algo == 'Machine_Learning' :
    
                            for machine in ML_codes_set :
    
                                try:
                                    print( '      Running : ' , Geo , Mod , Algo , machine , end = '' )
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , MODE=Mod , Algorithm=Algo , Machine_Learning_Algorithm=machine)
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print( '--> Success!!')
                                except:
                                    print( '--> ERROR.. MISMATCH!!')
    
                        if Algo != 'Machine_Learning' :
    
    
                                try:
                                    print( '      Running : ' , Geo , Mod , Algo , end = '' )
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , MODE=Mod , Algorithm=Algo )
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print( '--> Success!!')
    
                                except:
                                    print( '--> ERROR. MISMATCH!!')
    
            if Mod in [ 'Analytic' ]:
    
    
                                try:
                                    print( '      Running : ' , Geo , Mod , end = '' )
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , MODE=Mod )
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print( '--> Success!!')
    
                                except:
                                    print( '--> ERROR. MISMATCH!!')
    
    
    
    N_points = int( 1e3 )
    
    print( '\nComputing', N_points , 'random configurations of line profile  with each algorithms...\n')
    
    V_Arr     = np.random.rand( N_points ) * 900  + 100
    logNH_Arr = np.random.rand( N_points ) *   4.5 + 17.0
    logta_Arr = np.random.rand( N_points ) *   2.  -  2.5
   
    logEW_Arr = np.random.rand( N_points ) * 3 - 1.
    Wi_Arr    = np.random.rand( N_points ) * 2 + 0.1 
 
    wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10
    
    RUN_TEST_Lines = True
    if RUN_TEST_Lines :
        for Geo in Geometry_set:
    
            try: 
                print( '      Running : ' , Geo , end = '' )

                qq = RT_Line_Profile( Geo , wavelength_Arr , V_Arr , logNH_Arr , 10**logta_Arr , logEW_Arr=logEW_Arr , Wi_Arr=Wi_Arr )
                assert np.sum( np.isnan( qq ) ) == 0
                print( '--> Success!!')
            except:
                print( '--> ERROR. MISMATCH!!' )

    return
#====================================================================#
#====================================================================#
#====================================================================#
def Test_2( ):
    '''
        Script to test if everything looks fine.
    '''

    import matplotlib.pyplot as plt

    print( '\n Let us make some plots. This will show you just a glimpse of what LyaRT;Grid can do. Just wait for it...')

    # Plot some nice line profiles

    print( '\n    Plotting some line profiles...')

    wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10

    V_Arr = np.array( [ 10 , 50 , 100 , 200 , 300 ] )

    logNH_Arr = np.array( [ 20.0 ] * len( V_Arr ) )

    logta_Arr = np.array( [ -1. ] * len( V_Arr ) )

    Wi_Arr    = np.array( [ 0.4 ] * len( V_Arr ) )
    logEW_Arr = np.array( [ 1.5 ] * len( V_Arr ) )

    Inside_Bicone_Arr = np.zeros( len(V_Arr) ) == 0

    cm = plt.get_cmap( 'rainbow' )
    
    for geo in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' , 'Thin_Shell_Cont' ]:

        qq = RT_Line_Profile( geo , wavelength_Arr , V_Arr , logNH_Arr , 10.**logta_Arr , logEW_Arr=logEW_Arr , Wi_Arr=Wi_Arr ) 

        plt.figure()

        ax_ax = plt.subplot(111)

        for i in range( 0 ,len( V_Arr ) ):

            ax_ax.plot( wavelength_Arr*1e10 , qq[i] , color=cm( i*1./( len(V_Arr) -1 ) ) , lw=2 )

        ax_ax.set_ylabel( r'Flux [a.u.]' , size=20 )
        ax_ax.set_xlabel( r'Wavelength [AA]' , size=20 )

        ax_ax.set_title( r'Geometry = ' + geo , size=20 )

        ax_ax.set_xlim( 1212.5 , 1222.5 )

        ax_ax.legend(loc=0)

    print( '\n    Plotting some escape fractions...')

    logta_Arr = np.linspace( -2 , 0.5 , 20 )

    logNH_Arr = np.array( [20.0] * len( logta_Arr ) )

    for geo in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab_In' , 'Bicone_X_Slab_Out' ]  :

        plt.figure()

        ax_ax = plt.subplot(111)

        for i in range( 0 , len(V_Arr) ):

            V_Arr_tmp = np.array( [ V_Arr[i] ] * len( logta_Arr ) )

            f_esc = RT_f_esc( geo , V_Arr_tmp , logNH_Arr , 10**logta_Arr )

            ax_ax.semilogy( logta_Arr , f_esc , '--' , color=cm( i*1./( len(V_Arr) -1 ) ) , lw=2 )

        ax_ax.set_xlabel( r'log tau a' , size=20 )
        ax_ax.set_ylabel( r'f esc Ly alpha ' , size=20 )

        ax_ax.set_title( r'Geometry = ' + geo , size=20 )

        plt.legend( loc=0 )

    plt.show()

    return
#====================================================================#
#====================================================================#
#====================================================================#
#
#           CODE  FOR  THE  NEW  VERSION    !!!          YEEEY
#
#====================================================================#
#====================================================================#
#====================================================================#
def convert_gaussian_FWHM_to_sigma( FWHM_Arr ):

    '''
        This function computes the sigma of a gaussian from its FWHM.

        **Input**

        FWHM_Arr : 1-D sequence of float
                   Array with the Full Width Half Maximum that you 
                   want to convert

        **Output**

        sigma_Arr : 1-D sequence of float
                    The width of the FWHM_Arr
    '''

    sigma_Arr = FWHM_Arr * 1. / 2.3548

    return sigma_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def dilute_line_changing_FWHM( wave_Arr , Spec_Arr , FWHM_Arr , same_norm=False ):

    '''
        This functions dilutes a given spectrum by convolving with a gaussian 
        filter.

        **Input**

        wave_Arr : 1-D sequence of float
                   Array with the Wavelength where the spectrum is evaluated.
                   Same units as FWHM_Arr. This has to be sorted.
        
        Spec_Arr : 1-D sequence of float
                   Arrays with the flux of the spectrum.

        FWHM_Arr : 1-D sequence of float
                   Array with the Full width half maximuum of of the gaussian
                   to convolve. 
                   If FWHM_Arr is a single value, it uses the same value across
                   the x_Arr range.
                   If FWHM is a 1-D sequence, a different value of width of
                   the gaussian is used. In this case, the length of this array
                   has to be the same as wave_Arr and Spec_Arr.

        same_norm : optional bool.
                    If true return a line with the same normalization as the input

        **Output**

        new_Line : 1-D sequence of float
                   Spectrum after the convolution
    '''


    my_sigma_Arr = convert_gaussian_FWHM_to_sigma( FWHM_Arr )

    new_Line = gaussian_filter( wave_Arr , Spec_Arr , my_sigma_Arr ,same_norm=same_norm)

    return new_Line
#====================================================================#
#====================================================================#
#====================================================================#
def dilute_line( wave_Arr , Spec_Arr , FWHM ):

    '''
        This functions dilutes a given spectrum by convolving with a gaussian
        filter.

        **Input**

        wave_Arr : 1-D sequence of float
                   Array with the Wavelength where the spectrum is evaluated.
                   Same units as FWHM_Arr. This has to be sorted.

        Spec_Arr : 1-D sequence of float
                   Arrays with the flux of the spectrum.

        FWHM_Arr : 1-D sequence of float
                   Array with the Full width half maximuum of of the gaussian
                   to convolve.
                   If FWHM_Arr is a single value, it uses the same value across
                   the x_Arr range.
                   If FWHM is a 1-D sequence, a different value of width of
                   the gaussian is used. In this case, the length of this array
                   has to be the same as wave_Arr and Spec_Arr.

        same_norm : optional bool.
                    If true return a line with the same normalization as the input

        **Output**

        new_Line : 1-D sequence of float
                   Spectrum after the convolution
    '''

    sigma = convert_gaussian_FWHM_to_sigma( FWHM )

    bin_size = wave_Arr[1] - wave_Arr[0] 

    new_Line = gaussian_filter1d( Spec_Arr , sigma * 1. / bin_size )

    return new_Line
#====================================================================#
#====================================================================#
#====================================================================#
def bin_one_line( wave_Arr_line , Line_Prob_Arr , new_wave_Arr , Bin , same_norm=False ):
    '''
        This functions bins the line profile mimicking the pixelization in a CCD.

        **Input**

        wave_Arr_line : 1-D sequence of float
                   Array with the Wavelength where the spectrum is evaluated.
                   Same units as Bin. This has to be sorted.

        Line_Prob_Arr : 1-D sequence of float
                   Arrays with the flux of the spectrum.

        new_wave_Arr : 1-D sequence of float
                   Array with the nex wavelgnth where the fline profile will be
                   interpolated 

        Bin : float
                   Bin size.

        same_norm : optional bool.
                    If true return a line with the same normalization as the input

        **Output**

        binned_line : 1-D sequence of float
                   Spectrum after the convolution
    '''

    RES = 100

    binned_line = np.zeros( len( new_wave_Arr) )

    for i in range( 0 , len( new_wave_Arr) ) :

        wave_low = new_wave_Arr[i] - Bin*0.5
        wave_top = new_wave_Arr[i] + Bin*0.5

        High_res_wave_Arr = np.linspace( wave_low , wave_top , RES )

        High_res_line_in_bin = np.interp( High_res_wave_Arr , wave_Arr_line , Line_Prob_Arr , left=Line_Prob_Arr[0] , right=Line_Prob_Arr[-1] )

        binned_line[i] = np.mean( High_res_line_in_bin )

    #print( 'binnec_line = ' , binned_line )

    if same_norm:

        I_init = np.trapz( Line_Prob_Arr , wave_Arr_line )

        I_pix = np.trapz( binned_line , new_wave_Arr )

        binned_line = binned_line * I_init * 1. / I_pix

    return binned_line
#====================================================================#
#====================================================================#
#====================================================================#
def plot_a_rebinned_line( new_wave_Arr , binned_line , Bin ):
    '''
        This functions is used to plot line profiles. It transforms the line
        line in a histogram.

        **Input**

        new_wave_Arr : 1-D sequence of float
                   Array with the Wavelength where the spectrum is evaluated.

        binned_line : 1-D sequence of float
                   Arrays with the flux of the spectrum.

        Bin : float
                   Bin size.

        **Output**

        XX_Arr : 1-D sequence of float
                   Wavelength where the new line is evaluated

        YY_Arr : 1-D sequence of float
                   Flux density array
    '''

    DD = Bin  * 1e-10

    XX_Arr = np.zeros( len( new_wave_Arr ) * 2 )
    YY_Arr = np.zeros( len( new_wave_Arr ) * 2 )

    for i in range( 0 , len( new_wave_Arr ) ):

        i_0 = 2 * i
        i_1 = 2 * i + 1

        XX_Arr[ i_0 ] = new_wave_Arr[i] - 0.5 * Bin + DD
        XX_Arr[ i_1 ] = new_wave_Arr[i] + 0.5 * Bin

        YY_Arr[ i_0 ] = binned_line[i]
        YY_Arr[ i_1 ] = binned_line[i]

    return XX_Arr , YY_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Add_noise_to_line( wave_Arr_line , Line_Prob_Arr , SN , same_norm=False ):
    '''
        This functions adds noise to a line profile.

        **Input**

        wave_Arr_line : 1-D sequence of float
                   Array with the Wavelength where the spectrum is evaluated.
                   Same units as Bin. This has to be sorted.

        Line_Prob_Arr : 1-D sequence of float
                   Arrays with the flux of the spectrum.

        SN : float
                   Signal to noise of the desire line.

        same_norm : optional bool.
                    If true return a line with the same normalization as the input

        **Output**

         Noisy_Line_Arr : 1-D sequence of float
                   Spectrum with the noise

        Noise_Arr : 1-D sequence of float
                   Particular flux density of the noise applyed
    '''
    mask = Line_Prob_Arr > 0.05 * np.amax( Line_Prob_Arr )

    SS = np.mean( Line_Prob_Arr[ mask ] )

    Noise_level = SS * 1. / SN 

    Noise_Arr = np.random.randn( len(Line_Prob_Arr) ) * Noise_level 

    Noisy_Line_Arr = Line_Prob_Arr + Noise_Arr

    if same_norm :

        I_init = np.trapz( Line_Prob_Arr , wave_Arr_line )

        I_noise = np.trapz( Noisy_Line_Arr , wave_Arr_line )

        Noisy_Line_Arr = Noisy_Line_Arr * I_init * 1. / I_noise 

    return Noisy_Line_Arr , Noise_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def gaus( x_Arr , A , mu , sigma ):
    '''
        Retruns a gaussian evaluated in x_Arr.
        
        **Input**
        
        x_Arr : 1-D sequence of float
                   Where the gaussian has to be evaluated.
        
        A : float
            Amplitude
        
        mu : float
            Mean
        
        sigma : float
            width
        
        **Output**
        
        gauss_Arr : 1-D sequence of float
                   Gaussian
    '''
    
    gauss_Arr =  A * np.exp( -1*( x_Arr - mu )**2 * 1. / ( 2 * sigma**2 ) )
    
    return gauss_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Signal_to_noise_estimator( w_Arr , Line_Arr , w_line ):
    '''
        Estimates the signal to noise of a line profile

        **Input**

        w_Arr : 1-D sequence of float
                wavelgnth array

        Line_Arr : 1-D sequence float
                   Flux density of the line profile.

        w_line : float
                 wavelgnth of the line

        **Output**

        SNR : float
              Signal to noise ratio.
    '''
    popt , pcov = curve_fit( gaus , w_Arr , Line_Arr , p0=[ 1e-17 , w_line , 5.0 ] )

    sigma_line = popt[2]

    mask_line = ( w_Arr > w_line - 2.30*sigma_line ) * ( w_Arr < w_line + 2.30*sigma_line )

    RMS = np.sqrt( np.mean( Line_Arr[ ~mask_line ]**2 ) )

    SS = np.mean( Line_Arr[ mask_line ] )

    SNR = SS * 1. / RMS

    return SNR
#====================================================================#
#====================================================================#
#====================================================================#
def generate_a_obs_line( z_f , V_f , logNH_f , ta_f , DATA_LyaRT , Geometry , logEW_f=None , Wi_f=None ):
    '''
        Moves in redshift a line profile.

        **Input**

        z_f : float
              Redshift

        V_f : float
              Outflow expansion velocity [km/s]

        logNH_f : float
                  logarithmic of the neutral hydrogen column density in cm**-2

        ta_f : float
               Dust optical depth

        DATA_LyaRT : python dictionary
                     Contains the grid information. 

        Geometry : string
                   Outflow geometry to use.

        logEW_f : optional, float
                  Logarithmic of the rest frame intrisic equivalent width of the line [A]
                  Requiered if Geometry == 'Thin_Shell_Cont' 

        Wi_f : optional, float
               Rest frame intrisic width of the Lyman-alpha line [A]
               Requiered if Geometry == 'Thin_Shell_Cont' 

        **Output**

        w_rest_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the rest frame.

        wavelength_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the observed frame.

        line_Arr : 1-D sequence of float
                   Line profile flux density in arbitrary units.

    '''
    w_Lya = 1215.67

    w_rest_Arr = np.linspace( 1215.68 - 30 , 1215.68 + 30 , 2000 )

    line_Arr = RT_Line_Profile_MCMC( Geometry , w_rest_Arr*1e-10, V_f , logNH_f , ta_f , DATA_LyaRT , logEW_Value=logEW_f , Wi_Value=Wi_f )

    wavelength_Arr = ( 1 + z_f ) * w_rest_Arr

    return w_rest_Arr , wavelength_Arr , line_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def generate_a_REAL_line_Noise_w( z_f , V_f , logNH_f , ta_f , F_line_f , logEW_f , Wi_f , Noise_w_Arr , Noise_Arr , FWHM_f , PIX_f , w_min , w_max , DATA_LyaRT , Geometry ):
    '''
        Makes a mock line profile for the Thin_Shell_Cont geometry.

        **Input**

        z_f : float
              Redshift

        V_f : float
              Outflow expansion velocity [km/s]

        logNH_f : float
                  logarithmic of the neutral hydrogen column density in cm**-2

        ta_f : float
               Dust optical depth

        logEW_f : optional, float
                  Logarithmic of the rest frame intrisic equivalent width of the line [A]
                  Requiered if Geometry == 'Thin_Shell_Cont'

        Wi_f : optional, float
               Rest frame intrisic width of the Lyman-alpha line [A]
               Requiered if Geometry == 'Thin_Shell_Cont'

        Noise_w_Arr : 1-D sequence of float
                      wavelength array where the noise pattern is evaluated.

        Noise_Arr : 1-D sequence of float
                    Noise pattern. Evolution of the noise as a function of wavelength (Noise_w_Arr)
       
        FWHM_f : float
                 Full width half maximum [A] of the experiment. This dilutes the line profile.  

        PIX_f : float
                Pixel size in wavelgnth [A] of the experiment. This binnes the line profile.  

        w_min : float
                minimum wavelength in the observed frame [A] to use. This matches the minimum
                wavelgnth of wave_pix_Arr (see below).

        w_max : float
                maximum  wavelength in the observed frame [A] to use. This might not be exactly
                the maximum wavelgnth of wave_pix_Arr (see below) due to pixelization. 

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        **Output**

        wave_pix_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the observed frame.

        noisy_Line_Arr : 1-D sequence of float
                   Line profile flux density in arbitrary units.

        dic : python dictionaty.
              Contains all the meta data used to get the line profiles:
                'w_rest'    : restframe wavelength of line before reducing quality
                'w_obs'     : wavelength of line before reducing quality
                'Intrinsic' : line profile before quality reduction
                'Diluted'   : Line profile after the FWHM has been applyed.
                'Pixelated' : Line profile after the FWHM and PIX have been applyed
                'Noise'     : Particular noise patern used.
    '''
    w_rest_Arr , wavelength_Arr , line_Arr = generate_a_obs_line( z_f , V_f , logNH_f , ta_f , DATA_LyaRT , Geometry , logEW_f=logEW_f , Wi_f=Wi_f )

    diluted_Arr = dilute_line( wavelength_Arr , line_Arr , FWHM_f )

    wave_pix_Arr = np.arange( w_min , w_max , PIX_f )

    pixled_Arr = bin_one_line( wavelength_Arr , diluted_Arr , wave_pix_Arr , PIX_f )

    F_lambda_Arr = pixled_Arr * F_line_f * 1. / np.trapz( pixled_Arr , wave_pix_Arr )

    noise_in_my_w_Arr = np.interp( pixled_Arr , Noise_w_Arr , Noise_Arr )

    noise_Arr = np.random.randn( len( pixled_Arr ) ) * noise_in_my_w_Arr

    noisy_Line_Arr = F_lambda_Arr + noise_Arr

    # Other usful things :

    dic = {}

    dic[ 'w_rest'    ] = w_rest_Arr
    dic[ 'w_obs'     ] = wavelength_Arr
    dic[ 'Intrinsic' ] = line_Arr
    dic[ 'Diluted'   ] = diluted_Arr
    dic[ 'Pixelated' ] = F_lambda_Arr
    dic[ 'Noise'     ] = noise_Arr

    return wave_pix_Arr , noisy_Line_Arr , dic
#====================================================================#
#====================================================================#
#====================================================================#
def Generate_a_real_line( z_t , V_t, log_N_t, t_t, F_t, log_EW_t, W_t , PNR_t, FWHM_t, PIX_t, DATA_LyaRT, Geometry ):
    '''
        Makes a mock line profile for the Thin_Shell_Cont geometry.

        **Input**

        z_t : float
              Redshift

        V_t : float
              Outflow expansion velocity [km/s]

        log_N_t : float
                  logarithmic of the neutral hydrogen column density in cm**-2

        t_t : float
              Dust optical depth

        F_t : float
              Total flux of the line. You can pass 1.

        log_EW_t : optional, float
                   Logarithmic of the rest frame intrisic equivalent width of the line [A]
                   Requiered if Geometry == 'Thin_Shell_Cont'

        W_t : optional, float
               Rest frame intrisic width of the Lyman-alpha line [A]
               Requiered if Geometry == 'Thin_Shell_Cont'

        PNR_t : float
                Signal to noise ratio of the global maximum of the line profile.

        FWHM_t : float
                 Full width half maximum [A] of the experiment. This dilutes the line profile.

        PIX_t : float
                Pixel size in wavelgnth [A] of the experiment. This binnes the line profile.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        **Output**

        w_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the observed frame.

        f_Arr : 1-D sequence of float
                   Line profile flux density in arbitrary units.

        noise_Amplitude_Arr : 1-D sequence of float
                              1-sigma level of the applyed noise.
    '''

    w_min = ( 1215.67 - 25.5 ) * ( 1 + z_t )
    w_max = ( 1215.67 + 25.5 ) * ( 1 + z_t )

    w_Noise_Arr = np.linspace( w_min , w_max , 10 )

    tmp_Noise_Arr = w_Noise_Arr * 0.0

    w_Arr , f_noiseless_Arr , _ = generate_a_REAL_line_Noise_w( z_t, V_t, log_N_t, t_t , F_t , log_EW_t , W_t, w_Noise_Arr, tmp_Noise_Arr , FWHM_t , PIX_t ,  w_min, w_max, DATA_LyaRT, Geometry )

    noise_Amplitude = np.amax( f_noiseless_Arr ) * 1. / PNR_t

    noise_Arr = np.random.randn( len(w_Arr) ) * noise_Amplitude

    noise_Amplitude_Arr = np.ones( len( w_Arr ) ) * noise_Amplitude

    f_Arr = f_noiseless_Arr + noise_Arr

    return w_Arr , f_Arr , noise_Amplitude_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Define_wavelength_for_NN( Delta_min=-18.5 , Delta_max=18.5 , Nbins_tot=1000 , Denser_Center=True ):
    '''
        This function defines the wavelength used in for the neural netwroks.

        **Input**

        Delta_min : optional float
              Defines the minimum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = -18.5

        Delta_min : optional float
              Defines the maximum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = +18.5

        Nbins_tot : optional int
              Total number of wvelgnths bins.
              
              Default = 1000

        Denser_Center : optional bool
              Populates denser the regions close to Lyman-alpha
              
              Default = True

        **Output**

        rest_w_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the rest frame.
    '''
    if not Denser_Center:
    
        rest_w_Arr = np.linspace( 1215.67+Delta_min , 1215.67+Delta_max , Nbins_tot )
    
    else :
    
        Delta_core = 4.0
    
        ratio_bin_size_core = 5
    
        L_core = 2*Delta_core
        L_wing = Delta_max - Delta_min - L_core
    
        ratio_Ls = L_core * 1. / L_wing
    
        N_c = ratio_bin_size_core * ratio_Ls * Nbins_tot /  ( 1 + ratio_bin_size_core * ratio_Ls )
    
        N_w = Nbins_tot - N_c
    
        N_c = int( N_c )
        N_w = int( N_w )
    
        N_w = int( N_w * 0.5 ) * 2
    
        while N_c+N_w<Nbins_tot : N_c += 1
    
        #print( N_c , N_w )
    
        Delta_c = L_core * 1. / N_c
    
        low_wing = np.linspace( Delta_min          , -Delta_core-Delta_c , int( N_w / 2 ) )
        top_wing = np.linspace( Delta_core+Delta_c ,  Delta_max          , int( N_w / 2 ) )
    
        core = np.linspace( -Delta_core , Delta_core , N_c )
    
        rest_w_Arr = np.hstack([ low_wing , core , top_wing ]) + 1215.67

    return rest_w_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def Treat_A_Line_To_NN_Input( w_Arr , f_Arr , PIX , FWHM , Delta_min=-18.5 , Delta_max=18.5 , Nbins_tot=1000 , Denser_Center=True , normed=False, scaled=True ):
    '''
        Convert a line profile and the usefull information into the INPUT of the NN.

        **Input**

        w_Arr : 1-D sequence of floats
              Wavelgnth of the line profile in the observed frame. [A]

        f_Arr : 1-D sequence of floats
              Flux density of the observed line profile in arbitrary units.

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment. 
        

        Delta_min : optional float
              Defines the minimum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = -18.5

        Delta_min : optional float
              Defines the maximum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = +18.5

        Nbins_tot : optional int
              Total number of wvelgnths bins.

              Default = 1000

        Denser_Center : optional bool
              Populates denser the regions close to Lyman-alpha

              Default = True

        normed : optional bool
              If True, nomalizes the line profile.

        scaled : optinal bool
              If True, divides the line profile by its maximum.

        **Output**

        rest_w_Arr : 1-D sequence of float
              Wavelgnth array where the line is evaluated in the rest frame.

        NN_line : 1-D sequence of float
              Line profile evaluated in rest_w_Arr after normalization or scaling.

        z_max_i : float
              Redshift of the source if the global maximum of the spectrum is the
              Lyman-alpha wavelegth.

        INPUT: 1-D secuence of float
              The actuall input to use in the Neural networks.
    '''
    rest_w_Arr = Define_wavelength_for_NN( Delta_min=Delta_min , Delta_max=Delta_max , Nbins_tot=Nbins_tot , Denser_Center=Denser_Center ) 

    w_r_i_Arr , Delta_r_i_Arr , f_r_i_Arr , z_max_i = NN_convert_Obs_Line_to_proxy_rest_line( w_Arr, f_Arr, normed=normed , scaled=scaled)

    NN_line = np.interp( rest_w_Arr , w_r_i_Arr , f_r_i_Arr , left=f_r_i_Arr[0] , right=f_r_i_Arr[-1] )

    NN_line = NN_line * 1. / np.amax( NN_line )

    INPUT =  np.array( [ np.hstack( ( NN_line , z_max_i , np.log10( FWHM )  , np.log10( PIX )  ) ) ] )

    return rest_w_Arr , NN_line , z_max_i , INPUT 
#====================================================================#
#====================================================================#
#====================================================================#
def Generate_a_line_for_training( z_t , V_t, log_N_t, t_t, F_t, log_EW_t, W_t, PNR_t, FWHM_t, PIX_t, DATA_LyaRT, Geometry, normed=False, scaled=True , Delta_min = -18.5 , Delta_max=18.5 , Denser_Center=True , Nbins_tot=1000 ):
    
    '''
        Creates a mock line profile at the desired redshift and returns all the NN
        products.

        **Input**

        z_t : float
              Redshift

        V_t : float
              Outflow expansion velocity [km/s]

        log_N_t : float
                  logarithmic of the neutral hydrogen column density in cm**-2

        t_t : float
              Dust optical depth

        F_t : float
              Total flux of the line. You can pass 1.

        log_EW_t : optional, float
                   Logarithmic of the rest frame intrisic equivalent width of the line [A]
                   Requiered if Geometry == 'Thin_Shell_Cont'

        W_t : optional, float
               Rest frame intrisic width of the Lyman-alpha line [A]
               Requiered if Geometry == 'Thin_Shell_Cont'

        PNR_t : float
                Signal to noise ratio of the global maximum of the line profile.

        FWHM_t : float
                 Full width half maximum [A] of the experiment. This dilutes the line profile.

        PIX_t : float
                Pixel size in wavelgnth [A] of the experiment. This binnes the line profile.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        Delta_min : optional float
              Defines the minimum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = -18.5

        Delta_min : optional float
              Defines the maximum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = +18.5

        Nbins_tot : optional int
              Total number of wvelgnths bins.

              Default = 1000

        Denser_Center : optional bool
              Populates denser the regions close to Lyman-alpha

              Default = True

        normed : optional bool
              If True, nomalizes the line profile.

        scaled : optinal bool
              If True, divides the line profile by its maximum.

        **Output**

        rest_w_Arr : 1-D sequence of float
              Wavelgnth array where the line is evaluated in the rest frame.

        train_line : 1-D sequence of float
              Line profile.

        z_max_i : float
              Redshift of the source if the global maximum of the spectrum is the
              Lyman-alpha wavelegth.

        INPUT: 1-D secuence of float
              The actuall input to use in the Neural networks.
    '''

    w_t_Arr , f_t_Arr , Noise_t_Arr = Generate_a_real_line( z_t , V_t, log_N_t, t_t, F_t, log_EW_t, W_t , PNR_t, FWHM_t, PIX_t, DATA_LyaRT, Geometry )

    rest_w_Arr , train_line , z_max_i , INPUT = Treat_A_Line_To_NN_Input( w_t_Arr , f_t_Arr , PIX_t , FWHM_t , Delta_min=Delta_min , Delta_max=Delta_max , Nbins_tot=Nbins_tot , Denser_Center=Denser_Center , normed=normed , scaled=scaled )

    return rest_w_Arr , train_line , z_max_i , INPUT
#====================================================================#
#====================================================================#
#====================================================================#
def generate_a_REAL_line_SN( z_f , V_f , logNH_f , ta_f , F_line_f , SN_f , FWHM_f , PIX_f , w_min , w_max , DATA_LyaRT , Geometry ):
    '''
        Makes a mock line profile for the other geometries that are not Thin_Shell_Cont geometry.

        **Input**

        z_t : float
              Redshift

        V_t : float
              Outflow expansion velocity [km/s]

        log_N_t : float
                  logarithmic of the neutral hydrogen column density in cm**-2

        t_t : float
              Dust optical depth

        F_line_f : float
              Total flux of the line. You can pass 1.

        SN_t : float
                Signal to noise ratio.

        FWHM_t : float
                 Full width half maximum [A] of the experiment. This dilutes the line profile.

        PIX_t : float
                Pixel size in wavelgnth [A] of the experiment. This binnes the line profile.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        **Output**

        wave_pix_Arr : 1-D sequence of float
                     Wavelgnth array where the line is evaluated in the observed frame.

        noisy_Line_Arr : 1-D sequence of float
                   Line profile flux density in arbitrary units.

        dic : python dictionaty.
              Contains all the meta data used to get the line profiles:
                'w_rest'    : restframe wavelength of line before reducing quality
                'w_obs'     : wavelength of line before reducing quality
                'Intrinsic' : line profile before quality reduction
                'Diluted'   : Line profile after the FWHM has been applyed.
                'Pixelated' : Line profile after the FWHM and PIX have been applyed
                'Noise'     : Particular noise patern used.
    '''

    w_rest_Arr , wavelength_Arr , line_Arr = generate_a_obs_line( z_f , V_f , logNH_f , ta_f , DATA_LyaRT , Geometry )

    diluted_Arr = dilute_line( wavelength_Arr , line_Arr , FWHM_f )

    wave_pix_Arr = np.arange( w_min , w_max , PIX_f )

    pixled_Arr = bin_one_line( wavelength_Arr , diluted_Arr , wave_pix_Arr , PIX_f )

    pixled_Arr = pixled_Arr * F_line_f * 1. / np.trapz( pixled_Arr , wave_pix_Arr )

    noisy_Line_Arr , noise_Arr = Add_noise_to_line( wave_pix_Arr , pixled_Arr , SN_f , same_norm=False )

    # Other usful things :

    dic = {}

    dic[ 'w_rest'    ] = w_rest_Arr
    dic[ 'w_obs'     ] = wavelength_Arr
    dic[ 'Intrinsic' ] = line_Arr
    dic[ 'Diluted'   ] = diluted_Arr
    dic[ 'Pixelated' ] = pixled_Arr

    return wave_pix_Arr , noisy_Line_Arr , dic
#====================================================================#
#====================================================================#
#====================================================================#

## MCMC section

#====================================================================#
#====================================================================#
#====================================================================#
def log_likelihood( w_obs_Arr , f_obs_Arr , s_obs_Arr , w_model_Arr , f_model_Arr ):

    '''
        Logarithm of the likelihood between an observed spectrum and a model spectrum.

        **Input**

        w_obs_Arr : 1-D sequence of float
                    wavelength where the observed density flux is evaluated.

        f_obs_Arr : 1-D sequence of float
                    Observed flux density

        s_obs_Arr : 1-D sequence of float
                    Uncertanty in the observed flux density.

        w_model_Arr : 1-D sequence of float
                      wavelength where the model density flux is evaluated

        f_model_Arr : 1-D sequence of float
                      Model flux density

        **Output**

        log_like : float
                   Logarithm of the likelihood
    '''
    f_my_Arr = np.interp( w_obs_Arr , w_model_Arr , f_model_Arr )

    mask = np.isfinite( f_my_Arr )

    sigma2 = s_obs_Arr ** 2

    cc = 1.0

    f_my_Arr     = f_my_Arr[  mask ]
    sigma2       = sigma2[    mask ]
    my_f_obs_Arr = f_obs_Arr[ mask ]

    log_like = -0.5 * np.sum( cc *( my_f_obs_Arr - f_my_Arr ) ** 2 / sigma2 + np.log(sigma2))

    return log_like
#====================================================================#
#====================================================================#
#====================================================================#
def Prior_f( theta ):
    '''
        Decides when a walker from the MCMC is out for the Thin_Shell,
        Galactic wind and bicones.

        **Input**

        theta : 1-D sequence of float
                Contains the parameters of the mode:
                    theta[0] = logarithim of the expansion velocity
                    theta[1] = logarithim of the neutral hydrogen column density
                    theta[2] = logarithim of the dust optical depth

        **Output**

        True  if the walker is  inside
        False if the walker is outside
    '''

    log_V , log_N , log_t , redshift = theta

    log_V_min = -1. 
    log_V_max =  3. 

    log_N_min = 17.0 
    log_N_max = 21.5

    log_t_min = -3.75
    log_t_max = -0.125

    z_min =   0.0
    z_max = 100.0

    #if log_V < np.log10( 40 ) : log_N_max = 20.5
 
    if log_V < log_V_min or log_V > log_V_max : return False
    if log_N < log_N_min or log_N > log_N_max : return False
    if log_t < log_t_min or log_t > log_t_max : return False

    if redshift < z_min or redshift > z_max : return False

    return True
#====================================================================#
#====================================================================#
#====================================================================#
def Prior_f_5( theta ):
    '''
        Decides when a walker from the MCMC is out for the Thin_Shell_Cont,

        **Input**

        theta : 1-D sequence of float
                Contains the parameters of the mode:
                    theta[0] = logarithim of the expansion velocity
                    theta[1] = logarithim of the neutral hydrogen column density
                    theta[2] = logarithim of the dust optical depth
                    theta[3] = redshift
                    theta[4] = logarithm of the intrinsic equivalent width
                    theta[5] = intrinsic width

        **Output**

        True  if the walker is  inside
        False if the walker is outside
    '''

    log_V , log_N , log_t , redshift , log_EW , Wi = theta

    log_V_min = -1. 
    log_V_max =  3. 

    log_N_min = 17.0 
    log_N_max = 21.5

    log_t_min = -4.00
    log_t_max =  0.000

    log_EW_min = -1.
    log_EW_max =  3.

    Wi_min =  0.01
    Wi_max =  6.

    z_min =   0.0
    z_max =   7.0

    if log_V  < log_V_min  or log_V  > log_V_max  : return False
    if log_N  < log_N_min  or log_N  > log_N_max  : return False
    if log_t  < log_t_min  or log_t  > log_t_max  : return False
    if log_EW < log_EW_min or log_EW > log_EW_max : return False
    if Wi     <     Wi_min or     Wi >     Wi_max : return False

    if redshift < z_min or redshift > z_max : return False

    return True
#====================================================================#
#====================================================================#
#====================================================================#
def log_likeliehood_of_model_5( theta , w_obs_Arr , f_obs_Arr , s_obs_Arr , FWHM, PIX, w_min, w_max, DATA_LyaRT, Geometry , z_in , FORCE_z=False ):
    '''
        Logarithm of the likelihood between an observed spectrum and a model 
        configuration defined in theta

        **Input**

        theta : 1-D sequence of float
                Contains the parameters of the mode:
                    theta[0] = logarithim of the expansion velocity
                    theta[1] = logarithim of the neutral hydrogen column density
                    theta[2] = logarithim of the dust optical depth
                    theta[3] = redshift
                    theta[4] = logarithm of the intrinsic equivalent width
                    theta[5] = intrinsic width

        w_obs_Arr : 1-D sequence of float
                    wavelength where the observed density flux is evaluated.

        f_obs_Arr : 1-D sequence of float
                    Observed flux density

        s_obs_Arr : 1-D sequence of float
                    Uncertanty in the observed flux density.

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment.

        w_min : float
                minimum wavelength in the observed frame [A] to use. This matches the minimum
                wavelgnth of wave_pix_Arr (see below).

        w_max : float
                maximum  wavelength in the observed frame [A] to use. This might not be exactly
                the maximum wavelgnth of wave_pix_Arr (see below) due to pixelization.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        z_in : 1-D sequence of floats.
               Redshift range to be considered. In principle the redshift can be outside
               z_in[0] is the minimum redshift 
               z_in[1] is the maximum redshift 

        FORCE_z : optional bool
                  If True, force the redshift to be inside z_in 

        **Output**

        log_like : float
                   Logarithm of the likelihood
    '''

    if not Prior_f_5( theta ):
        return -np.inf

    log_V , log_N , log_t , redshift , log_EW , Wi = theta

    if not z_in is None and FORCE_z :
       if redshift < z_in[0] : return -np.inf 
       if redshift > z_in[1] : return -np.inf 

    V_f = 10 ** log_V
    t_f = 10 ** log_t

    FF = 1.

    w_model_Arr , f_model_Arr , _ = generate_a_REAL_line_Noise_w( redshift, V_f, log_N, t_f, FF , log_EW , Wi , w_obs_Arr , s_obs_Arr*0.0, FWHM, PIX, w_min, w_max, DATA_LyaRT, Geometry )

    f_model_Arr = f_model_Arr * 1. / np.amax( f_model_Arr ) 

    f_obs_Arr = f_obs_Arr * 1. / np.amax( f_obs_Arr )
    s_obs_Arr = s_obs_Arr * 1. / np.amax( f_obs_Arr )

    log_like = log_likelihood( w_obs_Arr , f_obs_Arr , s_obs_Arr , w_model_Arr , f_model_Arr )

    return log_like
#====================================================================#
#====================================================================#
#====================================================================#
def init_walkers_5( N_walkers , N_dim , log_V_in , log_N_in , log_t_in , z_in , log_E_in , W_in ):
    '''
        Creates the initial position for the walkers

        **Input**

        N_walkers : int
                    Number of walkers

        N_dim : int
                Number of dimensions (6)
        
        log_V_in : 1-D sequence of floats.
               Range of the logarithm of the bulk velocity
               log_V_in[0] is the minimum 
               log_V_in[1] is the maximum

        log_N_in : 1-D sequence of floats.
               Range of the logarithm of the neutral hydrogen column density
               log_N_in[0] is the minimum 
               log_N_in[1] is the maximum
 
        log_t_in : 1-D sequence of floats.
               Range of the logarithm of the dust optical depth
               log_t_in[0] is the minimum 
               log_t_in[1] is the maximum

        z_in : 1-D sequence of floats.
               Redshift range to be considered. 
               z_in[0] is the minimum redshift
               z_in[1] is the maximum redshift

        log_E_in : 1-D sequence of floats.
               Range of the logarithm of the intrinsic equivalent width
               log_E_in[0] is the minimum 
               log_E_in[1] is the maximum

        W_in : 1-D sequence of floats.
               Instrinsic line width range to be considered. 
               W_in[0] is the minimum redshift
               W_in[1] is the maximum redshift

        **Output**

        theta_0 : 1-D sequence of float
                Contains the parameters of the mode:
                    theta_0[:,0] = logarithim of the expansion velocity
                    theta_0[:,1] = logarithim of the neutral hydrogen column density
                    theta_0[:,2] = logarithim of the dust optical depth
                    theta_0[:,3] = redshift
                    theta_0[:,4] = logarithm of the intrinsic equivalent width
                    theta_0[:,5] = intrinsic width

    '''
    init_V_Arr = np.random.rand( N_walkers ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
    init_N_Arr = np.random.rand( N_walkers ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0]
    init_t_Arr = np.random.rand( N_walkers ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0]
    init_E_Arr = np.random.rand( N_walkers ) * ( log_E_in[1] - log_E_in[0] ) + log_E_in[0]
    init_z_Arr = np.random.rand( N_walkers ) * (     z_in[1] -     z_in[0] ) +     z_in[0]
    init_W_Arr = np.random.rand( N_walkers ) * (     W_in[1] -     W_in[0] ) +     W_in[0]

    Theta_0 = np.zeros( N_walkers * N_dim ).reshape( N_walkers , N_dim )

    Theta_0[ : , 0 ] = init_V_Arr
    Theta_0[ : , 1 ] = init_N_Arr
    Theta_0[ : , 2 ] = init_t_Arr
    Theta_0[ : , 3 ] = init_z_Arr
    Theta_0[ : , 4 ] = init_E_Arr
    Theta_0[ : , 5 ] = init_W_Arr

    for i in range( 0 , N_walkers ) :

        theta = Theta_0[i]

        while not Prior_f_5( theta ) :

            Theta_0[ i , 0 ] = np.random.rand( ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
            Theta_0[ i , 1 ] = np.random.rand( ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0] 
            Theta_0[ i , 2 ] = np.random.rand( ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0] 
            Theta_0[ i , 3 ] = np.random.rand( ) * (     z_in[1] -     z_in[0] ) +     z_in[0]
            Theta_0[ i , 4 ] = np.random.rand( ) * ( log_E_in[1] - log_E_in[0] ) + log_E_in[0] 
            Theta_0[ i , 5 ] = np.random.rand( ) * (     W_in[1] -     W_in[0] ) +     W_in[0] 

            theta = Theta_0[i]

    return Theta_0
#====================================================================#
#====================================================================#
#====================================================================#
def MCMC_get_region_6D( MODE , w_tar_Arr , f_tar_Arr , s_tar_Arr , FWHM , PIX , DATA_LyaRT , Geometry , Geometry_Mode='Outflow'):
    '''
        Computes the region of where the walkers are initialize

        **Input**

        MODE : string
               Method, DNN or PSO

        w_tar_Arr : 1-D sequence of floats
                    wavelength where the densit flux is evaluated

        f_tar_Arr : 1-D sequence of floats
                    Densit flux is evaluated

        s_tar_Arr : 1-D sequence of floats
                    Uncertainty of the densit flux is evaluated

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment.

        w_min : float
                minimum wavelength in the observed frame [A] to use. This matches the minimum
                wavelgnth of wave_pix_Arr (see below).

        w_max : float
                maximum  wavelength in the observed frame [A] to use. This might not be exactly
                the maximum wavelgnth of wave_pix_Arr (see below) due to pixelization.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.


        Geometry_Mode : optinal string
                        Changes from inflow ('Inflow') to outflow ('Outflow')
                        Default: 'Outflow'

        **Output**

        log_V_in : 1-D sequence of floats.
               Range of the logarithm of the bulk velocity
               log_V_in[0] is the minimum
               log_V_in[1] is the maximum

        log_N_in : 1-D sequence of floats.
               Range of the logarithm of the neutral hydrogen column density
               log_N_in[0] is the minimum
               log_N_in[1] is the maximum

        log_t_in : 1-D sequence of floats.
               Range of the logarithm of the dust optical depth
               log_t_in[0] is the minimum
               log_t_in[1] is the maximum

        z_in : 1-D sequence of floats.
               Redshift range to be considered.
               z_in[0] is the minimum redshift
               z_in[1] is the maximum redshift

        log_E_in : 1-D sequence of floats.
               Range of the logarithm of the intrinsic equivalent width
               log_E_in[0] is the minimum
               log_E_in[1] is the maximum

        W_in : 1-D sequence of floats.
               Instrinsic line width range to be considered.
               W_in[0] is the minimum redshift
               W_in[1] is the maximum redshift
    '''
    if MODE == 'PSO' :

        n_particles = 400
        n_iters     = 100

        cost , pos = PSO_Analysis( w_tar_Arr , f_tar_Arr , FWHM , PIX , DATA_LyaRT , Geometry , n_particles , n_iters )

        log_V_in = [ 0.999 * pos[1] , 1.001 * pos[1] ]
        log_N_in = [ 0.999 * pos[2] , 1.001 * pos[2] ]
        log_t_in = [ 0.999 * pos[3] , 1.001 * pos[3] ]
        log_E_in = [ 0.999 * pos[4] , 1.001 * pos[4] ]

        z_in = [ 0.999 *       pos[0] , 1.001 *       pos[0] ]
        W_in = [ 0.999 * 10 ** pos[5] , 1.001 * 10 ** pos[5] ]

        Best = [ pos[0] , pos[1] , pos[2] , pos[3] , pos[4] , np.log10( pos[5] ) ]

    if MODE == 'DNN':

        machine_data = Load_NN_model( Geometry_Mode )

        loaded_model = machine_data['Machine']

        w_rest_Machine_Arr = machine_data['w_rest']

        _ , _ , log_V_sol_Arr , log_N_sol_Arr , log_t_sol_Arr , z_sol_Arr , log_E_sol_Arr , log_W_sol_Arr = NN_measure( w_tar_Arr , f_tar_Arr , s_tar_Arr , FWHM , PIX , loaded_model , w_rest_Machine_Arr , N_iter=1000 )

        log_V_in = [ np.percentile( log_V_sol_Arr ,  5 ) , np.percentile( log_V_sol_Arr , 95 ) ]
        log_N_in = [ np.percentile( log_N_sol_Arr ,  5 ) , np.percentile( log_N_sol_Arr , 95 ) ]
        log_t_in = [ np.percentile( log_t_sol_Arr ,  5 ) , np.percentile( log_t_sol_Arr , 95 ) ]
        log_E_in = [ np.percentile( log_E_sol_Arr ,  5 ) , np.percentile( log_E_sol_Arr , 95 ) ]
        log_W_in = [ np.percentile( log_W_sol_Arr ,  5 ) , np.percentile( log_W_sol_Arr , 95 ) ]

        z_in = [ np.percentile( z_sol_Arr ,  5 ) , np.percentile( z_sol_Arr , 95 ) ]

        W_in = 10**np.array( log_W_in )

        Best = [ np.percentile( z_sol_Arr , 50 ) , np.percentile( log_V_sol_Arr , 50 ) ,
                                                   np.percentile( log_N_sol_Arr , 50 ) ,
                                                   np.percentile( log_t_sol_Arr , 50 ) ,
                                                   np.percentile( log_E_sol_Arr , 50 ) ,
                                                   np.percentile( log_W_sol_Arr , 50 ) ]

    if not MODE in [ 'PSO' , 'DNN' ] :

        log_V_in = None
        log_N_in = None
        log_t_in = None
        log_E_in = None
        W_in     = None

        w_f_max = w_tar_Arr[ f_tar_Arr==np.amax(f_tar_Arr) ]

        z_f_max = w_f_max / 1215.67 - 1.

        z_in = [ z_f_max*0.99 , z_f_max*1.01 ]

        Best = [ z_f_max , None , None , None , None , None ]

    return log_V_in , log_N_in , log_t_in , log_E_in , W_in , z_in , Best
#====================================================================#
#====================================================================#
#====================================================================#
def MCMC_Analysis_sampler_5( w_target_Arr , f_target_Arr , s_target_Arr , FWHM , N_walkers , N_burn , N_steps , Geometry , DATA_LyaRT , log_V_in=None , log_N_in=None , log_t_in=None , z_in=None , log_E_in=None , W_in=None , progress=True , FORCE_z=False ):

    '''
        Full MCMC anaylsis for the Thin_Shell_Cont

        **Input**

        w_tar_Arr : 1-D sequence of floats
                    wavelength where the densit flux is evaluated

        f_tar_Arr : 1-D sequence of floats
                    Densit flux is evaluated

        s_tar_Arr : 1-D sequence of floats
                    Uncertainty of the densit flux is evaluated

        FWHM : float
              Full width half maximum [A] of the experiment.

        N_walkers : int
                    Number of walkers

        N_dim : int
                Number of dimensions (6)

        N_burn : int
                 Number of steps in the burnin-in phase

        N_steps : int
                  Number of steps 

        Geometry : string
                   Outflow geometry to use.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry_Mode : optinal string
                        Changes from inflow ('Inflow') to outflow ('Outflow')
                        Default: 'Outflow'

        log_V_in : optional 1-D sequence of floats.
               Range of the logarithm of the bulk velocity
               log_V_in[0] is the minimum
               log_V_in[1] is the maximum

        log_N_in : optional 1-D sequence of floats.
               Range of the logarithm of the neutral hydrogen column density
               log_N_in[0] is the minimum
               log_N_in[1] is the maximum

        log_t_in : optional 1-D sequence of floats.
               Range of the logarithm of the dust optical depth
               log_t_in[0] is the minimum
               log_t_in[1] is the maximum

        z_in : optional 1-D sequence of floats.
               Redshift range to be considered.
               z_in[0] is the minimum redshift
               z_in[1] is the maximum redshift

        log_E_in : optional 1-D sequence of floats.
               Range of the logarithm of the intrinsic equivalent width
               log_E_in[0] is the minimum
               log_E_in[1] is the maximum

        W_in : optional 1-D sequence of floats.
               Instrinsic line width range to be considered.
               W_in[0] is the minimum redshift
               W_in[1] is the maximum redshift

        progress : optional bool
              If True shows the MCMC progress.
              Default True
        
        
        FORCE_z : optional bool
                  If True, force the redshift to be inside z_in

        **Output**

        samples : emcee python packge object.
                  Contains the information of the MCMC.

    '''
    N_dim = 6

    if log_V_in is None : log_V_in =  [  1.   ,  3.  ]
    if log_N_in is None : log_N_in =  [ 17.   , 22.  ]
    if log_t_in is None : log_t_in =  [ -0.2  , -3.  ]
    if log_E_in is None : log_E_in =  [  -1.  ,  3.  ]
    if     z_in is None :     z_in =  [  0.0  , 10.  ]
    if     W_in is None :     W_in =  [  0.01  , 6. ]

    Theta_0 = init_walkers_5( N_walkers , N_dim , log_V_in , log_N_in , log_t_in , z_in , log_E_in , W_in )

    PIX = w_target_Arr[1] - w_target_Arr[0]

    w_min = np.amin( w_target_Arr )
    w_max = np.amax( w_target_Arr )

    my_args = ( w_target_Arr , f_target_Arr , s_target_Arr , FWHM, PIX, w_min, w_max, DATA_LyaRT, Geometry , z_in , FORCE_z )

    if progress : print( 'defining samples' )

    sampler = emcee.EnsembleSampler( N_walkers , N_dim, log_likeliehood_of_model_5 , args=my_args )

    if progress : print( 'burning in' )

    state = sampler.run_mcmc( Theta_0 , N_burn , progress=progress )
    sampler.reset()

    if progress : print( 'Running main MCMC' )

    sampler.run_mcmc(state, N_steps , progress=progress )

    if progress : print( 'Done' )

    return sampler
#====================================================================#
#====================================================================#
#====================================================================#
def get_solutions_from_sampler( sampler , N_walkers , N_burn , N_steps , Q_Arr ):
    '''
        function to get the solution from the emcee sampler sin some given
        percentiles.

        **Input**

        sampler : emcee python packge object.
                  Contains the information of the MCMC.

        N_walkers : int
                    Number of walkers

        N_burn : int
                 Number of steps in the burnin-in phase

        N_steps : int
                  Number of steps

        Q_Arr : 1-D list of floats
                List of the percentiles that will be computed, for 
                example [ 16.0 , 50.0 , 84.0 ]

        **Output**

        matrix_sol: 2-D sequence of floats
                    The percentiles of each of the 6 properties.

        flat_samples : 2-D sequence of floats
                       The MCMC chains but flat.
    '''
    chains = sampler.get_chain()

    print( 'chains_shape = ' , chains.shape )

    log_probs = sampler.get_log_prob()

    print( 'log_probs_shape = ' , log_probs.shape )

    N_dim = len( chains[0,0] )

    flat_samples = np.zeros( N_walkers * N_steps * N_dim ).reshape( N_walkers * N_steps , N_dim )

    flat_log_prob = log_probs.ravel()

    for i in range( 0 , N_dim ):
    
        flat_samples[ : , i ] = chains[ : , : , i ].ravel()

    print( 'flat_samples shape = ' , flat_samples.shape )

    print( 'flat_log_prob shape = ' , flat_log_prob.shape )

    N_Q = len( Q_Arr )
    
    matrix_sol = np.zeros( N_dim * N_Q ).reshape( N_dim , N_Q )
    
    for i in range( 0 , N_dim ):
        for j in range( 0 , N_Q ):

            matrix_sol[i,j] = np.percentile( flat_samples[ : , i ] , Q_Arr[j] )

    return matrix_sol , flat_samples
#====================================================================#
#====================================================================#
#====================================================================#
def get_solutions_from_sampler_mean( sampler , N_walkers , N_burn , N_steps ):

    '''
        function to get the solution from the emcee sampler as the mean.

        **Input**

        sampler : emcee python packge object.
                  Contains the information of the MCMC.

        N_walkers : int
                    Number of walkers

        N_burn : int
                 Number of steps in the burnin-in phase

        N_steps : int
                  Number of steps

        **Output**

        matrix_sol: 1-D sequence of floats
                    Mean of each of the 6 properties.

        flat_samples : 2-D sequence of floats
                       The MCMC chains but flat.
    '''
    chains = sampler.get_chain()

    N_dim = len( chains[0,0] )

    flat_samples = np.zeros( N_walkers * N_steps * N_dim ).reshape( N_walkers * N_steps , N_dim )

    for i in range( 0 , N_dim ):
    
        flat_samples[ : , i ] = chains[ : , : , i ].ravel()

    #N_Q = len( Q_Arr )
    
    matrix_sol = np.zeros( N_dim )
    
    for i in range( 0 , N_dim ):
    
            matrix_sol[i] = np.mean( flat_samples[ : , i ] )

    return matrix_sol , flat_samples
#====================================================================#
#====================================================================#
#====================================================================#
def get_solutions_from_sampler_peak( sampler , N_walkers , N_burn , N_steps , N_hist_steps ):

    '''
        function to get the solution from the emcee sampler as the
        global maximum of the distribution of the posteriors.

        **Input**

        sampler : emcee python packge object.
                  Contains the information of the MCMC.

        N_walkers : int
                    Number of walkers

        N_burn : int
                 Number of steps in the burnin-in phase

        N_steps : int
                  Number of steps

        N_hist_steps : int 
                     Number of bins to sample the PDF of all properties
        **Output**

        matrix_sol: 1-D sequence of floats
                    Mean of each of the 6 properties.

        flat_samples : 2-D sequence of floats
                       The MCMC chains but flat.
    '''
    chains = sampler.get_chain()

    N_dim = len( chains[0,0] )

    flat_samples = np.zeros( N_walkers * N_steps * N_dim ).reshape( N_walkers * N_steps , N_dim )

    for i in range( 0 , N_dim ):

        flat_samples[ : , i ] = chains[ : , : , i ].ravel()

    #N_Q = len( Q_Arr )

    matrix_sol = np.zeros( N_dim )

    for i in range( 0 , N_dim ):

            #matrix_sol[i] = np.mean( flat_samples[ : , i ] )

            H_1 , edges_1 = np.histogram( flat_samples[ : , i ] , N_hist_steps ) 

            H_1 = H_1 * 1. / np.sum( H_1 )

            centers_1 = 0.5 * ( edges_1[:-1] + edges_1[1:] )

            tmp_edges = centers_1[ H_1 > 0.01 ]

            v_min = np.amin( tmp_edges )
            v_max = np.amin( tmp_edges )

            H_2 , edges_2 = np.histogram( flat_samples[ : , i ] , N_hist_steps , range=[ v_min , v_max ] )

            centers = 0.5 * ( edges_2[:-1] + edges_2[1:] )

            matrix_sol[i] = centers[ H_2==np.amax(H_2) ][0]

    return matrix_sol , flat_samples
#====================================================================#
#====================================================================#
#====================================================================#
def get_solutions_from_flat_chain( flat_chains , Q_Arr ):
    '''
        function to get the solution from the emcee sampler sin some given
        percentiles.

        **Input**

        flat_samples : 2-D sequence of floats
                       The MCMC chains but flat.

        Q_Arr : 1-D list of floats
                List of the percentiles that will be computed, for
                example [ 16.0 , 50.0 , 84.0 ]

        **Output**

        matrix_sol: 2-D sequence of floats
                    The percentiles of each of the 6 properties.

    '''
    N_dim = flat_chains.shape[1]

    N_Q = len( Q_Arr )

    matrix_sol = np.zeros( N_dim * N_Q ).reshape( N_dim , N_Q )

    for i in range( 0 , N_dim ):
        for j in range( 0 , N_Q ):

            matrix_sol[i,j] = np.percentile( flat_chains[ : , i ] , Q_Arr[j] )

    return matrix_sol 
#====================================================================#
#====================================================================#
#====================================================================#

# NN

#====================================================================#
#====================================================================#
#====================================================================#
def Load_NN_model( Mode , iteration=1 ):
    '''
        Loads the saved parameters of the deep neural networks

        **Input**

        Mode : string
               'Inflow' or 'Outflow'

        iteration : optional int
                    Number of the iteration. Currently only 1
                    Default 1

        **Output**

        machine_data : python dictionaty
                       Contains all the info for the DNN

    '''
    this_dir, this_filename = os.path.split(__file__) 

    if Mode == 'Inflow' : my_str = 'INFLOWS'
    if Mode == 'Outflow': my_str = 'OUTFLOW'

    machine_name = 'nn_'+my_str+'_N_2500000_Npix_1000_Dl_-18.5_18.5_Dc_True_nor_False_sca_True_256_256_256_256_256_it_'+str(iteration)+'.sav'
  
    extra_dir = '/DATA/'

    machine_data = pickle.load(open( this_dir + extra_dir +  machine_name , 'rb'))
 
    return machine_data
#====================================================================#
#====================================================================#
#====================================================================#
def NN_convert_Obs_Line_to_proxy_rest_line( w_obs_Arr , f_obs_Arr , s_obs_Arr=None , normed=False , scaled=True ):
    '''
        Converts an observed line profile to the rest frame of the maximum
        of the line profile.

        **Input**

        w_obs_Arr : 1-D sequence of floats
                    wavelength where the line profile is evaluated.

        f_obs_Arr : 1-D sequence of floats
                    Flux density of the observed line profile.

        s_obs_Arr : optional 1-D sequence of floats
                    Uncertainty in the flux density of the observed line profile.

        normed : optional bool
              If True, nomalizes the line profile.

        scaled : optinal bool
              If True, divides the line profile by its maximum.

        **Output**

        w_rest_Arr : 1-D sequence of floats
                    wavelength where the line profile is evaluated in the rest frame
                    of the global maximum

        Delta_rest_Arr : 1-D sequence of floats
                         w_rest_Arr - Lyman-alpha.

        f_rest_Arr : 1-D sequence of floats
                    Flux density in the rest frame of the global maximum

        z_max : float
                Redshift if the global maximum is the Lyaman-alpha wavelength

        if s_obs_Arr is not None it also returns:
        s_rest_Arr : 1-D sequence of floats
                    Uncertainty of the flux density in the rest frame of the global maximum
    '''
    w_Lya = 1215.67

    w_obs_max = w_obs_Arr[ f_obs_Arr == np.amax( f_obs_Arr ) ]

    z_max = w_obs_max * 1. / w_Lya - 1.

    ########
    z_max = np.atleast_1d( z_max )

    z_max = z_max[0]
    #######
    #print( 'z_max = ' , z_max )

    w_rest_Arr = w_obs_Arr * 1. / ( 1 + z_max )

    Delta_rest_Arr = w_rest_Arr - w_Lya

    if normed :

        II = np.trapz( f_obs_Arr , w_rest_Arr )

        f_rest_Arr = f_obs_Arr / II

        if not s_obs_Arr is None :
            s_rest_Arr = s_obs_Arr / II

    if scaled :

        f_max = np.amax( f_obs_Arr )

        f_rest_Arr = f_obs_Arr * 1. / f_max 

        if not s_obs_Arr is None :
            s_rest_Arr = s_obs_Arr * 1. / f_max

    if not scaled and not normed:

        f_rest_Arr = np.copy( f_obs_Arr )

        if not s_obs_Arr is None :
            s_rest_Arr = np.copy( s_obs_Arr )

    if s_obs_Arr is None :
        return w_rest_Arr , Delta_rest_Arr , f_rest_Arr , z_max

    if not s_obs_Arr is None :
        return w_rest_Arr , Delta_rest_Arr , f_rest_Arr , z_max , s_rest_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def NN_generate_random_outflow_props( N_walkers , log_V_in , log_N_in , log_t_in , Allow_Inflows=True ):

    '''
        Generates random poperties for the Thin_Shell, Galactic_Wind, etc. 
        (Not for Thin_Shell_Cont)        

        **Input**

        N_walkers : int
                    Number of walkers

        log_V_in : optional 1-D sequence of floats.
               Range of the logarithm of the bulk velocity
               log_V_in[0] is the minimum
               log_V_in[1] is the maximum

        log_N_in : optional 1-D sequence of floats.
               Range of the logarithm of the neutral hydrogen column density
               log_N_in[0] is the minimum
               log_N_in[1] is the maximum

        log_t_in : optional 1-D sequence of floats.
               Range of the logarithm of the dust optical depth
               log_t_in[0] is the minimum
               log_t_in[1] is the maximum

        Allow_Inflows : optional Bool
               If True it also return negative values of V in the same range.
               Default True           
 
        **Output**

        init_V_Arr : 1-D sequence of floats
                     Expansion velocity

        init_log_N_Arr : 1-D sequence of floats
                     Logarithms of the column density  

        init_log_t_Arr : 1-D sequence of floats
                     Logarithms of the dust optical depth


    '''
    init_log_V_Arr = np.random.rand( N_walkers ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
    init_log_N_Arr = np.random.rand( N_walkers ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0]
    init_log_t_Arr = np.random.rand( N_walkers ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0]

    for i in range( 0 , N_walkers ) :

        theta = [ init_log_V_Arr[i] , init_log_N_Arr[i] , init_log_t_Arr[i] , 1.0 ]

        while not Prior_f( theta ) :

            init_log_V_Arr[ i ] = np.random.rand( ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
            init_log_N_Arr[ i ] = np.random.rand( ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0]
            init_log_t_Arr[ i ] = np.random.rand( ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0]

            theta = [ init_log_V_Arr[i] , init_log_N_Arr[i] , init_log_t_Arr[i] , 1.0 ]

    if Allow_Inflows:

        V_sign = np.sign( np.random.rand( N_walkers ) - 0.5 )

    else: 
        V_sign = np.ones( N_walkers )

    init_V_Arr = 10**init_log_V_Arr * V_sign 

    return init_V_Arr , init_log_N_Arr , init_log_t_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def NN_generate_random_outflow_props_5D( N_walkers , log_V_in , log_N_in , log_t_in , log_E_in , log_W_in , MODE='Outflow' ):
    '''
        Generates random poperties for the Thin_Shell_Cont

        **Input**

        N_walkers : int
                    Number of walkers

        log_V_in : optional 1-D sequence of floats.
               Range of the logarithm of the bulk velocity
               log_V_in[0] is the minimum
               log_V_in[1] is the maximum

        log_N_in : optional 1-D sequence of floats.
               Range of the logarithm of the neutral hydrogen column density
               log_N_in[0] is the minimum
               log_N_in[1] is the maximum

        log_t_in : optional 1-D sequence of floats.
               Range of the logarithm of the dust optical depth
               log_t_in[0] is the minimum
               log_t_in[1] is the maximum

        log_E_in : optional 1-D sequence of floats.
               Range of the logarithm of the instrinsic equivalent width
               log_E_in[0] is the minimum
               log_E_in[1] is the maximum

        log_W_in : optional 1-D sequence of floats.
               Range of the logarithm of the intrinsic width of the line
               log_W_in[0] is the minimum
               log_W_in[1] is the maximum

        MODE : optional string
               'Outflow' for outflows 'Inflow' for inflows

        **Output**

        init_V_Arr : 1-D sequence of floats
                     Expansion velocity

        init_log_N_Arr : 1-D sequence of floats
                     Logarithms of the column density

        init_log_t_Arr : 1-D sequence of floats
                     Logarithms of the dust optical depth

        init_log_E_Arr : 1-D sequence of floats
                     Logarithms of the instrinsic equivalent width

        init_log_W_Arr : 1-D sequence of floats
                     Logarithms of the intrinsic width of the line
    '''

    if MODE not in [ 'Outflow' , 'Inflow' , 'Mixture' ]:

        print( '    -> Wrong MODE when generating gas properties, using only outflows' )

    init_log_V_Arr = np.random.rand( N_walkers ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
    init_log_N_Arr = np.random.rand( N_walkers ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0]
    init_log_t_Arr = np.random.rand( N_walkers ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0]
    init_log_E_Arr = np.random.rand( N_walkers ) * ( log_E_in[1] - log_E_in[0] ) + log_E_in[0]
    init_log_W_Arr = np.random.rand( N_walkers ) * ( log_W_in[1] - log_W_in[0] ) + log_t_in[0]

    for i in range( 0 , N_walkers ) :

        theta = [ init_log_V_Arr[i] , init_log_N_Arr[i] , init_log_t_Arr[i] , init_log_E_Arr[i] , init_log_W_Arr[i] , 1.0 ]

        while not Prior_f_5( theta ) :

            init_log_V_Arr[ i ] = np.random.rand( ) * ( log_V_in[1] - log_V_in[0] ) + log_V_in[0]
            init_log_N_Arr[ i ] = np.random.rand( ) * ( log_N_in[1] - log_N_in[0] ) + log_N_in[0]
            init_log_t_Arr[ i ] = np.random.rand( ) * ( log_t_in[1] - log_t_in[0] ) + log_t_in[0]
            init_log_E_Arr[ i ] = np.random.rand( ) * ( log_E_in[1] - log_E_in[0] ) + log_E_in[0]
            init_log_W_Arr[ i ] = np.random.rand( ) * ( log_W_in[1] - log_W_in[0] ) + log_W_in[0]

            theta = [ init_log_V_Arr[i] , init_log_N_Arr[i] , init_log_t_Arr[i] , init_log_E_Arr[i] , init_log_W_Arr[i] , 1.0 ]

    if MODE == 'Mixture' :

        V_sign = np.sign( np.random.rand( N_walkers ) - 0.5 )

    if MODE == 'Inflow':

        V_sign = -1. * np.ones( N_walkers ) 

    if MODE == 'Outflow':

        V_sign =       np.ones( N_walkers ) 


    init_V_Arr = 10**init_log_V_Arr * V_sign 

    return init_V_Arr , init_log_N_Arr , init_log_t_Arr , init_log_E_Arr , init_log_W_Arr 
#====================================================================#
#====================================================================#
#====================================================================#
def NN_measure( w_tar_Arr , f_tar_Arr , s_tar_Arr , FWHM_tar , PIX_tar , loaded_model , w_rest_Machine_Arr , N_iter=None , normed=False , scaled=True, Delta_min=-18.5 , Delta_max=18.5 , Nbins_tot=1000 , Denser_Center=True ):
    '''
        Generates random poperties for the Thin_Shell_Cont

        **Input**

        w_tar_Arr : 1-D sequence of floats
                    wavelength where the densit flux is evaluated

        f_tar_Arr : 1-D sequence of floats
                    Densit flux is evaluated

        s_tar_Arr : 1-D sequence of floats
                    Uncertainty of the densit flux is evaluated

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX_tar : float
                  Pixelization of the line profiles due to the experiment [A]

        loaded_model : python dictionaty
                       Contains all the info for the DNN form Load_NN_model()

        w_rest_Machine_Arr : 1-D sequence of floats
                             wavelength used by the INPUT of the DNN       
                           
        N_iter : optional int
                 Number of Monte Carlo iterations of the observed espectrum.
                 If None, no iteration is done.
                 Default None
 
        Delta_min : optional float
              Defines the minimum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = -18.5

        Delta_min : optional float
              Defines the maximum rest frame wavelegnth with respecto to Lyman-alpha.

              Default = +18.5

        Nbins_tot : optional int
              Total number of wvelgnths bins.

              Default = 1000

        Denser_Center : optional bool
              Populates denser the regions close to Lyman-alpha

              Default = True

        normed : optional bool
              If True, nomalizes the line profile.

        scaled : optinal bool
              If True, divides the line profile by its maximum. 

        **Output**

        if N_iter is None: 
            Sol : 1-D sequence of float
                  Array with the solution of the observed spectrum. No Monte Carlo perturbation.

            z_Sol : float
                    Best resdhift

        if N_iter is a float:
            Sol and z_sol and

            log_V_sol_2_Arr : 1-D sequence
                              Logarithm of the expasion velocity

            log_N_sol_2_Arr : 1-D sequence
                              Logarithm of the neutral hydrogen column density

            log_t_sol_2_Arr : 1-D sequence
                              Logarithm of the dust optical depth

            z_sol_2_Arr : 1-D sequence
                              redshift

            log_E_sol_2_Arr : 1-D sequence
                              Logarithm of the intrinsic equivalent width

            log_W_sol_2_Arr : 1-D sequence
                              Logarithm of the instrinsic width of the line


    '''
    w_rest_tar_Arr , f_rest_tar_Arr , z_max_tar , INPUT = Treat_A_Line_To_NN_Input( w_tar_Arr , f_tar_Arr , PIX_tar , FWHM_tar , Delta_min=Delta_min , Delta_max=Delta_max , Nbins_tot=Nbins_tot , Denser_Center=Denser_Center , normed=normed, scaled=scaled )

    assert np.sum( w_rest_tar_Arr - w_rest_Machine_Arr) == 0 , 'wavelength array of machine and measure dont match. Check that Delta_min, Delta_max, Nbins_tot and Denser_Center are the same in the model and the imput here.'

    Sol = loaded_model.predict( INPUT )

    w_Lya = 1215.673123 #A

    z_sol = ( w_Lya + Sol[0,0] ) * ( 1 + z_max_tar ) * 1. / ( w_Lya ) - 1.

    if N_iter is None :
        return Sol , z_sol

    if N_iter > 0 :
        log_V_sol_2_Arr = np.zeros( N_iter )
        log_N_sol_2_Arr = np.zeros( N_iter )
        log_t_sol_2_Arr = np.zeros( N_iter )
        log_E_sol_2_Arr = np.zeros( N_iter )
        log_W_sol_2_Arr = np.zeros( N_iter )

        z_sol_2_Arr = np.zeros( N_iter )

        for i in range( 0 , N_iter ) :

            f_obs_i_Arr = f_tar_Arr + np.random.randn( len( f_tar_Arr ) ) * s_tar_Arr

            w_rest_i_Arr , f_rest_i_Arr , z_max_i , INPUT_i = Treat_A_Line_To_NN_Input( w_tar_Arr , f_obs_i_Arr , PIX_tar , FWHM_tar , Delta_min=Delta_min , Delta_max=Delta_max , Nbins_tot=Nbins_tot , Denser_Center=Denser_Center , normed=normed, scaled=scaled )

            Sol_i = loaded_model.predict( INPUT_i )

            z_sol_i = ( w_Lya + Sol_i[0,0] ) * ( 1 + z_max_i ) * 1. / ( w_Lya ) - 1.

            log_V_sol_2_Arr[i] = Sol_i[0,1]
            log_N_sol_2_Arr[i] = Sol_i[0,2]
            log_t_sol_2_Arr[i] = Sol_i[0,3]
            log_E_sol_2_Arr[i] = Sol_i[0,4]
            log_W_sol_2_Arr[i] = Sol_i[0,5]

            z_sol_2_Arr[i] = z_sol_i

        return Sol , z_sol , log_V_sol_2_Arr , log_N_sol_2_Arr , log_t_sol_2_Arr , z_sol_2_Arr , log_E_sol_2_Arr , log_W_sol_2_Arr
#====================================================================#
#====================================================================#
#====================================================================#
# PSO
#====================================================================#
#====================================================================#
#====================================================================#
def PSO_compute_xi_2_ONE_6D( x , w_tar_Arr , f_tar_Arr , FWHM , PIX , DATA_LyaRT, Geometry ):
    '''
        Compute the chi esquare for the PSO analysis

        **Input**

        x: 1-D sequence of float
                Contains the parameters of the mode:
                    x[0] = logarithim of the expansion velocity
                    x[1] = logarithim of the neutral hydrogen column density
                    x[2] = logarithim of the dust optical depth
                    x[3] = redshift
                    x[4] = logarithm of the intrinsic equivalent width
                    x[5] = intrinsic width

        w_tar_Arr : 1-D sequence of float
                    wavelength where the observed density flux is evaluated.

        f_tar_Arr : 1-D sequence of float
                    Observed flux density

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        **Output**

        xi_2 : float
               Chi square of the configuration

        w_pso_Arr : 1-D sequence of floats
                    Wavelgnth of the line profile computed by the PSO 

        my_f_pso_Arr : 1-D sequence of floats
                       Flux density of the line profile computed by the PSO 

    '''
    my_f_tar_Arr = ( f_tar_Arr - np.amin( f_tar_Arr ) ) * 1. / np.amax( f_tar_Arr )

    w_min = np.amin( w_tar_Arr )
    w_max = np.amax( w_tar_Arr ) + 0.000001

    w_s_tar_Arr = np.linspace( w_min , w_max , 100 )
    s_tar_Arr = np.zeros( len( w_s_tar_Arr ))

    redshift  = x[0]
    log_V_pso = x[1]
    log_N_pso = x[2]
    log_t_pso = x[3]
    log_E_pso = x[4]
    log_W_pso = x[5]    

    F_pso = 1.0

    w_pso_Arr , f_pso_Arr , dic = generate_a_REAL_line_Noise_w( redshift      , 10**log_V_pso, log_N_pso  , 
                                                                10**log_t_pso , F_pso        , log_E_pso  , 
                                                                10**log_W_pso , w_s_tar_Arr ,
                                                                 s_tar_Arr    , FWHM         , PIX        , 
                                                                 w_min        , w_max        , DATA_LyaRT , 
                                                                 Geometry     )

    my_f_pso_Arr = f_pso_Arr * 1. / np.amax( f_pso_Arr)

    my_f_pso_Arr = np.interp( w_tar_Arr , w_pso_Arr , my_f_pso_Arr )

    my_f_tar_Arr = my_f_tar_Arr * 1. / np.amax( my_f_tar_Arr )

    xi_2 = np.sum( ( my_f_pso_Arr - my_f_tar_Arr ) ** 2 )

    if xi_2 == np.nan :
        print( 'found np.nan in xi_2!!!' )
        xi_2 = np.inf

    return xi_2 , w_pso_Arr , my_f_pso_Arr
#====================================================================#
#====================================================================#
#====================================================================#
def PSO_compute_xi_2_MANY( X , w_tar_Arr , f_tar_Arr , FWHM , PIX , DATA_LyaRT, Geometry ):
    '''
        Compute the chi esquare for the PSO analysis for many configurations

        **Input**

        x: 1-D sequence of float
                Contains the parameters of the mode:
                    x[0] = logarithim of the expansion velocity
                    x[1] = logarithim of the neutral hydrogen column density
                    x[2] = logarithim of the dust optical depth
                    x[3] = redshift
                    x[4] = logarithm of the intrinsic equivalent width
                    x[5] = intrinsic width

        w_tar_Arr : 1-D sequence of float
                    wavelength where the observed density flux is evaluated.

        f_tar_Arr : 1-D sequence of float
                    Observed flux density

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        **Output**

        xi_2_Arr : 1-D sequence of float
               Chi square of the configurations
    '''
    xi_2_Arr = np.zeros( len(X) )

    for i in range( 0 , len(X) ):

        xi_2 , w_pso_Arr , f_pso_Arr = PSO_compute_xi_2_ONE_6D( X[i] , w_tar_Arr , f_tar_Arr , FWHM , PIX , DATA_LyaRT, Geometry )

        xi_2_Arr[i] = xi_2

    return xi_2_Arr
#====================================================================#
#====================================================================#
#====================================================================#
##====================================================================================#
def PSO_Analysis( w_tar_Arr , f_tar_Arr , FWHM , PIX , DATA_LyaRT , Geometry , n_particles , n_iters ):

    '''
        Does a PSO analysis to find in a fast way a close good fit

        **Input**

        w_tar_Arr : 1-D sequence of float
                    wavelength where the observed density flux is evaluated.

        f_tar_Arr : 1-D sequence of float
                    Observed flux density

        FWHM : float
              Full width half maximum [A] of the experiment.

        PIX : float
              Pixel size in wavelgnth [A] of the experiment.

        DATA_LyaRT : python dictionary
                     Contains the grid information.

        Geometry : string
                   Outflow geometry to use.

        n_particles : int
                      Number of particles in the PSO

        n_iters : int
                  Number of steps in the PSO

        **Output**

        cost : float
               Cost of the best configuration

        pos : 1-D sequence of floats
                Position of the best configuration
    '''
    w_lya = 1215.67

    #w_max = np.atleast_1d( w_tar_Arr[ f_tar_Arr == np.amax( f_tar_Arr ) ])[0]

    print( 'max = ' , np.amax( f_tar_Arr ) )

    print( f_tar_Arr == np.amax( f_tar_Arr ) )

    print( sum( f_tar_Arr == np.amax( f_tar_Arr ) ) )

    w_max = w_tar_Arr[ np.where( f_tar_Arr == np.amax( f_tar_Arr ) ) ][0]

    print( w_max )

    z_of_the_max = w_max / w_lya - 1.

    print( z_of_the_max )

    Dz = ( 1 + z_of_the_max ) * 2e-3

    pso_z_min = z_of_the_max - Dz
    pso_z_max = z_of_the_max + Dz

    if pso_z_min < 0 : pso_z_min = 1e-10

    X_min = [ pso_z_min , 1.0 , 17. , -4.00 , 0.1 , 0.01 ]
    X_max = [ pso_z_max , 3.0 , 22. , -0.25 , 3.  , 6.0  ]

    bounds = (X_min, X_max)

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    n_particles = n_particles
    dimensions  = 6

    optimizer = GlobalBestPSO( n_particles=n_particles , dimensions=dimensions , options=options, bounds=bounds)

    cost, pos = optimizer.optimize( PSO_compute_xi_2_MANY , iters=n_iters , w_tar_Arr=w_tar_Arr , f_tar_Arr=f_tar_Arr , FWHM=FWHM , PIX=PIX , DATA_LyaRT=DATA_LyaRT , Geometry=Geometry )

    return cost , pos
#====================================================================#
#====================================================================#
#====================================================================#

# Enjoy

##====================================================================================#
if __name__ == '__main__':
    pass
##====================================================================================#














