import numpy as np

import Lya_zelda as Lya

import sys

import pylab as plt

######################################################################
######################################################################
######################################################################

#your_grids_location = '/Users/sidgurung/Desktop/ZELDA/Grids/'
your_grids_location = '/global/users/sidgurung/PROMARE/Grids/'

Lya.funcs.Data_location = your_grids_location

Geometry = 'Thin_Shell_Cont'

LyaRT_Grid = Lya.load_Grid_Line( Geometry )

log_V_in = [  1.0   ,  3.0   ]
log_N_in = [ 17.0   , 21.5   ]
log_t_in = [ -4.0   , 0.0  ]
log_E_in = [ 0.0    , 2.3  ]
log_W_in = [ -2.    , 0.7  ]

z_in = [ 0.0001 , 4.00 ]

log_FWHM_in = [ -1.  ,   0.3  ]
log_PIX_in  = [ -1.3 ,   0.3  ]

log_PNR_in = [ 0.7 , 1.6 ]

N_train = 1000

V_Arr , log_N_Arr , log_t_Arr , log_E_Arr , log_W_Arr = Lya.NN_generate_random_outflow_props_5D( N_train , log_V_in , log_N_in , log_t_in , log_E_in , log_W_in , MODE=MODE )

z_Arr = np.random.rand( N_train ) * ( z_in[1] - z_in[0] ) + z_in[0]

log_FWHM_Arr = np.random.rand( N_train ) * ( log_FWHM_in[1] - log_FWHM_in[0] ) + log_FWHM_in[0]
log_PIX_Arr  = np.random.rand( N_train ) * (  log_PIX_in[1] -  log_PIX_in[0] ) +  log_PIX_in[0]
log_PNR_Arr  = np.random.rand( N_train ) * (  log_PNR_in[1] -  log_PNR_in[0] ) +  log_PNR_in[0]

F_t = 1.0

Delta_True_Lya_Arr = np.zeros( N_train )

z_PEAK_Arr = np.zeros( N_train )

LINES_train = np.zeros( N_train * N_bins ).reshape( N_train , N_bins )

N_bins_input = 1000 + 3

INPUT_train = np.zeros( N_train * N_bins_input ).reshape( N_train , N_bins_input )

print( 'Generating training set' )

cc = 0.0
for i in range( 0, N_train ):

    per = 100. * i / N_train

    if per >= cc :
        print( cc , '%' )
        cc += 1.0

    V_t = V_Arr[i]
    t_t = 10**log_t_Arr[i]

    log_N_t = log_N_Arr[i]

    log_E_t = log_E_Arr[i]

    W_t = 10**log_W_Arr[i]

    z_t = z_Arr[i]

    FWHM_t = 10**log_FWHM_Arr[ i ]
    PIX_t  = 10**log_PIX_Arr[  i ]

    PNR_t = 10**log_PNR_Arr[i]

    rest_w_Arr , train_line , z_max_i , input_i = Lya.Generate_a_line_for_training( z_t , V_t, log_N_t, t_t, F_t, log_E_t, W_t , PNR_t, FWHM_t, PIX_t, DATA_LyaRT, Geometry)

    z_PEAK_Arr[i] = z_max_i

    Delta_True_Lya_Arr[ i ] = 1215.67 * ( (1+z_t)/(1+z_max_i) - 1. )

    LINES_train[i] = train_line

    INPUT_train[i] = input_i


dic = {}

dic[ 'lines' ] = LINES_train

dic[ 'NN_input' ] = INPUT_train

dic['z_PEAK'         ] = z_PEAK_Arr
dic['z'              ] = z_Arr

dic['Delta_True_Lya'] = Delta_True_Lya_Arr
dic['V'             ] = V_Arr
dic['log_N'         ] = log_N_Arr
dic['log_t'         ] = log_t_Arr
dic['log_PNR'       ] = log_PNR_Arr
dic['log_W'         ] = log_W_Arr
dic['log_EW'        ] = log_E_Arr
dic['log_PIX'       ] = log_PIX_Arr
dic['log_FWHM'      ] = log_FWHM_Arr

dic['rest_w'] = rest_w_Arr

np.save( 'data_for_training.npy' , dic )






























