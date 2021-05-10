import numpy as np

import Lya_zelda as Lya

import sys

import pylab as plt

your_grids_location = '/Users/sidgurung/Desktop/ZELDA/Grids/'

Lya.funcs.Data_location = your_grids_location

Geometry = 'Thin_Shell_Cont'

MODE = 'FULL'

LyaRT_Grid = Lya.load_Grid_Line( Geometry , MODE=MODE )

print( LyaRT_Grid.keys() )

z_t      = 0.5   # redshift of the source 
V_t      = 50.0  # Outfloe expansion velocity [km/s]
log_N_t  = 20.   # Logarithmic of the neutral hydrogen column density [cm**-2]
t_t      = 0.01  # Dust optical depth
log_EW_t = 1.5   # Logarithmic the intrinsic equivalent width [A]
W_t      = 0.5   # Intrinsic width of the line [A]
F_t      = 1.    # Total flux of the line

PNR_t  = 15.0 # Signal to noise ratio of the maximum of the line.
FWHM_t = 0.2  # Full width half maximum diluting the line. Mimics finite resolution. [A]
PIX_t  = 0.1  # Wavelength binning of the line. [A]

########################################
########################################
########################################
w_Arr , f_Arr , s_Arr  = Lya.Generate_a_real_line( z_t , V_t, log_N_t, t_t, F_t, log_EW_t, W_t , PNR_t, FWHM_t, PIX_t, LyaRT_Grid, Geometry )

w_pix_Arr , f_pix_Arr = Lya.plot_a_rebinned_line( w_Arr , f_Arr , PIX_t )

plt.plot( w_pix_Arr , f_pix_Arr ) 

plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_3_1.png' )
plt.clf()
########################################
########################################
########################################


machine_data =  Lya.Load_NN_model( 'Outflow' )

print(  machine_data.keys() )

machine = machine_data['Machine'] 
w_rest_Arr = machine_data[ 'w_rest' ]

Sol , z_sol = Lya.NN_measure( w_Arr , f_Arr , s_Arr , FWHM_t , PIX_t , machine , w_rest_Arr , N_iter=None )

print( 'The measured redshift                                                     is' , z_sol    )
print( 'The measured logarithm of the expasion velocity                           is' , Sol[0,1] )
print( 'The measured logarithm of the HI column density                           is' , Sol[0,2] )
print( 'The measured logarithm of the dust optical depth                          is' , Sol[0,3] )
print( 'The measured logarithm of the intrinsic equivalent width                  is' , Sol[0,4] )
print( 'The measured logarithm of the intrinsic            width                  is' , Sol[0,5] )
print( 'The measured shift of the true Lya wavelgnth from the maximum of the line is' , Sol[0,0] )

PNR = 100000.

V_sol    = 10**Sol[0,1]
logN_sol =     Sol[0,2]
t_sol    = 10**Sol[0,3]
logE_sol =     Sol[0,4]
W_sol    = 10**Sol[0,5]

########################################
########################################
########################################
w_One_Arr , f_One_Arr , _  = Lya.Generate_a_real_line( z_sol , V_sol, logN_sol, t_sol, F_t, logE_sol, W_sol, PNR, FWHM_t, PIX_t, LyaRT_Grid, Geometry )

w_pix_One_Arr , f_pix_One_Arr = Lya.plot_a_rebinned_line( w_One_Arr , f_One_Arr , PIX_t )

plt.plot( w_pix_Arr     , f_pix_Arr     , label='Target' )
plt.plot( w_pix_One_Arr , f_pix_One_Arr , label='1 iter' )

plt.legend(loc=0)
plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_3_2.png' )
plt.clf()
########################################
########################################
########################################

Sol , z_sol , log_V_Arr , log_N_Arr , log_t_Arr , z_Arr , log_E_Arr , log_W_Arr = Lya.NN_measure( w_Arr , f_Arr , s_Arr , FWHM_t , PIX_t , machine     , w_rest_Arr , N_iter=1000 )

z_50     = np.percentile(    z_Arr , 50 )
z_16     = np.percentile(    z_Arr , 16 )
z_84     = np.percentile(    z_Arr , 84 )

log_N_50 = np.percentile( log_N_Arr , 50 )
log_N_16 = np.percentile( log_N_Arr , 16 )
log_N_84 = np.percentile( log_N_Arr , 84 )

log_E_50 = np.percentile( log_E_Arr , 50 )
log_E_16 = np.percentile( log_E_Arr , 16 )
log_E_84 = np.percentile( log_E_Arr , 84 )

V_50     = 10 ** np.percentile( log_V_Arr , 50 )
V_16     = 10 ** np.percentile( log_V_Arr , 16 )
V_84     = 10 ** np.percentile( log_V_Arr , 84 )

t_50     = 10 ** np.percentile( log_t_Arr , 50 )
t_16     = 10 ** np.percentile( log_t_Arr , 16 )
t_84     = 10 ** np.percentile( log_t_Arr , 84 )

W_50     = 10 ** np.percentile( log_W_Arr , 50 )
W_16     = 10 ** np.percentile( log_W_Arr , 16 )
W_84     = 10 ** np.percentile( log_W_Arr , 84 )


########################################
########################################
########################################
w_50th_Arr , f_50th_Arr , _  = Lya.Generate_a_real_line( z_50 , V_50, log_N_50, t_50, F_t, log_E_50, W_50, PNR, FWHM_t, PIX_t, LyaRT_Grid, Geometry )

w_pix_50th_Arr , f_pix_50th_Arr = Lya.plot_a_rebinned_line( w_50th_Arr , f_50th_Arr , PIX_t )

plt.plot( w_pix_Arr      , f_pix_Arr      , label='Target'   )
plt.plot( w_pix_One_Arr  , f_pix_One_Arr  , label='1 iter'   )
plt.plot( w_pix_50th_Arr , f_pix_50th_Arr , label='1000 iter')

plt.legend(loc=0)
plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_3_3.png' )
plt.clf()
########################################
########################################
########################################

print( 'The true redshift                 is' , z_t      , 'and the predicted is' , z_50     , '(-' , z_50-z_16         , ', +' , z_84-z_50         , ')' )
print( 'The true expansion velocity       is' , V_t      , 'and the predicted is' , V_50     , '(-' , V_50-V_16         , ', +' , V_84-V_50         , ')' )
print( 'The true dust optical depth       is' , t_t      , 'and the predicted is' , t_50     , '(-' , t_50-t_16         , ', +' , t_84-t_50         , ')' )
print( 'The true intrinsic width          is' , W_t      , 'and the predicted is' , W_50     , '(-' , W_50-W_16         , ', +' , W_84-W_50         , ')' )
print( 'The true log of HI column density is' , log_N_t  , 'and the predicted is' , log_N_50 , '(-' , log_N_50-log_N_16 , ', +' , log_N_84-log_N_50 , ')' )
print( 'The true log of equivalent width  is' , log_EW_t , 'and the predicted is' , log_E_50 , '(-' , log_E_50-log_E_16 , ', +' , log_E_84-log_E_50 , ')' )
 






