import numpy as np

import Lya_zelda as Lya

import sys

import pylab as plt


######################################################################
######################################################################
######################################################################
def make_corner_plots( my_chains_matrix ):

    N_dim = 6

    ax_list = []

    label_list = [ 'log V' , 'log N' , 'log ta' , 'z' , 'log EW', 'Wi'  ]

    MAIN_VALUE_mean   = np.zeros(N_dim)
    MAIN_VALUE_median = np.zeros(N_dim)
    MAIN_VALUE_MAX    = np.zeros(N_dim)

    for i in range( 0 , N_dim ):

        x_prop = my_chains_matrix[ : , i ]

        x_prop_min = np.percentile( x_prop , 10 )
        x_prop_50  = np.percentile( x_prop , 50 )
        x_prop_max = np.percentile( x_prop , 90 )

        x_min = x_prop_50 - ( x_prop_max - x_prop_min ) * 1.00
        x_max = x_prop_50 + ( x_prop_max - x_prop_min ) * 1.00

        mamamask = ( x_prop > x_min ) * ( x_prop < x_max )

        MAIN_VALUE_mean[  i] = np.mean(       x_prop[ mamamask ] )
        MAIN_VALUE_median[i] = np.percentile( x_prop[ mamamask ] , 50 )

        HH , edges_HH = np.histogram( x_prop[ mamamask ] , 30 , range=[ x_prop_min , x_prop_max ] )

    plt.figure( figsize=(15,15) )

    Q_top = 80
    Q_low = 20

    for i in range( 0 , N_dim ):

        y_prop = my_chains_matrix[ : , i ]

        y_prop_min = np.percentile( y_prop , Q_low )
        y_prop_50  = np.percentile( y_prop , 50 )
        y_prop_max = np.percentile( y_prop , Q_top  )

        mask_y = ( y_prop > y_prop_min ) * ( y_prop < y_prop_max )

        y_min = y_prop_50 - np.std( y_prop[ mask_y ] )
        y_max = y_prop_50 + np.std( y_prop[ mask_y ] )

        for j in range( 0 , N_dim ):

            if i < j : continue

            x_prop = my_chains_matrix[ : , j ]

            x_prop_min = np.percentile( x_prop , Q_low )
            x_prop_50  = np.percentile( x_prop , 50 )
            x_prop_max = np.percentile( x_prop , Q_top )

            mask_x = ( x_prop > x_prop_min ) * ( x_prop < x_prop_max )

            x_min = x_prop_50 - np.std( x_prop[ mask_x ] )
            x_max = x_prop_50 + np.std( x_prop[ mask_x ] )

            ax = plt.subplot2grid( ( N_dim , N_dim ) , (i, j)  )

            ax_list += [ ax ]

            DDX = x_max - x_min
            DDY = y_max - y_min

            if i==j :

                H , edges = np.histogram( x_prop , 30 , range=[x_min,x_max] )

                ax.hist( x_prop , 30 , range=[x_min,x_max] , color='cornflowerblue' )

                ax.plot( [ MAIN_VALUE_median[i] , MAIN_VALUE_median[i] ] , [ 0.0 , 1e10 ] , 'k--' , lw=2 )

                ax.set_ylim( 0 , 1.1 * np.amax(H) )

            else :

                XX_min = x_min - DDX * 0.2
                XX_max = x_max + DDX * 0.2

                YY_min = y_min - DDY * 0.2
                YY_max = y_max + DDY * 0.2

                H , edges_y , edges_x = np.histogram2d( x_prop , y_prop , 30 , range=[[XX_min , XX_max],[YY_min , YY_max]] )

                y_centers = 0.5 * ( edges_y[1:] + edges_y[:-1] )
                x_centers = 0.5 * ( edges_x[1:] + edges_x[:-1] )

                H_min = np.amin( H )
                H_max = np.amax( H )

                N_bins = 10000

                H_Arr = np.linspace( H_min , H_max , N_bins )[::-1]

                fact_up_Arr = np.zeros( N_bins )

                TOTAL_H = np.sum( H )

                for iii in range( 0 , N_bins ):

                    mask = H > H_Arr[iii]

                    fact_up_Arr[iii] = np.sum( H[ mask ] ) / TOTAL_H

                H_value_68 = np.interp( 0.680 , fact_up_Arr , H_Arr )
                H_value_95 = np.interp( 0.950 , fact_up_Arr , H_Arr )

                ax.pcolormesh( edges_y , edges_x , H.T , cmap='Blues' )

                ax.contour( y_centers, x_centers , H.T , colors='k' , levels=[ H_value_95 ] )
                ax.contour( y_centers, x_centers , H.T , colors='r' , levels=[ H_value_68 ] )

                X_VALUE =  MAIN_VALUE_median[j]
                Y_VALUE =  MAIN_VALUE_median[i]

                ax.plot( [ X_VALUE , X_VALUE ] , [    -100 ,     100 ] , 'k--' , lw=2 )
                ax.plot( [    -100 ,     100 ] , [ Y_VALUE , Y_VALUE ] , 'k--' , lw=2 )

                ax.set_ylim( y_min-0.05*DDY , y_max+0.05*DDY )

            ax.set_xlim( x_min-0.05*DDX , x_max+0.05*DDX )

            if i==N_dim-1:
                ax.set_xlabel( label_list[j] , size=20 )

            if j==0 and i!=0 :
                ax.set_ylabel( label_list[i] , size=20 )

            if j!=0:
                plt.setp( ax.get_yticklabels(), visible=False)

            if j==0 and i==0:
                plt.setp( ax.get_yticklabels(), visible=False)

            if i!=len( label_list)-1 :
                plt.setp( ax.get_xticklabels(), visible=False)

    plt.subplots_adjust( left = 0.09 , bottom = 0.15 , right = 0.98 , top = 0.99 , wspace=0., hspace=0.)

    return None
######################################################################
######################################################################
######################################################################

#your_grids_location = '/Users/sidgurung/Desktop/ZELDA/Grids/'
your_grids_location = '/global/users/sidgurung/PROMARE/Grids/'

Lya.funcs.Data_location = your_grids_location

Geometry = 'Thin_Shell_Cont'

MODE = 'FULL'

LyaRT_Grid = Lya.load_Grid_Line( Geometry , MODE=MODE )

print( LyaRT_Grid.keys() )

z_t      = 0.5   # redshift of the source 
V_t      = 40.0  # Outfloe expansion velocity [km/s]
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
plt.savefig( 'fig_Tutorial_4_1.png' )
plt.clf()
########################################
########################################
########################################

N_walkers = 200
N_burn    = 200
N_steps   = 300

MODE = 'DNN'

log_V_in , log_N_in , log_t_in , log_E_in , W_in , z_in , Best = Lya.MCMC_get_region_6D( MODE , w_Arr , f_Arr , s_Arr , FWHM_t , PIX_t , LyaRT_Grid , Geometry )

sampler = Lya.MCMC_Analysis_sampler_5( w_Arr , f_Arr , s_Arr , FWHM_t , N_walkers , N_burn , N_steps , Geometry , LyaRT_Grid , z_in=z_in , log_V_in=log_V_in , log_N_in=log_N_in , log_t_in=log_t_in , log_E_in=log_E_in , W_in=W_in )

Q_Arr = [ 16 , 50 , 84 ]

N_hist_steps = 100

perc_matrix_sol , flat_samples = Lya.get_solutions_from_sampler( sampler , N_walkers , N_burn , N_steps , Q_Arr )

print( flat_samples.shape )

print( perc_matrix_sol )

z_16     =     perc_matrix_sol[ 3 , 0 ]
z_50     =     perc_matrix_sol[ 3 , 1 ]
z_84     =     perc_matrix_sol[ 3 , 2 ]

V_16     = 10**perc_matrix_sol[ 0 , 0 ]
V_50     = 10**perc_matrix_sol[ 0 , 1 ]
V_84     = 10**perc_matrix_sol[ 0 , 2 ]

t_16     = 10**perc_matrix_sol[ 2 , 0 ]
t_50     = 10**perc_matrix_sol[ 2 , 1 ]
t_84     = 10**perc_matrix_sol[ 2 , 2 ]

W_16     =     perc_matrix_sol[ 5 , 0 ]
W_50     =     perc_matrix_sol[ 5 , 1 ]
W_84     =     perc_matrix_sol[ 5 , 2 ]

log_E_16 =     perc_matrix_sol[ 4 , 0 ]
log_E_50 =     perc_matrix_sol[ 4 , 1 ]
log_E_84 =     perc_matrix_sol[ 4 , 2 ]

log_N_16 =     perc_matrix_sol[ 1 , 0 ]
log_N_50 =     perc_matrix_sol[ 1 , 1 ]
log_N_84 =     perc_matrix_sol[ 1 , 2 ]

PNR = 100000.

########################################
########################################
########################################
w_One_Arr , f_One_Arr , _  = Lya.Generate_a_real_line( z_50, V_50, log_N_50, t_50, F_t, log_E_50, W_50, PNR, FWHM_t, PIX_t, LyaRT_Grid, Geometry )

w_pix_One_Arr , f_pix_One_Arr = Lya.plot_a_rebinned_line( w_One_Arr , f_One_Arr , PIX_t )

plt.plot( w_pix_Arr     , f_pix_Arr     , label='Target' )
plt.plot( w_pix_One_Arr , f_pix_One_Arr , label='MCMC'   )

plt.legend(loc=0)
plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_4_2.png' )
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

########################################
########################################
########################################

make_corner_plots( flat_samples )

plt.savefig( 'fig_Tutorial_4_3.png' )

























