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

PNR_t  = 10.0 # Signal to noise ratio of the maximum of the line.
FWHM_t = 0.5  # Full width half maximum diluting the line. Mimics finite resolution. [A]
PIX_t  = 0.2  # Wavelength binning of the line. [A]

#w_Lya = 1215.68
#
#wavelength_Arr = np.linspace( w_Lya-10 , w_Lya+10 , 1000 ) * 1e-10

w_Arr , f_Arr , _ = Lya.Generate_a_real_line( z_t , V_t, log_N_t, t_t, F_t, log_EW_t, W_t , PNR_t, FWHM_t, PIX_t, LyaRT_Grid, Geometry )

plt.plot( w_Arr , f_Arr ) 

plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_2_1.png' )
plt.clf()

########################################
########################################
########################################

w_pix_Arr , f_pix_Arr = Lya.plot_a_rebinned_line( w_Arr , f_Arr , PIX_t )

plt.plot( w_pix_Arr , f_pix_Arr ) 

plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )
plt.xlim(1815,1835)
plt.savefig( 'fig_Tutorial_2_2.png' )
plt.clf()
########################################
########################################
########################################




