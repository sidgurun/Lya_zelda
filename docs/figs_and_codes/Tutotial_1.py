import numpy as np

import Lya_zelda as Lya

import sys

import pylab as plt

#your_grids_location = '/Users/sidgurung/Downloads/Grids/'
your_grids_location = '/Users/sidgurung/Desktop/ZELDA/Grids/'

Lya.funcs.Data_location = your_grids_location

Geometry = 'Thin_Shell_Cont'

MODE = 'FULL'

LyaRT_Grid = Lya.load_Grid_Line( Geometry , MODE=MODE )

print( LyaRT_Grid.keys() )
 
V_Value     = 50.0 # Outfloe expansion velocity [km/s]
logNH_Value = 20.   # Logarithmic of the neutral hydrogen column density [cm**-2]
ta_Value    = 0.01  # Dust optical depth
logEW_Value = 1.5   # Logarithmic the intrinsic equivalent width [A]
Wi_Value    = 0.5   # Intrinsic width of the line [A]

w_Lya = 1215.68

wavelength_Arr = np.linspace( w_Lya-10 , w_Lya+10 , 1000 ) * 1e-10

Line_Arr = Lya.RT_Line_Profile_MCMC( Geometry , wavelength_Arr , V_Value , logNH_Value , ta_Value , LyaRT_Grid , logEW_Value=logEW_Value , Wi_Value=Wi_Value )

plt.plot( wavelength_Arr *1e10 , Line_Arr ) 

plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )

plt.savefig( 'fig_Tutorial_1_1.png' )
plt.clf()

########################################
########################################
########################################

V_Arr     = [ 50.0 , 100.   , 200.    ] # Outfloe expansion velocity [km/s]
logNH_Arr = [ 18.0 ,  19.   ,  20.    ] # Logarithmic of the neutral hydrogen column density [cm**-2]
ta_Arr    = [  0.1 ,   0.01 ,   0.001 ] # Dust optical depth
logEW_Arr = [  1.0 ,   1.5  ,   2.0   ] # Logarithmic the intrinsic equivalent width [A]
Wi_Arr    = [  0.1 ,   0.5  ,   1.0   ] # Intrinsic width of the line [A]

Line_Matrix = Lya.RT_Line_Profile( Geometry , wavelength_Arr , V_Arr , logNH_Arr , ta_Arr , logEW_Arr=logEW_Arr , Wi_Arr=Wi_Arr )

for i in range( 0 , 3 ) :

    print( Line_Matrix[i])

    plt.plot( wavelength_Arr *1e10 , Line_Matrix[i] )

plt.xlabel('wavelength[A]' , size=15 )
plt.ylabel('Flux density [a.u.]' , size=15 )

plt.savefig( 'fig_Tutorial_1_2.png' )
plt.clf()





