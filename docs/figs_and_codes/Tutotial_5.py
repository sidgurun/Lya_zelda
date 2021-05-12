import numpy as np

import Lya_zelda as Lya

import sys

import pylab as plt

# Setting the locations of your grids

your_grids_location = '/Users/sidgurung/Desktop/test_2/Grids/'

Lya.funcs.Data_location = your_grids_location

# Line profile grids:

Geometry = 'Thin_Shell'

LyaRT_Grid = Lya.load_Grid_Line( Geometry )

print( LyaRT_Grid.keys() ) 

print( 'The expansion velocity [km/s] is evaluated in : ') 
print( LyaRT_Grid['V_Arr'    ] )

print( 'The logarithmic of the HI column density [cm**-2] is evaluated in : ') 
print( LyaRT_Grid['logNH_Arr'] )

print( 'The logarithmic of the dust optical depth is evaluated in : ') 
print( LyaRT_Grid['logta_Arr'] )

print( LyaRT_Grid['Grid'     ].shape )
print( LyaRT_Grid['V_Arr'    ].shape )
print( LyaRT_Grid['logNH_Arr'].shape )
print( LyaRT_Grid['logta_Arr'].shape )

w_Arr = Lya.convert_x_into_lamda( LyaRT_Grid['x_Arr'] )

plt.plot( w_Arr , LyaRT_Grid['Grid'][0,1,2,] )
plt.xlim( 1213*1e-10 , 1218*1e-10 )
plt.xlabel( 'wavelength [m]' )
plt.ylabel( 'Flux density [a.u.]' )
plt.savefig( 'fig_Tutorial_5_1.png' )
plt.clf()

# Line grids with smaller RAM occupation

Geometry = 'Thin_Shell_Cont'

print( Lya.load_Grid_Line )

LyaRT_Grid_Full = Lya.load_Grid_Line( Geometry , MODE='FULL' )

print( LyaRT_Grid_Full.keys() )

LyaRT_Grid_Light = Lya.load_Grid_Line( Geometry , MODE='LIGHT' )

print( LyaRT_Grid_Light.keys() )


print( 'The expansion velocity [km/s] is evaluated in : ') 
print( LyaRT_Grid_Full['V_Arr'] )

print( 'The logarithmic of the HI column density [cm**-2] is evaluated in : ') 
print( LyaRT_Grid_Full['logNH_Arr'] )

print( 'The logarithmic of the dust optical depth is evaluated in : ') 
print( LyaRT_Grid_Full['logta_Arr'] )

print( 'The logarithmic of the intrinsic equivalent width [A] is evaluated in : ') 
print( LyaRT_Grid_Full['logEW_Arr'] )

print( 'The logarithmic of the intrinsic line width [A] is evaluated in : ') 
print( LyaRT_Grid_Full['Wi_Arr'] )

print('=========================')

print( 'The logarithmic of the intrinsic equivalent width [A] is evaluated in : ') 
print( LyaRT_Grid_Light['logEW_Arr'] )

print( 'The logarithmic of the intrinsic line width [A] is evaluated in : ') 
print( LyaRT_Grid_Light['Wi_Arr'] )














