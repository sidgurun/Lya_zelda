import numpy as np

import matplotlib
# see http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
matplotlib.use('Svg')

from pylab import *

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import Lya_zelda as Lya

GEO = 'COOL_SHELL_EDGE'#'HALF_SIN_SHELL'#'SIN'#'COOL' #'SHELL' # 'COOL'

LINE_MODEL = 'DOUBLE'#'DOUBLE'#'SINGLE'

LINE_range = 0.8#2.#2#0.80
#LINE_range = 0.8#2#0.80

zELDA = 30
s_p=100

if LINE_range == 2.0:
    zELDA = 30
    s_p   = 100  

if LINE_range == 0.8:
    zELDA = 40
    s_p   = 180  

corlor_shell = 'cornflowerblue'

color_cool = 'w'

#################################################
#################################################
#################################################
your_grids_location = '/global/users/sidgurung/PROMARE/Grids/'

Lya.funcs.Data_location = your_grids_location

Geometry = 'Thin_Shell_Cont'

DATA_LyaRT = Lya.load_Grid_Line( Geometry , MODE='LIGHT' )

w_Lya = 1215.68 # Lyman-alpha wavelength in amstrongs
w_Arr = np.linspace( w_Lya-10 , w_Lya+10 , 1000 ) * 1e-10

if LINE_MODEL == 'SINGLE' :
    V_Value     = 10**2.2  # Outflow expansion velocity [km/s]
    logNH_Value = 19.6     # Logarithmic of the neutral hydrogen column density [cm**-2]
    ta_Value    = 0.1      # Dust optical depth
    logEW_Value = 1.15     # Logarithmic the intrinsic equivalent width [A]
    Wi_Value    = 10**-0.2 # Intrinsic width of the line [A]

if LINE_MODEL == 'DOUBLE' :
    V_Value     = 50.0  # Outflow expansion velocity [km/s]
    logNH_Value = 20.   # Logarithmic of the neutral hydrogen column density [cm**-2]
    ta_Value    = 0.01  # Dust optical depth
    logEW_Value = 1.5   # Logarithmic the intrinsic equivalent width [A]
    Wi_Value    = 0.5   # Intrinsic width of the line [A]
    
Line_Arr = Lya.RT_Line_Profile_MCMC( Geometry , w_Arr , V_Value , logNH_Value , ta_Value , DATA_LyaRT , logEW_Value=logEW_Value , Wi_Value=Wi_Value )
#################################################
#################################################
#################################################

w_Arr = w_Arr * 1e10

w_Arr = ( w_Arr - w_Lya ) * 0.2

#################################################
#################################################
#################################################

Line_Arr = Lya.dilute_line( w_Arr , Line_Arr , 0.1 )

Line_Arr = Line_Arr - Line_Arr[0]

Line_Arr = Line_Arr * 0.8 / np.amax(Line_Arr)

#################################################
#################################################
#################################################

mask_range = np.absolute( w_Arr ) < LINE_range 

w_Arr    =    w_Arr[ mask_range ]
Line_Arr = Line_Arr[ mask_range ]

#################################################
#################################################
#################################################

ax = subplot(111)

ax.set_aspect('equal', 'box')

ang_Arr = np.linspace(0, 2 * np.pi, 1000 )

R = 1.0

#################################################
#################################################
#################################################

# Plot Shell

#################################################
if GEO == 'SHELL':
    X_Arr = R * np.cos( ang_Arr )
    Y_Arr = R * np.sin( ang_Arr )
    
    ax.plot( X_Arr , Y_Arr , color=corlor_shell , lw=4 )
#################################################
if GEO == 'COOL':

    T = 8.

    A = 0.05

    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 100 )

    X_1_Arr = R * np.cos( ang_Arr ) + A * np.cos( ang_1_Arr ) 
    Y_1_Arr = R * np.sin( ang_Arr ) + A * np.sin( ang_1_Arr )
    
    X_2_Arr = R * np.cos( ang_Arr ) - A * np.cos( ang_1_Arr ) 
    Y_2_Arr = R * np.sin( ang_Arr ) - A * np.sin( ang_1_Arr )

    ax.plot( X_1_Arr , Y_1_Arr , color=corlor_shell , lw=1 )
    ax.plot( X_2_Arr , Y_2_Arr , color=corlor_shell , lw=1 )
#################################################
if GEO == 'COOL_SHELL':
    X_Arr = R * np.cos( ang_Arr )
    Y_Arr = R * np.sin( ang_Arr )
    
    ax.plot( X_Arr , Y_Arr , color=corlor_shell , lw=12 )

    T = 8.

    A = 0.02

    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 1000 )

    X_1_Arr = R * np.cos( ang_Arr ) + A * np.cos( ang_1_Arr )
    Y_1_Arr = R * np.sin( ang_Arr ) + A * np.sin( ang_1_Arr )

    X_2_Arr = R * np.cos( ang_Arr ) - A * np.cos( ang_1_Arr )
    Y_2_Arr = R * np.sin( ang_Arr ) - A * np.sin( ang_1_Arr )

    ax.plot( X_1_Arr , Y_1_Arr , color=color_cool , lw=1 )
    ax.plot( X_2_Arr , Y_2_Arr , color=color_cool , lw=1 )
#################################################
if GEO == 'COOL_SHELL_EDGE':

    A = 0.05

    XX_1_Arr = ( R + A ) * np.cos( ang_Arr )
    YY_1_Arr = ( R + A ) * np.sin( ang_Arr )

    XX_2_Arr = ( R - A ) * np.cos( ang_Arr )
    YY_2_Arr = ( R - A ) * np.sin( ang_Arr )

    ax.plot( XX_1_Arr , YY_1_Arr , color=corlor_shell , lw=2 )
    ax.plot( XX_2_Arr , YY_2_Arr , color=corlor_shell , lw=2 )

    T = 8.


    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 1000 )

    X_1_Arr = R * np.cos( ang_Arr ) + A * np.cos( ang_1_Arr )
    Y_1_Arr = R * np.sin( ang_Arr ) + A * np.sin( ang_1_Arr )

    X_2_Arr = R * np.cos( ang_Arr ) - A * np.cos( ang_1_Arr )
    Y_2_Arr = R * np.sin( ang_Arr ) - A * np.sin( ang_1_Arr )

    ax.plot( X_1_Arr , Y_1_Arr , color=corlor_shell , lw=2 )
    ax.plot( X_2_Arr , Y_2_Arr , color=corlor_shell , lw=2 )
#################################################
if GEO == 'SIN':

    T = 16.

    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 1000 )

    sin_Arr = np.sin( ang_1_Arr * T )    

    X_Arr = R * np.cos( sin_Arr )
    Y_Arr = R * np.sin( sin_Arr )

    ax.plot(  X_Arr , Y_Arr , color=corlor_shell , lw=1 )
    ax.plot( -X_Arr , Y_Arr , color=corlor_shell , lw=1 )
#################################################
if GEO == 'SIN_SHELL':
    X_Arr = R * np.cos( ang_Arr )
    Y_Arr = R * np.sin( ang_Arr )
    
    ax.plot( X_Arr , Y_Arr , color=corlor_shell , lw=4 )

    T = 16.

    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 1000 )

    sin_Arr = np.sin( ang_1_Arr * T )    

    X_Arr = R * np.cos( sin_Arr )
    Y_Arr = R * np.sin( sin_Arr )

    ax.plot(  X_Arr , Y_Arr , color=corlor_shell , lw=1 )
    ax.plot( -X_Arr , Y_Arr , color=corlor_shell , lw=1 )
#################################################
if GEO == 'HALF_SIN_SHELL':
    X_Arr = R * np.cos( ang_Arr )
    Y_Arr = R * np.sin( ang_Arr )

    ax.plot( X_Arr , Y_Arr , color=corlor_shell , lw=4 )

    T = 16.

    ang_1_Arr = np.linspace(0, 2 * np.pi * T , 1000 )

    sin_Arr = np.sin( ang_1_Arr * T )

    X_Arr = R * np.cos( sin_Arr )
    Y_Arr = R * np.sin( sin_Arr )

    ax.plot(  X_Arr , Y_Arr , color=corlor_shell , lw=1 )
#################################################
#################################################
#################################################
#Plot line:
ax.plot( w_Arr , Line_Arr , 'k' , lw=2 )

ax.text( .5 , 0.30 , r'$z\:{\rmELDA}$' , verticalalignment='center', horizontalalignment='center', transform=ax.transAxes, fontsize=zELDA)

ax.scatter( [0] , [0] , marker='o' , color=corlor_shell , s=s_p )

#ax.set_xlabel( 'zELDA' , size=20 )

plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_xticklabels(), visible=False)

ax.tick_params(axis=u'both', which=u'both',length=0)

ax.axis('off')

savefig( 'fig_log_'+LINE_MODEL+'_'+GEO+'_r_' + str(LINE_range) + '_s_' + str(zELDA) +'.svg' , bbox_inches='tight' )
savefig( 'fig_log_'+LINE_MODEL+'_'+GEO+'_r_' + str(LINE_range) + '_s_' + str(zELDA) +'.pdf' , bbox_inches='tight' )
savefig( 'fig_log_'+LINE_MODEL+'_'+GEO+'_r_' + str(LINE_range) + '_s_' + str(zELDA) +'.png' , bbox_inches='tight' , dpi = 1000 )




