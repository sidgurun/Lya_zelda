Tutorial : Train your own neural network
=================================================================

In this tutorial you will, hopefully, learn how to train your own deep neural network to predict the proeprties of outflos/inflows. For this we are going to use the python package `scikitlearn` (https://scikit-learn.org/stable/).

Generating data sets for the training
*************************************

Let's start by loading `zELDA` grid of lines:

.. code:: python

          >>> import Lya_zelda as Lya
          >>> your_grids_location = '/This/Folder/Contains/The/Grids/'
          >>> Lya.funcs.Data_location = your_grids_location

          >>> Geometry = 'Thin_Shell_Cont'

          >>> Lya.load_Grid_Line( Geometry )

Let's define the region where we want to generate mock line profiles. You can adjust this to whatever your want. The values presented here are the standard in `zELDA`, but you can change them.

.. code:: python

          >>> # Logarithm of the expansion velocity in [km/s]
          >>> log_V_in = [  1.0   ,  3.0   ]

          >>> # Logarithm of the HI column density [cm**-2]
          >>> log_N_in = [ 17.0   , 21.5   ]

          >>> # Logarithm of the dust optical depth
          >>> log_t_in = [ -4.0   , 0.0  ]

          >>> # Logarithm of the intrinsic equivalent width [A]
          >>> log_E_in = [ 0.0    , 2.3  ]

          >>> # Logarithm of the intrinsic line width [A]
          >>> log_W_in = [ -2.    , 0.7  ]

          >>> #Redshift interval
          >>> z_in = [ 0.0001 , 4.00 ]

          >>> # Logarithm of the full width half maximum convolving the spectrum (resolution) [A]
          >>> log_FWHM_in = [ -1.  ,   0.3  ]

          >>> # Logarithm of the pixel size [A]
          >>> log_PIX_in  = [ -1.3 ,   0.3  ]

          >>> # Logarithm of the signal to noise of the peak of the line
          >>> log_PNR_in = [ 0.7 , 1.6 ]
  
Each of these lists have 2 elementes. For example, `log_V_in[0]` indicates the lower border of the interval and `log_V_in[1]` the upper limit.



