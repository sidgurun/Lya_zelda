Tutorial : Train your own neural network
=================================================================

In this tutorial you will, hopefully, learn how to train your own deep neural network to predict the proeprties of outflos/inflows. For this we are going to use the python package `scikitlearn` (https://scikit-learn.org/stable/).

Generating data sets for the training
*************************************

Let's start by loading `zELDA` grid of lines:

.. code:: python

          >>> import numpy as np
          >>> import Lya_zelda as Lya
          >>> import pickle
          >>> from sklearn.neural_network import MLPRegressor

          >>> your_grids_location = '/This/Folder/Contains/The/Grids/'
          >>> Lya.funcs.Data_location = your_grids_location

          >>> Geometry = 'Thin_Shell_Cont'

          >>> Lya.load_Grid_Line( Geometry )

And let's do it for outflows,

.. code:: python

          >>> MODE = 'Outflow' # 'Inflow' for inflows

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

Let's set the number of sources that we want in our sample, for example 1000,

.. code:: python

          >>> N_train = 1000

Let's generate the properties of each of the training examples:

.. code:: python

          >>> V_Arr , log_N_Arr , log_t_Arr , log_E_Arr , log_W_Arr = Lya.NN_generate_random_outflow_props_5D( N_train , log_V_in , log_N_in , log_t_in , log_E_in , log_W_in , MODE=MODE )
          
          >>> z_Arr = np.random.rand( N_train ) * ( z_in[1] - z_in[0] ) + z_in[0]
          
          >>> log_FWHM_Arr = np.random.rand( N_train ) * ( log_FWHM_in[1] - log_FWHM_in[0] ) + log_FWHM_in[0]
          >>> log_PIX_Arr  = np.random.rand( N_train ) * (  log_PIX_in[1] -  log_PIX_in[0] ) +  log_PIX_in[0]
          >>> log_PNR_Arr  = np.random.rand( N_train ) * (  log_PNR_in[1] -  log_PNR_in[0] ) +  log_PNR_in[0]

each of these arrays contains random values that will be used in the training, for example, `V_Arr` contains the expansion velocity, etc.

Let's initializate the arrays where we want to store the data that we will need for the training

.. code:: python

          >>> F_t = 1.0
          
          >>> Delta_True_Lya_Arr = np.zeros( N_train )
          
          >>> N_bins = 1000
          
          >>> z_PEAK_Arr = np.zeros( N_train )
          
          >>> LINES_train = np.zeros( N_train * N_bins ).reshape( N_train , N_bins )
          
          >>> N_bins_input = N_bins + 3
          
          >>> INPUT_train = np.zeros( N_train * N_bins_input ).reshape( N_train , N_bins_input )

Let's generate the lines using the function `Lya.Generate_a_line_for_training`,

.. code:: python

          >>> print( 'Generating training set' )
          
          >>> cc = 0.0
          >>> for i in range( 0, N_train ):
          
          >>>     per = 100. * i / N_train
          >>>     if per >= cc :
          >>>         print( cc , '%' )
          >>>         cc += 1.0
          
          >>>     V_t = V_Arr[i]
          >>>     t_t = 10**log_t_Arr[i]
          >>>     log_N_t = log_N_Arr[i]
          >>>     log_E_t = log_E_Arr[i]
          >>>     W_t = 10**log_W_Arr[i]
          
          >>>     z_t = z_Arr[i]
          
          >>>     FWHM_t = 10**log_FWHM_Arr[ i ]
          >>>     PIX_t  = 10**log_PIX_Arr[  i ]
          >>>     PNR_t = 10**log_PNR_Arr[i]
          
          >>>     rest_w_Arr , train_line , z_max_i , input_i = Lya.Generate_a_line_for_training( z_t , V_t, log_N_t, t_t, F_t, log_E_t, W_t , PNR_t, FWHM_t, PIX_t, DATA_LyaRT, Geometry)
          
          >>>     z_PEAK_Arr[i] = z_max_i
          
          >>>     Delta_True_Lya_Arr[ i ] = 1215.67 * ( (1+z_t)/(1+z_max_i) - 1. )
          
          >>>     LINES_train[i] = train_line
          >>>     INPUT_train[i] = input_i

.. code:: python

`rest_w_Arr` is the wavelength array where the profiles are evaluated in the rest frame of the peak of the line. `train_line` is the line profile evaluated in `rest_w_Arr`, `z_max_i` is the redshift of the source if the maximum of the line matches the Lyman-alpha line and `input_i` is the actual input that we will use for the DNN. 

Now let's save all the data

.. code:: python

          >>> dic = {}
          >>> dic[ 'lines' ] = LINES_train

          >>> dic[ 'NN_input' ] = INPUT_train

          >>> dic['z_PEAK'         ] = z_PEAK_Arr
          >>> dic['z'              ] = z_Arr
          >>> dic['Delta_True_Lya'] = Delta_True_Lya_Arr
          >>> dic['V'             ] = V_Arr
          >>> dic['log_N'         ] = log_N_Arr
          >>> dic['log_t'         ] = log_t_Arr
          >>> dic['log_PNR'       ] = log_PNR_Arr
          >>> dic['log_W'         ] = log_W_Arr
          >>> dic['log_E'         ] = log_E_Arr
          >>> dic['log_PIX'       ] = log_PIX_Arr
          >>> dic['log_FWHM'      ] = log_FWHM_Arr

          >>> dic['rest_w'] = rest_w_Arr

          >>> np.save( 'data_for_training.npy' , dic )

Done, now you have a set of data that can be used as training set. Of cource we have done it with only 1000 galaxies. In general you want to use about 100 000 or more. You can divide the data in small data sets for parallelitation and then combine them, for example.

Generating data sets for the training
*************************************

Let's load the data that we have just saved,

.. code:: python

          >>> Train_data = np.load( 'data_for_training.npy' , allow_pickle=True ).item()

Let's get the input that we will use in the training 

.. code:: python

          >>> Input_train = Train_data['NN_input']

Now let's load the properties that we want to predict,

.. code:: python

          >>> Train_Delta_True_Lya_Arr = Train_data['Delta_True_Lya']

          >>> Train_log_V_Arr = np.log10( Train_data[    'V'] )
          >>> Train_log_N_Arr =           Train_data['log_N']
          >>> Train_log_t_Arr =           Train_data['log_t']
          >>> Train_log_E_Arr =           Train_data['log_E']
          >>> Train_log_W_Arr =           Train_data['log_W']

and let's prepare it for skitlearn,

.. code:: python

          >>> TRAINS_OBSERVED = np.zeros( N_train * 6 ).reshape( N_train , 6 )

          >>> TRAINS_OBSERVED[ : , 0 ] = Train_Delta_True_Lya_Arr
          >>> TRAINS_OBSERVED[ : , 1 ] = Train_log_V_Arr
          >>> TRAINS_OBSERVED[ : , 2 ] = Train_log_N_Arr
          >>> TRAINS_OBSERVED[ : , 3 ] = Train_log_t_Arr
          >>> TRAINS_OBSERVED[ : , 4 ] = Train_log_E_Arr
          >>> TRAINS_OBSERVED[ : , 5 ] = Train_log_W_Arr

Now let's actually do the training. For this we have to decide what kind of deep learning configuration we want. For this tutorial let's use 2 hidden layers, each of 100 nodes, 

.. code:: python

          >>> hidden_shape = ( 100 , 100 )

And train,

.. code:: python

          >>> from sklearn.neural_network import MLPRegressor

          >>> est = MLPRegressor( hidden_layer_sizes=hidden_shape , max_iter=1000 )

          >>> est.fit( Input_train , TRAINS_OBSERVED )

Done! You have now your custom DNN. Let's save it now so that you can use it later

.. code:: python

          >>> dic = {}

          >>> dic['Machine'] = est
          >>> dic['w_rest' ] = rest_w_Arr

          >>> pickle.dump( dic , open( 'my_custom_DNN.sav' , 'wb'))


Done! Perfect. Now, remember, if you want to use you custom DNN you can follow all the steps in :doc:`Fitting a line profile using deep learning <Tutorial_DNN>`. The only difference is that, instead of loading the default DNN with `Lya.Load_NN_model()`, you have to load your DNN, which will also have the `dic['Machine']` and `dic['w_rest']` entries, as well the default one. Have fun! 














 
