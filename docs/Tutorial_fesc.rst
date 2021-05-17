Tutorial : Computing Lyman-alpha escape fractions
=================================================

In this tutorial you will, hopefully, learn how to compute Lyman-alpha escape fractions with `zELDA`. Note that this part of the code compres directly from `FLaREON` ( https://github.com/sidgurun/FLaREON , Gurung-lopez et al. 2019b).

Default computation of escape fractions
***************************************

Let's move to one of the most powerful products of `FLaREON`: predicting huge amounts of Lyman alpha escape fractions.

However, `zELDA` implements several gas geometries and is optimized to obtain large amount of escape fractions with only one line of code, so lets expand this a l    ittle bit more. If we want to compute the escape fraction in a thin shell outflow with the configurations { V , logNH , ta } , { 200 , 19.5 , 0.1 }, { 300 , 20.0 ,     0.01 } and { 400 , 20.5 , 0.001 } we could do

.. code:: python

          >>> import Lya_zelda as Lya
          >>> your_grids_location = '/This/Folder/Contains/The/Grids/'
          >>> Lya.funcs.Data_location = your_grids_location

          >>> Geometry = 'Thin_Shell' 
          >>> # Other options: 'Galactic Wind' or 'Bicone_X_Slab_In' or 'Bicone_X_Slab_Out'

          >>> # Expansion velocity array in km/s
          >>> V_Arr     = [  200 ,  300 , 400   ] 

          >>> # Logarithmic of column densities array in cm**-2
          >>> logNH_Arr = [ 19.5 , 20.0 , 20.5  ] 

          >>> # Dust optical depth Array
          >>> ta_Arr    = [  0.1 , 0.01 , 0.001 ] 

Where `Geometry` indicates the gas disitrubitons that is being used. 'Bicone_X_Slab_In' indicates the bicone geometry look through the outflow, while 'Bicone_X_Slab_In' is looking through the optically thick gas. The 'Thin_Shell_Cont' model does not support escape fractions yet.  

Now let's compute the escape fraction for this configuraitons:

.. code:: python

          >>> f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr )

The variable `f_esc_Arr` is an Array of 1 dimension and length 3 that encloses the escape fractions for the configurations. In particular `f_esc_Arr[i]` is computed     using `V_Arr[i]` ,  `logNH_Arr[i]` and `ta_Arr[i]`.

Deeper options on predicting the escape fraction 
************************************************

There are many algorithims implemented to compute `f_esc_Arr`. By default `FLaREON` uses a machine learning decision tree regressor and a parametric equation for th    e escape fraction as function of the dust optical depth (Go to the `FLaREON` presentation paper Gurung-Lopez et al. in prep for more information). These settings we    re chosen as default since they give the best performance. However the user might want to change the computing algorithm so here leave a guide with all the availabl    e options.

+ `MODE` variable refers to mode in which the escape fraction is computed. There are 3 ways in which `FLaREON` can compute this. i) `'Raw'` Using the raw data from     the RTMC (Orsi et al. 2012). ii) `'Parametrization'` Assume a parametric equation between the escape fraction and the dust optical depth that allows to extend calcu    lations outside the grid with the highest accuracy (in `FLaREON`). iii) `'Analytic'` Use of the recalibrated analytic equations presented by Gurung-Lopez et al. 201    8. Note that the analytic mode is not enabled in the bicone geometry although it is in the `'Thin_Shel'` and `'Galactic_Wind'`


+ `Algorithm` varible determines the technique used. This can be i) `'Intrepolation'`: lineal interpoation is used.  ii) `'Machine_Learning'` machine learning is us    ed. To determine which machine learning algorithm you would like to use please, provide the variable `Machine_Learning_Algorithm`. The machine learning algorithms i    mplemented are Decision tree regressor (`'Tree'`), Random forest regressor (`'Forest'`) and KN regressor (`'KN'`). The machine learning is implemented by `Sci-kit-learn`, please, visit their webside for more information (http://scikit-learn.org/stable/).

.. code:: python
          MODE = 'Raw' # Other : 'Parametrization' , 'Analytic'
          
          Algorithm = 'Intrepolation' # Other : 'Machine_Learning'
          
          Machine_Learning_Algorithm = 'KN' # Other 'Tree' , 'Forest'
          
          f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE=MODE )

Finally, any combination of `MODE` , `Algorithm` and `Machine_Learning_Algorithm` is allowed. However, note that the variable `Machine_Learning_Algorithm` is useles    s if `Algorithm='Intrepolation'`.







