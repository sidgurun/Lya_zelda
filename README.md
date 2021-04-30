# `FLaREON`

In this Universe everything is neat, even stochasticity.

Lyman alpha Resonant Scatter is dominated by random processes.

Although there is chaos, there is also an order.

## Authors

Siddhartha Gurung Lopez

Alvaro Orsi

Silvia Bonoli


## Publication links:

ADS : http://adsabs.harvard.edu/abs/2018arXiv181109630G

arXiv : https://arxiv.org/abs/1811.09630

## Origins and motivation

Due to the Lyman alpha Radiative Transfer large complexity, the efforts of understanding it moved from pure analytic studies to the so-called radiative transfer Monte Carlo (RTMC) codes that simulate Lyman alpha photons in arbitrary gas geometries. These codes provide useful information about the fraction of photons that manage to escape and the resulting Lyman alpha line profiles. The RTMC approach has shown to reproduce the observed properties of LAEs.

`FLaREON` is a publicly available `python` package based on a RTMC (Orsi et al. 2012) able to predict large amounts of Lyman alpha line profiles and escape fractions with high accuracy. We designed this code hoping that it helps researches all over the wolrd to get a better understanding of the Universe.

`FLaREON` is **fast** and **simple**. 

+ **Fast** : This code is able to predict Lyman alpha escape fractions and line profiles in an unprecedentedly low amount of time. In particular thousands of escape fractions and line profiles can be computed in less than a second of computational time. 

+ **Simple** : This code is *One-Line-Running*. Everyone, from a good trained monkey, passing through undergrade studients, to researches with basic python knowlege should be able to use `FLaREON`.

Although `FLaREON` is completely open source and is available for everyone, please, if you are using this code ( for whatever reason ) we will be very glad to hear about you and how you are using it. it's just a curiosity and efficiency matter. Maybe we can help you to have a smoother experience with `FLaREON`. For any issue or suggestion, please, contact `sidgurung@cefca.es`. Thank you.

## Installation

The easiest way to get `FLaREON` in your machine is through `pip`:

`pip install FLaREON`

This should do the trick. However, you can also install `FLaREON` downloading this repository and running

`pip install .`

in it.

In order to finish installation and check that everything is working properly, please, opening a `python`/`ipython` terminal and execute:

```python
import FLaREON as Lya

Lya.Test_Installation( Make_Plots = True )
```

This function checks if the pip installation downloaded everything. Then, if the data files are not found it tries to download them. Please, note that you should have about 1GB of free memory in you machine in order to install `FLaREON`. Also note that as ~1GB of data has to be downloaded, it might take some time depending on your internet conection.

After Checking that you have got the data in your machine `FLaREON` will check that everyhting works smoothly. For this some escape fractions and line profiles will be computed. The status of the operation should appear in the screen. Everything should get a `Succsess!!`. The only exception is 

`Running :  Bicone_X_Slab Analytic --> ERROR. HUMAN IS DEAD. MISMATCH!!`

Do not worry. The only reason why you are getting this error is because this algorithm is not implemented yet. Then the function will produce some plots. In case you want to run the tests without plotting just set `Make_Plots = False`  when calling the `Lya.Test_Installation`.

This should be all for the installation. If you find any trouble/bug duruing it, please, contact us at `sidgurung@cefca.es`. Thank you for your patience.

## Hands on the code.

( Assuming everything went smoothly in the installation... )

**Congratulations!!** You have just become one of the few chosen ones in the history of humankind to have the great pleasure of using `FLaREON`.

**The units**: This code uses velocites in km/s , column densities in cm^{-2} and wavelengths in meters.

First let's have a look to the dynamical range of the different parameters covered by `FLaREON`. For this we only need to do:

```python
import FLaREON as Lya

Lya.Print_the_grid_edges()
```


### Predicting thousands of Lyman alpha escape fractions.

Let's move to one of the most powerful products of `FLaREON`: predicting huge amounts of Lyman alpha escape fractions. 

In theory only one line is needed to predict the escape fraction for a thin shell geometry with expasion velocity (V) of 200km/s, logarithmic of column density (logNH) of 19.5 and dust optical depth (ta) of 0.1 :

```python
f_esc_Arr = Lya.RT_f_esc( 'Thin_Shell' , [ 200 ] , [ 19.5 ] , [ 0.1 ] )
```  

In this way `f_esc_Arr` is an Array of 1 dimension and length 1 that contains the predicted escape fraction for this configuration. 

However, `FLaREON` implements several gas geometries and is optimized to obtain large amount of escape fractions with only one line of code, so lets expand this a little bit more. If we want to compute the escape fraction in a thin shell outflow with the configurations { V , logNH , ta } , { 200 , 19.5 , 0.1 }, { 300 , 20.0 , 0.01 } and { 400 , 20.5 , 0.001 } we could do

```python
Geometry = 'Thin Shell' # Other options: 'Galactic Wind' or 'Bicone_X_Slab'

V_Arr     = [  200 ,  300 , 400   ] # Expansion velocity array in km/s

logNH_Arr = [ 19.5 , 20.0 , 20.5  ] # Logarithmic of column densities array in cm**-2

ta_Arr    = [  0.1 , 0.01 , 0.001 ] # Dust optical depth Array

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr ) 
```  

The variable `f_esc_Arr` is an Array of 1 dimension and length 3 that encloses the escape fractions for the configurations. In particular `f_esc_Arr[i]` is computed using `V_Arr[i]` ,  `logNH_Arr[i]` and `ta_Arr[i]`.

If the user wants to change the outflow gas geometry, they only have to do

```python
Geometry = 'Galactic Wind' # Other options: 'Thin Shell' or 'Bicone_X_Slab'

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr ) 
```  

Note that only one geometry can be used at the same time. If you want to compute different escape fractions ( or line profiles ) for different configurations you will need to call `FLaREON` once per geometry. 

These examples show how the `'Thin_Shell'` and `'Galactic_Wind'` geometries work. These geometries have spherical symmetry so there is no Line of Sight (LoS) dependence in the output escape fraction or line profile. However, `FLaREON` implements a non-spherical-symmetric geometry, the `'Bicone_X_Slab'` geometry (for details, again, we refer you to the presentation letter). In this particular geometry the escape fraction (and line profile) depends on the LoS. In particular, if you observed face-on (throught the biconical outflow) the optical depth is lower than observeing edge-on (through the static dense slab).

In order to tell `FLaREON` the orientation of observation ( edge-on or face-on ) the user needs to provide anther varible when calling the code: `Inside_Bicone_Arr`. If this is not given it is assumed that it's always observed face-on. This variable has to be a boolean array with the same size as `V_Arr` , `logNH_Arr` ot `ta_Arr`. Additionally, the aperture angle of the bicone is 45deg, so to produce a set of escape fractions with random orientations in the biconenical geometry you should use:

```python
import numpy as np

Geometry = 'Bicone_X_Slab'

Area_in_bicone = 1. - np.cos( np.pi/4. ) # the apperture angle is pi/4

Inside_Bicone_Arr = np.random.rand( len(V_Arr) ) < Area_in_bicone

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=Inside_Bicone_Arr ) 
```  

#### Deeper options on predicting the escape fraction .

There are many algorithims implemented to compute `f_esc_Arr`. By default `FLaREON` uses a machine learning decision tree regressor and a parametric equation for the escape fraction as function of the dust optical depth (Go to the `FLaREON` presentation paper Gurung-Lopez et al. in prep for more information). These settings were chosen as default since they give the best performance. However the user might want to change the computing algorithm so here leave a guide with all the available options.

+ `MODE` variable refers to mode in which the escape fraction is computed. There are 3 ways in which `FLaREON` can compute this. i) `'Raw'` Using the raw data from the RTMC (Orsi et al. 2012). ii) `'Parametrization'` Assume a parametric equation between the escape fraction and the dust optical depth that allows to extend calculations outside the grid with the highest accuracy (in `FLaREON`). iii) `'Analytic'` Use of the recalibrated analytic equations presented by Gurung-Lopez et al. 2018. Note that the analytic mode is not enabled in the bicone geometry although it is in the `'Thin_Shel'` and `'Galactic_Wind'`


+ `Algorithm` varible determines the technique used. This can be i) `'Intrepolation'`: lineal interpoation is used.  ii) `'Machine_Learning'` machine learning is used. To determine which machine learning algorithm you would like to use please, provide the variable `Machine_Learning_Algorithm`. The machine learning algorithms implemented are Decision tree regressor (`'Tree'`), Random forest regressor (`'Forest'`) and KN regressor (`'KN'`). The machine learning is implemented by `Sci-kit-learn`, please, visit their webside for more information (http://scikit-learn.org/stable/).

```python
MODE = 'Raw' # Other : 'Parametrization' , 'Analytic'

Algorithm = 'Intrepolation' # Other : 'Machine_Learning'

Machine_Learning_Algorithm = 'KN' # Other 'Tree' , 'Forest'

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE=MODE ) 
```  

Finally, any combination of `MODE` , `Algorithm` and `Machine_Learning_Algorithm` is allowed. However, note that the variable `Machine_Learning_Algorithm` is useless if `Algorithm='Intrepolation'`.


### Predicting thousands of Lyman alpha Line profiles.

In this section we explain how to obtain in a fast way an arbitray number of Lyman alpha line porfiles. The syntax is very similar to the one of escape fraction functions. The main difference is that the user must provide a wavelength array (in meters) where the line profile will be evaluated. The line profile of a thin shell outfow with expansion velocity (V) 200 km/s, logarithmic of column density (logNH) of 19.5 and dust optical depth (ta) of 0.1 in 20 amstrongs arround Lyman alpha can be computed as

```python
wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10 # meters, please!

Line_profile_Arr = Lya.RT_Line_Profile( 'Thin_Shell' , wavelength_Arr , [ 200 ] , [ 19.5 ] , [ 0.1 ] )
```

As in the case of the escape fraction, in order to compute multiple line profiles at the same time just make

```python
wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10 # meters, please!

V_Arr     = [  200 ,  300 , 400   ] # Expansion velocity array in km/s

logNH_Arr = [ 19.5 , 20.0 , 20.5  ] # Logarithmic of column densities array in cm**-2

ta_Arr    = [  0.1 , 0.01 , 0.001 ] # Dust optical depth Array

Line_profile_Arr = Lya.RT_Line_Profile( 'Thin_Shell' , wavelength_Arr , V_Arr , logNH_Arr , ta_Arr )
```


In this case, `Line_profile_Arr` is an array with shape (3,1000) that contains the computed line profiles. In particular `Line_profile_Arr[i,:]` is the line profile evaluated in `wavelength_Arr` computed with `V_Arr[i]` ,  `logNH_Arr[i]` and `ta_Arr[i]`. 

The other geometries (`'Galactic_Wind'` or `'Bicone_X_Slab'`) are also implemented. In particular, in the biconical geometry it is also possible to chose a line of sight observations. This is implemented in the same way as in the escape fraction. 


In opposite to escape fraction calculations, the line profile only supports by now linear interpolation between the pre-computed grids. Machine learning or deep learning might be implement in a future.


### A powerfull aliance :  Joining FLaREON and emcee.

`emcee` (http://dfm.io/emcee/current/) is a popular `python` package including tool to perform MCMC analysis. 

In the folder `EXAMPLE_MCMC/` we include a basic examplem of how to wrap up `emcee` and `FLaREON`.

`MCMC_fitter.py` is a script that generate a problem line profile and run an MCMC analysis to determine the ( column density - expansion velocty - dust optical depth ) combination that best reproduces the problem line profile. You can find a highly detailed explanation of the how the code works within it. 

`MCMC_fitter.py` saves a file with all the MCMC information and name it acordingly to the MCMC configuration used.

`Analize_mcmc.py` is another script to extract information from the saved file. You can set the MCMC parameters inside the code or you might just pass as an argument the name of the saved file:

```python
ipython Analize_mcmc.py mcmc_output_Galactic_Wind_None_ndim_3_nsteps_1000_nwalkers_100.txt
```

where the `mcmc_output_Galactic...txt` is the files saved by `MCMC_fitter.py`. 

In this particular case, `Analize_mcmc.py` prints:

```python
if the ouflow geometry is Galactic_Wind, then, the best fitting values are...:
    V_exp         =  62.4315
    V_exp_err_top =  0.0
    V_exp_err_bot =  0.0
    logNH         =  17.14153
    logNH_err_top =  0.0
    logNH_err_bot =  0.0
    logta         =  -0.250951
    logta_err_top =  0.0
    logta_err_bot =  0.0
```

where `V_exp` , `logNH` and `logta` are the best fitting expansion velocity, logarithm of column density and logarithm of dust optical depth for the line profile produced inside `MCMC_fitter.py` .

We remark that this is just a simple script to ilustrate how to make a efficient wrap up in `emcee`. Users should develope their own codes to obtain better results on their individual estudies. However, we hope that the structure of `MCMC_fitter.py` can help users. 
