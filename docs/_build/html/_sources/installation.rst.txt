Installation
============

`zELDA`, installation is divided in two blocks. First you will need to install the python package containing all the scritps. With this you can already use the Deep Neural Network methodologies to extract information from observed Lyman-alpha line profiles. The second block contains all the grids computed from `FLarEON`. This are necessary in order to compute line profiles and escape fractions for all the outflow geometries. As a consequence, the second block is mandatory to make MCMC analysis.  

Python package
**************

The simpliest way of installing `zELDA`'s scripts is pip:

.. code:: python

          $ pip install Lya_zelda

An alternative method to install `zELDA`'scripts is downloading the code from GitHub:

.. code:: python

          $ git clone https://github.com/sidgurun/Lya_zelda.git
          $ cd Lya_zelda
          $ pip install .

Remember that you can also add the tag ``--user`` ,  if necessary.

LyaRT data grids
****************

Next, let's download the data grids necessary for generating mock Lyman-alpha line profiles as escape fractions. The data is stored at https://zenodo.org/record/4733518#.YJjw_y_Wf0c . Download the `Grids.zip` file and unzip it in the place that you want to keep it.

In order to compute line profiles and escape fraction you will need to indicate `zELDA` the location of grids by doing 

.. code:: python

          >>> import Lya_zelda as Lya

          >>> your_grids_location = '/This/Folder/Contains/The/Grids/'

          >>> Lya.funcs.Data_location = your_grids_location

where `your_grids_location` is a `string` with the place where you have stored the grids. If you run the `ls` command you should see something like this:

.. code:: python

          $ ls /This/Folder/Contains/The/Grids/

          Dictonary_Bicone_X_Slab_Grid_Lines_In_Bicone_False.npy
          .
          .
          .
          GRID_data__V_29_logNH_19_logta_9_EW_20_Wi_31.npy
          GRID_data__V_29_logNH_19_logta_9_EW_8_Wi_9.npy
          GRID_info__V_29_logNH_19_logta_9_EW_20_Wi_31.npy
          GRID_info__V_29_logNH_19_logta_9_EW_8_Wi_9.npy
          .
          .
          .
          finalized_model_wind_f_esc_Tree_f_esc.sav

You can check if you have set properly the directoy by loading a grid after setting `Lya.funcs.Data_location`, for example:

.. code:: python

          >>> grid = Lya.load_Grid_Line( 'Thin_Shell' )

If the location has been properly set the command should run smoothly and `grid` should contain the information for the interpolation.  




