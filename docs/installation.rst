.. _Target_Installation:

Installation
============

`zELDA`, installation is divided in two blocks. First you will need to install the python package containing all the scritps. With this you can already use the Deep Neural Network methodologies to extract information from observed Lyman-alpha line profiles. The second block contains all the grids computed from `LyaRT`. These are necessary in order to compute line profiles and escape fractions for all the outflow geometries. As a consequence, the second block is mandatory to make MCMC analysis.  

Python package
**************

The simplest way of installing `zELDA`'s scripts is via pip:

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

Next, let's download the data grids necessary for generating mock Lyman-alpha line profiles as escape fractions. The data is stored at https://zenodo.org/record/4733518#.YJjw_y_Wf0c . Download the `Grids.zip` file. You can do this in different ways. The recomended method is using the commamnd `wget` or `curl`, which should be more stable. For example, for downloawing it with `curl`, you can do:

.. code:: python

          $ curl -0 --compressed --output Grids.zip https://zenodo.org/record/4733518/files/Grids.zip

The download might take a while, as it is about 12Gb, so grab your fauvorite snack and be patient =D .

Other way of getting the data is going to the `zenodo`  webpage and download it through your internet borwser. As this is a large file, if you brower is a little bit unstable the download might stop in halfway, causing you to restart the download again. 

Once you have the `Grids.zip` file, unzip it in the place that you want to keep it.

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

          >>> print( Lya.Check_if_DATA_files_are_found() )

If the location has been properly set the command should return 1. If the data files are not found, then 0 is return. This function will also tell you the current value of `Lya.funcs.Data_location`. If the funtions returns 0 make sure than running `ls` gives you the expected output (see just above). 




