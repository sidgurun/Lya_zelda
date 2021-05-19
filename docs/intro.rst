Introduction
============

`zELDA` [redshift (`z`) Estimator using Line profiles of Distant lyman-Alpha emitters], a code to understand Lyman-alpha emission.

Authors
*******

| Siddhartha Gurung Lopez
| Max Gronke
| Alvaro Orsi
| Silvia Bonoli
| Shun Saito

Publication links:
******************

`zELDA` paper:

| ADS   : ???
| arXiv : ???

| `zELDA` is based on its previous version, `FLaREON`. Please, if you used `zELDA` in your project, cite also `FLaREON`:

| ADS   : http://adsabs.harvard.edu/abs/2018arXiv181109630G
| arXiv : https://arxiv.org/abs/1811.09630

Origins and motivation
**********************

The main goal of `zELDA` is to provide to the scientific community a common tool to analyze and model Lyman-alpha line profiles. 


`zELDA` is a publicly available `python` package based on a RTMC (Orsi et al. 2012) and `FLaREON` (Gurung-Lopez 2019) able to fit observed Lyman-alpha spectrum and to predict large amounts of Lyman alpha line profiles and escape fractions with high accuracy. We designed this code hoping that it helps researches all over the world to get a better understanding of the Universe. In particular `zELDA` is divided in two main functionalites:

*  **Mocking Lyman-alpha line profiles**. Due to the Lyman alpha Radiative Transfer large complexity, the efforts to understand Lyman-alpha emission moved from pure analytic studies to the so-called radiative transfer Monte Carlo (RTMC) codes that simulate Lyman alpha photons in arbitrary gas geometries. These codes provide useful information about the fraction of photons that manage to escape and the resulting Lyman alpha line profiles. The RTMC approach has probed very successful in reproducing the observed properties of Lyman-alpha emitters. `zELDA` contains several data grids of `LyaRT`, the RTMC described in Orsi et al. 2012 (https://github.com/aaorsi/LyaRT), from which Lyman-alpha line profiles are computed using lineal interpolation. This methodology allow us to predict line profiles with a high accuracy at a low computational cost. In fact, the time used by `zELDA` to predict a single line profiles y usually eight orders of magnitud smaller than the full radiative transfer analysis done by `LyaRT`. Additionally, in order to mock observed Lyman-alpha spectrum, `zELDA` also includes routines to mimik the artifacts induced by observations in the line profiles, such a finite spectral resolution or the wavelength binning. 
*  **Fitting observed Lyman-alpha line profiles**. The main update from `FLaREON` to `zELDA` is the inclusion of several fitting algotirhms to model observed Lyman-alhpa line profiles. On the basics, `zELDA` uses mock Lyman-alpha line profiles to fit observed spectrums in two main phasions :

  *  **Monte Carlo Markov Chain** : This is the most classic approach taken in the literaute (e.g. Gronke et al. 2017). `zELDA` implementation is powered by the public code `emcee` (https://emcee.readthedocs.io/en/stable/) by Daniel Foreman-Mackey et al. (2013). 
  
  *  **Deep learning** : `zELDA` is the first open source code that uses machine learning to fit Lyman-alpha line profiles. `zELDA` includes some trained deep neural networks that predicts the best inflow/outflow model and redshift for a given observed line profile. This approach is about 3 orders of magnitud faster than the MCMC analysis and provides similar accuracies. This methodology will prove decesive in the upcoming years when tens of thousands of Lyman-alpha line profiles will be measure by instruments such as the James Webb Space Telescope. The neural network engine powering `zELDA` is `scikitlearn` (https://scikit-learn.org/stable/).  








