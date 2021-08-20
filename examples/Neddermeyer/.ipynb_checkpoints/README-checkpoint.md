In the Neddermeyer experiments, steel cylinders were imploded using high explosive. In this example we reproduce the experiments by generating synthetic data. The system response is the inner radius of the cylinder, indexed over both time and angle. In neddermeyer.ipynb, we show how the data is generated and reproduce some plots from Higdon et al. 2008, Computer Model Calibration Using High-Dimensional Output.
In this full example you will see how to:
  1. Generate observed and simulated data
  2. Generate basis decomposition matrices
  3. Package data and basis elements into SepiaData object
  4. Run MCMC
  5. Make predictions using posterior samples
