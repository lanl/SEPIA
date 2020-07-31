#### Development needs, with questions, issues, observations, discussion. 
At the top level, the big coding issues are, first cut of priority:
1. sensitivity analysis (started translating from matlab but put aside for now)
2. kron enabling

Documentation:
1. View current build on [Read the Docs](https://sepia-lanl.readthedocs.io/en/latest/)
2. See docstring commenting style used in SepiaModel.py to add more documentation
3. Can also edit docs/*.rst files to add text/sections to doc
4. Local build of documentation: see doc/README.md

Community and Engagement
- recreate/document (and go further with?) historical examples
- Announce broadly
- Know what to do next...

Performance Profiling and Optimization

---

##### Overall structure
Establish appropriate (and/or pythonic) naming more systematically in 
- package, classes, methods, variables. (started refactoring, some not done yet -- use "Refactor" option in Pycharm if renaming)
- Repo structure for test, docs, other? 
- distinguish "core" vs. "helper" functions? (e.g. sens, xval, quantile)
- Some weirdness can be straightened out: maybe SepiaData should not handle K/D but SepiaModel, consider separating
some items of model.num into model.num and model.info, be more clear about what will be there/always gets updated

##### Sensitivity analysis
- gSens: Sobol methodology

##### Kron enabling 
- setup, logLik, and predict
- predict is pretty complicated.

##### Categorical variate enabling
- setup, logLik, predict, covariance model
- this is relatively easy.

##### Additions to SepiaModelSetup
- lamVzGroups : functionality should be in the code, maybe just add to setup
- thetaconstraints option

##### fledging out the predict class
- Establish semantics for prediction calls and data structures

##### Analysis helpers
- Quantile estimation

##### Performance Profiling and Optimization
- Seems OK for now

##### Save/load models
- Pickle? Want to be sure loaded models will be compatible with new code

##### Examples, worked and documented 
- The other examples in "Examples"?

##### Development principles going forward
- maintenance
- management
- devs and community ownership


