# Wu_SailencyMap_2025

This repository accompanies the following article:

**WU, R (2025). Preference-Independent Saliency Map in the Mouse Superior Colliculus. Communications Biology.**

  
## Contents of the repo
This repo contains the code needed to reproduce the figures in the paper.

* `nm_analysis.py`: the main code for generating figures. Running it generates all figures. Figures can be saved to `/figures/` by setting save_fig = True.
* `nm_utils.py`: functions that are used in `nm_analysis.py`

## How to find code for a specific figure panel
* Search for "nm_fig_nX" in `nm_analysis.py`, where "n" is 1,2,3..., "X" is A,B,C,...
