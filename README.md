# Real-time bias-aware data assimilation


Open-source repository for advanced data assimilation techniques, focusing on bias correction in real-time thermoacoustic systems and beyond. 


--- 

## 🚀 Getting started

1. **Prerequisites**

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. **Clone the Repository**
```
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd yourproject
```

3. **Create and activate the Conda Environment** 
```
conda env create -f environment.yml
conda activate real-time-da
```

4. Install the package in editable mode 
```
pip install -e .
```

5. (Optional) Run the tests and save the output onto a text file
```
pytest test_tutorials.py > test_output.txt
```

Checkout the [Tutorials folder](https://github.com/andreanovoa/real-time-bias-aware-DA/tree/main/tutorials), which includes several jupyter notebooks aiming to ease the understanding of the repository.



---


## 🌟 What is available?
   Data assimilation methods [`src.data_assimilation`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/src/DA.py)
   * EnKF - ensemble Kalman filter
   * EnSRKF - ensemble square-root Kalman filter
   * rBA-EnKF - regularized bias-aware EnKF
   
   Physical models [`src.models_physical`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/src/models_physical.py)
   * Rijke tube model (dimensional with Galerkin projection)
   * Van der Pols
   * Lorenz63
   * Azimuthal thermoaocustics model
   
   Data-driven models [`essentials.models_datadriven`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/models_datadriven.py)
   * ESN_model -- Echo state network as a frorecasting tool
   * POD-ESN -- Combines Proper Orthogonal Decomposition (POD) and ESN_model
    
   Bias estimators[`src.bias`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/src/bias.py)
   * Echo State Network
   * NoBias


---
## 📂 Structure
```

.
├── data/              # (Generated) Data files
├── docs/              # Documents and media used in the notebooks
├── results/           # Any results generated will be stored here
├── scripts/           # Includes the files which run by themselves (e.g., main files, tutorials)
│   ├── mains/
│   ├── post_process/ 
│   └── tutorials/ 
├── src/               # Source code including all the objects, classes and functions required in scripts
│   ├── ML_models/
│   │   │── EchoStateNetwork.py
│   │   └── POD.py
│   ├── bias.py
│   ├── create.py   
│   ├── DA.py
│   ├── model.py
│   ├── models_datadriven.py
│   ├── models_physics.py
│   ├── plot_fns.py
│   ├── run.py
│   └── util.py
├─ tests_tutorials.py    # Unit tests
├─ environment.yml       # Conda environment definition
├─ setup.py              # Python package setup
└─ README.md             # This file

```



---

## 📚 Main publications and presentations

##### Journal papers

- [x] Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic instabilities. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2024.1052) |  🏷️ [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0). 
- [x] Nóvoa, Racca & Magri (2023). Inferring unknown unknowns. Computer Methods in Applied Mechanics and Engineering. [Published paper](https://doi.org/10.1016/j.cma.2023.116502) | [_Legacy_ repository](https://github.com/MagriLab/rBA-EnKF).
- [x] Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2022.653) | [_Legacy_ repository](https://github.com/MagriLab/Real-time-TA-DA).

##### Conference papers and proceedings
- [x] Nóvoa & Magri (2025). Online model learning with data-assimilated reservoir computers. [Preprint](https://doi.org/10.48550/arXiv.2504.16767) | 🏷️ [v1.1](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.1).
- [x] Nóvoa & Magri (2024). Real-time digital twins of multiphysics and turbulent flows. [Paper](https://web.stanford.edu/group/ctr/ctrsp24/ii11_NOVOA.pdf).
- [x] Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. In_ 51st International Congress and Exposition on Noise Control Engineering. [Paper](https://az659834.vo.msecnd.net/eventsairwesteuprod/production-inconference-public/808b4f8c38f944d188db8a326a98c65c). | [_Legacy_ repository](https://github.com/MagriLab/IN22-Bias-aware-TADA).

##### PhD theses
- [x] Nóvoa (2024). Real-time data assimilation in nonlinear dynamcal systems. University of Cambridge. [Thesis](https://doi.org/10.17863/CAM.113001). 

##### Conference presentations _(incomplete list)_
- **APS-DFD 2024, Salt Lake City:** [Abstract](https://meetings.aps.org/Meeting/DFD24/Session/C02.14) | [Poster](https://github.com/user-attachments/files/17966063/APS-poster-final-version.pdf).
- **APS-DFD 2023, Washington DC:** [Abstract](https://meetings.aps.org/Meeting/DFD23/Session/L30.8).
- **EFMC14 2022, Athens:** [Abstract](https://euromech.org/conferences/proceedings.htm).
- **APS-DFD 2022, Phoenix:** [Abstract](https://meetings.aps.org/Meeting/DFD22/Session/G12.4).

--- 
## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request. For questions or collaborations, please reach out to [A. Nóvoa](https://scholar.google.com/citations?user=X0TjtAgAAAAJ&hl=en).
