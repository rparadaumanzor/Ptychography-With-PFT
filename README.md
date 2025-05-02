# Ptychography With PFT
 Application of the Partial Fourier Transform (PFT) for the ptychography problem

## Associated Publications

### Fast Partial Fourier Transforms for Large-Scale Ptychography
Full-Paper: available soon

Pre-Print: https://arxiv.org/abs/2408.03532

Please cite as

    @misc{parada2024fastpartialfouriertransforms,
         title={Fast Partial Fourier Transforms for Large-Scale Ptychography}, 
         author={Ricardo Parada and Samy Wu Fung and Stanley Osher},
         year={2024},
         eprint={2408.03532},
         archivePrefix={arXiv},
         primaryClass={math.NA},
         url={https://arxiv.org/abs/2408.03532
         }
         
## Set-up

Install all the requirments (designed for python 3.12.7)
```
pip install -r requirements.txt
```

## Experiments

### Non-Blind Ptychography
Large-Scale PIE Experiment
```
python PIE_driver.py
```

PIE Relative Errors Experiment
```
python nonblind_rel_err_experiments.py
```

### Blind Ptychography
Large-Scale ePIE Experiment
```
python ePIE_driver.py
```

ePIE Relative Errors Experiment
```
python blind_rel_err_experiments.py
```

