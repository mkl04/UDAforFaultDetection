# Unsupervised Domain Adaptation for Fault Detection

Code use for training and evaluation of experiments for my dissertation.


## Objectives

- Implement and assess UDA techniques for the semantic segmentation of seismic faults.
- Conduct both qualitative and quantitative comparisons on real field data.
- Compare with fine-tuning approach.


## Dataset

- Public real labeled dataset from the **Thebe** Gas Field in the Exmouth Plateau of the Carnarvan Basin on the NW shelf of Australia ([Yu An et al., 2021a](https://doi.org/10.1016/j.dib.2021.107219)).

- Public synthetic labeled dataset from **FaultSeg3D** ([Xinming Wu et al., 2019](https://github.com/xinwucwp/faultSeg)).

- Public real dataset **Netherlands F3** ([Xinming Wu et al., 2019](https://drive.google.com/drive/folders/1aw_f29yXloAeLclOvIshfuBukaOVQAJ1)).


## Installation


```
pip install -r requirements.txt
```


## Experiments

- Download all the datasets and place them as follows:

```
datasets/
|
└───Thebe
│   │   faulttrain1.npy
│   │   faulttrain2.npy
│   │   ...
│   │   faultval1.npy
│   │   faultval2.npy
│   │   seistrain1.npy
│   │   seistrain2.npy
│   │   ...
│   │   seisval1.npy
│   │   seisval2.npy
│   │
│   └───test
│       └───annotation
│       └───seismic
|
└───FaultSeg3D
│   └───train
│   |   └───fault
│   |   └───seis
│   └───val
│       └───fault
│       └───seis
|
└───SubF3
    │   gxl.dat
```

- Create a file `config.py`, where is defined the DATAFOLDER and SAVEFOLDER variables.

- Before training the model, perform the generation of patches according to the desired protocol configuration: `preprocessing/`.

- To train models, review notebooks on `training/`.

- For testing, review notebooks on `evaluation/`


## Final Results

The following table shows the results of the metrics for our best model using a tecnique with its respective best network when using FaultSeg as ''source domain'' and Thebe as target domain.

| Technique | Model | IOU | F1-score |
|:----: | :----: | :----: | :----: |
| No-Adaptation |Atrous U-Net     | 54.16 | 70.19  |
| MMD | U-Net    | 54.30 | 70.29  |
| DANN | Atrous U-Net       | 53.57 | 69.69  |
| FDA | Atrous U-Net  | **57.05** | **72.58** |

For better inspection, the final version of the dissertation will be posted soon.