# Funnels

This repository contains the code needed to reproduce the experiments from the paper:


> Funnels: Exact maximum likelihood with dimensionality reduction.

## Usage
The environment variable REPOROOT must be set to point to the top level of the repository.
Once the container has been built

## Dependencies
See `requirements.txt` for the necessarypip packages to be installed into the container built using `.gitlab-ci.yml`.

## Data

The [preprocessed datasets for MAF experiments](https://zenodo.org/record/1161203#.YaomofHMKji) was used for the density
estimation comparisons.

All image experiments have the args saved in json, with cifar-10 and imagenet funnel experiments run
with `--model 'funnel_conv'`

For the anomaly detection experiments the defaults can be found in `experiments/image_configs/AD_config.json`. With
these defaults set across experiments the different models can be run as follows:

VAE
```angular2html
python image_generation.py --model 'VAE' --latent_size 4
python image_generation.py --model 'VAE' --latent_size 16
```

F-NSF
```angular2html
python image_generation.py --model 'funnel_conv_deeper' --latent_size 4
python image_generation.py --model 'funnel_conv_deeper' --latent_size 16
```

F-MLP
```angular2html
python image_generation.py --model 'funnelMLP' --levels 4
python image_generation.py --model 'funnelMLP' --levels 3
```

NSF
```angular2html
python image_generation.py --model 'glow'
```