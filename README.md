# CIFAR workshop on imablanced datasets

This repo contains the code to begin a workshop on CIFAR dataset with imbalanced classes.

To run it :

To install the dependencies for this project, please run the following command in your terminal:

```bash
pip install -r requirements.txt
```

Then :
```bash
python3 generate.py
```

This script will download CIFAR dataset and remove some data to make it imbalanced, by default class 3 is selected to be 1/10th of other classes.
It will also generate test_data.pkl which is required for evaluation of this project.

The jupyter notebook 
```check.ipynb``` is a simple example of how to check data and can be used as a boilerplate to start experiments. It also contains a minimal exaple on how to produce required output.
