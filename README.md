# AML-ADDA

# UniTN Applied Machine Learning project 2020/2021

Implementation of Adversarial Discriminative Domain Adaptatio proposed by Eric Tzeng et al. in [Tzeng, Eric, et al. "Adversarial discriminative domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1702.05464)

### Virtual environment setup and requirements

In order to to execute the project, it is suggested to create a virtual environment and install the required modules.  
To create a virtual environment in a UNIX system, type in a terminal:

```
# Use python or python3 (depending on the recognised command)
python3 -m venv ./venv
```

Then, activate your virtual environment:

```
source venv/bin/activate
```

Finally, install all the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To perform a sample run, from the root of the project run the following command in a terminal:

```
python3 run.py
```

### Credentials

To store run results in a WandB instance, edit the template file stored in `adda/settings/wandb_settings.py.template` with the desired credentials, as well as the `run.py` file with the proper run names (and entities).

### Parameters

TODO: parametrizziamo?

Every run comes with a source and a target datasets. They have to be specified in the `run.py` file, together with the desired number of epochs for each step.

## Results

TODO.
