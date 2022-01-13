# AML-ADDA

# UniTN Applied Machine Learning project 2020/2021

Implementation of Adversarial Discriminative Domain Adaptation proposed by Eric Tzeng et al. in [Tzeng, Eric, et al. "Adversarial discriminative domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1702.05464)

### Datasets and Git Large Files

All datasets used in the project (MNIST, USPS, SVHN) come with cloning. However, given that SVHN dataset exceeds GitHub storage limits, you can either download the dataset on a local folder (in ```datasets/svhn```, ```.mat``` format required, url: http://ufldl.stanford.edu/housenumbers/) or, alternatively, install git-lfs on your machine. Steps:
+ ```$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash```
+ ```$ sudo apt-get install git-lfs```
+ ```$ git-lfs install```  

And then clone the project. 

### Virtual environment setup and requirements

In order to to execute the project after the cloning, it is suggested to create a virtual environment and install the required modules.  
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

The program will check automatically if a GPU is available for the task. If not, the CPU will be used to complete the adaptation.

### Credentials

To store run results in a WandB instance, edit the template file stored in `adda/settings/wandb_settings.py.template` with the desired credentials, as well as the `run.py` file with the proper run names (and entities).

### Parameters

Every run comes with a source and a target datasets, as long with a given amount of epochs each for training, adaptation and test set. Every parameter could be specified through command line parameters when running `run.py` file. Here is the list of the available ones:

* `-source`: specify the source dataset (options: MNIST, USPS, SVHN); default: MNIST
* `-target`: specify the target dataset (options: MNIST, USPS, SVHN); default: USPS
* `-phase`: specify which step(s) you want to run (options: 1 (Pre-training), 2 (Adversarial Adaptation), 3 (Testing)); default: 1
* `-e_tr`: specify the number of epochs for training step; default: 10
* `-e_ad`: specify the number of epochs for adaptation step; default: 50
* `-e_te`: specify the number of epochs for test step; default: 20
* `-bs`: specify the batch size; default: 50
* `-sample_tr`: specify if you want to use a sample of the source dataset (available only for MNIST and USPS); default: False
* `-sample_ad`: specify if you want to use a sample of the target dataset (available only for MNIST and USPS); default: True
* `-sample_te`: specify if you want to use a sample of the dataset in the test phase (available only for MNIST and USPS); default: False
* `-model_arch`: creates an image containing the architecture of the selected model (options: LeNetEncoder, LeNetClassifier, Discriminator, Phase1Model)
* `-wandb`: specify if you want to log on a WandB instance; default: True

## Results

| Method | MNIST &rarr; USPS | USPS &rarr; MNIST | SVHN &rarr; MNIST |
| :------: | :-------------: | :-------------: | :-------------: |
| Source only | 7.52 | 49.80 | 53.00 |
| ADDA | 13.80 | 50.70 | 53.00 |