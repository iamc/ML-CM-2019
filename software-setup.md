---
layout: page
title: Software Setup
subtitle: 
---

For the School we will be using Python for the code examples. The packages needed are:

* Matplotlib
* PyTorch
* scikit-learn
* TensorFlow
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq) *(Optional, see instructions in the package webpage)*

and their installation is very easy using [Anaconda Python Distribution](https://www.anaconda.com/distribution/). See instructions below:

1. [Anaconda Installation](#anaconda-installation)
2. [Global Anaconda installation](#global-anaconda-installation)
3. [Alternative installation in a new conda environment](#alternative-installation-in-a-new-conda-environment)

{: .box-note}
![Work in progress](img/warn.png) Please bring all the software pre-installed to the School. The download size is ~1GB and we will collapse the WiFi connection if everybody tries to install it on arrival.

If you are experienced using Anaconda you can use the following [ML-School-environment.yml](https://raw.githubusercontent.com/iamc/ML-CM-2019/master/ML-School-environment.yml) environment file.


# Anaconda Installation

By far the easiest way to have everything installed is using [Anaconda Python Distribution](https://www.anaconda.com/distribution/). Anaconda installation itself is a very easy process: just follow the [installation instructions](https://docs.anaconda.com/anaconda/install/) for your platform in case you don't have it already installed.

Once you have Anaconda installed, open a terminal (or an `Anaconda Prompt` from the Start Menu if you are using Windows) and install the required packages using the `conda` installer as described below.


# Global Anaconda installation

Here we will install all the sofware so that it can be used everytime you use Anaconda Python in this computer. See below for installation into a conda environent.


### PyTorch

    conda install pytorch-cpu torchvision-cpu -c pytorch

For MacOs you may have to remove the `-cpu`part. Check the PyTorch [official installation instructions](https://pytorch.org/get-started/locally/)


### Matplotlib, scikit-learn, TensorFlow

    conda install ipython matplotlib scikit-learn tensorflow

Where we also install IPython for convenience.

For more information about TensorFlow installation (eg. GPU enabled versions) see [here](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)


# Alternative installation in a new conda environment

Alternatively, we can install eveything inside a new conda environment using the provided [ML-School-environment.yml](https://raw.githubusercontent.com/iamc/ML-CM-2019/master/ML-School-environment.yml) environment file. This will create a new environment named `ML-School` and install all the requiered software within it. Just download the [file](https://raw.githubusercontent.com/iamc/ML-CM-2019/master/ML-School-environment.yml) and:

    conda env create -f ML-School-environment.yml 

This will take a while depending on your computer and internet connection speed.

Once the environment is created you can activate it with:

    conda activate ML-School

and deactivate it with:

    conda deactivate

Should you want to remove it, type:

    conda remove --name ML-School --all


Of course all this can also be done through the `Anaconda Navigator`, should you be more accustomed to it.


