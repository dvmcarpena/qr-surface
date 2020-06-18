# TFG Eng. Informàtica - David Martinez Carpena

## Introduction

This is the implementation of the TFG of Eng. Informàtica by 
David Martinez Carpena.
The implementation consist in several Python modules.
The main one, `tfg`, is the code used to run the experiments 
explained in the TFG. 

Inside the module `tfg` there is a submodule name `qrsurface`, which 
is a library for building custom modular QR Code decoders.

## How to install the environment

The environment is specified in `requirements.txt` file and in the 
`Pipfile` and `Pipfile.lock`.
All the Python packages can be installed by any of these two commands:
```shell
pip install -r requirements.txt
```
or
```shell
pipenv install
```

We have tested this code only with Python 3.7.2.

## How to use this module

This module has a CLI interface offered through the module `tfg` 
main function. 
The CLI provides two subcommands: `process` and `results`.

The `process` subcommand can run the pipeline for comparing the 
different correction methods to all the images in the datasets 
selected. 

```shell
python -m tfg process
```

There are several options to configure the purpouse of the execution.
For example, we can pass a option to show the difference between 
this processing and the last saved.
We can choose to save this processing as the last one.
Or we can choose if we want to plot and in which situations.

Once we executed and we have at least one processing saved, we can 
run the `results` subcommand.
This one is for showing meaningful information about the experiments 
done.
It will generate data in the standard output, tables in CSV and 
some plots, which will be save in the folder `data/results`.

```shell
python -m tfg results
```
