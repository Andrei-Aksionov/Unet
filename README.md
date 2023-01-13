<h1 align="center">Welcome to U-Net implementation in PyTorch</h1>

<p align=center><img src="references/UNET_architecture.png"></p>

> Implementation of U-Net architecture in PyTorch that achieves on 'Carvana' dataset accuracy of 96.84 and dice score of 92.66 after training for 3 epochs.

## Install

As this project uses pyproject.toml - [poetry](https://python-poetry.org/docs/) has to be installed.

Also take a look at the required python version (described in **pyproject.toml** file).

In order to install all required packages run this command (when you are in the folder with pyproject.toml file).

```sh
poetry install
```

## Usage

### Parameters

All parameters like model architecture (list of features for each U-Net block) or folder paths where the data is stored can be changed in 'src/config/hyperparameters.py' file.

### Data

In order to train model the first step will be to provide data for training and evaluation. As this project is configured data should be stored in 'data/raw' folder with such structure:

- train_images
- train_masks
- val_images
- val_masks

Name of the corresponding mask should be the same as image name but with '_mask' postfix. For example: in 'train_images' folder there is a file '1.jpg' and in 'train_masks' folder the corresponding masks should have name as '1_mask.gif'.

The data that was used for this project can be download from [Kaggle competition](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).

### Training the model

After data is prepared we can start training by executing this command in terminal:

```sh
python src/model/train.py
```

### Model's checkpoint

In folder 'models' one can find checkpoint of the model with the highest accuracy score.

***

## Additional: git pre-commit hook

In order to run `black` formatter and `flake8` linter before each commit you need to add them into `.git/hooks` folder either manually or with helper script:

```bash
sh .add_git_hooks.sh`
```

This script will put `pre-commit` file into `.git/hooks` folder of the project and make it executable.
