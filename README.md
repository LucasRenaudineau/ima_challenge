# ima_challenge
Image classification challenge implementation for Telecom Paris IMA205 course

## Dataset

The dataset is loaded by :
- loading a kaggle API token
- importing the dataset using the command `kaggle competitions download -c ima205-challenge-2026`

It is situated in the folder ./IMA205-challenge and inside are the train and test folder as well as test_metadata.csv and train_metadata.csv

## Choice of the backbone

To choose the backbone, I took a look at this website : `https://keras.io/api/applications/`

I then chose a relatively efficient model. I can be sure it was trained on image net with the parameter weights="imagenet" when importing the model.
