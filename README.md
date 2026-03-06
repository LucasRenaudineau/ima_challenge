# ima_challenge
Image classification challenge implementation for Telecom Paris IMA205 course

## Dataset

The dataset is loaded by :
- loading a kaggle API token
- importing the dataset using the command `kaggle competitions download -c ima205-challenge-2026`

It is situated in the folder ./IMA205-challenge and inside are the train and test folder as well as test_metadata.csv and train_metadata.csv

## Data aumentation

I chose to flip horizontaly/vertically and to rotate the images for data aumentation. This is because cells have no orientation, and are all centered (so I don't want to translate them). I could also in the future add zooming, blurring and noise but I want to keep it simple for now on.
I saw on a few tests that a machine looks like it added lines to the sides of the square, meaning rotating added weird lines in diagonal. However, I strongly think the sides won't be considered much by the NN and it won't have any impact.

## Choice of the backbone

To choose the backbone, I took a look at this website : `https://keras.io/api/applications/`

I then chose a relatively efficient model. I can be sure it was trained on image net with the parameter weights="imagenet" when importing the model.

## Classifying layers

(To do)
