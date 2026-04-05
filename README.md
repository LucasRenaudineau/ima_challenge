# ima_challenge
Image classification challenge implementation for Telecom Paris IMA205 course

## Dataset

The dataset is downloaded by :
- loading a kaggle API token
- importing the dataset using the command `kaggle competitions download -c ima205-challenge-2026`

It is situated in the folder ./IMA205-challenge and inside are the train and test folder as well as test_metadata.csv and train_metadata.csv
80% of the data is used for training and 20% for validation.

To see how my code loads the dataset, see load.py

## Data aumentation

I chose to flip horizontaly/vertically and to rotate the images for data aumentation. This is because cells have no orientation, and are all centered (so I don't want to translate them). I could also in the future add zooming, blurring and noise but I want to keep it simple for now on.
I saw on a few tests that a machine looks like it added lines to the sides of the square, meaning rotating added weird lines in diagonal. However, I strongly think the sides won't be considered much by the NN and it won't have any impact.

To see how my code uses data aumentation, see data_augmentation.py

## Choice of the backbone

To choose the backbone, I took a look at this website : `https://keras.io/api/applications/`

I then chose a relatively efficient model. I can be sure it was trained on image net with the parameter weights="imagenet" when importing the model.
I first chose ResNET for sake of simplicity, and then when my code began to fully work, I then chose EfficientNetV2 which is slightly better for the same size.

To see how the backbone is downloaded, see model.py

## Classifying layers

For the classifying layers, I first tried with something very simplistic (3 layers). It was clearly not enough. I then increased a bit the size. Now, the architecture is as follows :

- A 512 Dense layer
- A 128 Dense layer
- A 13 Dense layer

They have dropOut, normalization and a relu activation function.

To see the full code for classifying layers, see model.py

## Training

For the training phase, I first tried to freeze the backbone and train the classifying layers and then unfreeze and train everything. I did not get improvements in the first phase. One of my suspicions is that I either had not enough layers to train, or that just imagenet's training is too far from recognising good patterns in a cell. So now I skipped this part and I directly fully train all the weights together. Any epoch is saved in ./outputs/

The code for training is in training.py

## Testing

The code for testing (evaluating a model on the test set and creating the .csv) is in evaluates.py
