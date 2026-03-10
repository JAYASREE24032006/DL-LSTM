# EX-6 : DL- DEVELOPING A DEEP LEARNING MODEL FOR NER USING LSTM

#### NAME : R.JAYASREE
#### R.NO : 212223040074

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

An LSTM-based model for recognizing named entities is a type of neural network that uses Long Short-Term Memory (LSTM) layers to identify and classify proper names and entities within a text, such as person names, locations, organizations, dates, etc. It is commonly employed in Named Entity Recognition (NER) tasks because LSTMs are effective at capturing sequential dependencies and context within text. Typically, these models process tokenized input data, learn contextual representations, and output labels for each token indicating whether it belongs to a specific entity type. This approach improves the accuracy of extracting meaningful information from unstructured text data.

<img width="406" height="464" alt="image" src="https://github.com/user-attachments/assets/bf4a775a-c16e-4bb7-8389-832295d67cb1" />



## DESIGN STEPS
### STEP 1:
Load data, create word/tag mappings, and group sentences.

### STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3:
Define dataset and DataLoader for batching.

### STEP 4:
Build a bidirectional LSTM model for sequence tagging.

### STEP 5:
Train the model over multiple epochs, tracking loss.

### STEP 6:
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.


## PROGRAM:




### OUTPUT

<img width="705" height="518" alt="image" src="https://github.com/user-attachments/assets/994ffd47-9423-4ee8-bc8d-92b82f103fac" />


### Loss Vs Epoch Plot

Include your plot here

#### Sample Text Prediction
Include your sample text prediction here

## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
