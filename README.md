# Line_Decryption
### Predict which text is the line taken from

The decryptor_source.py is a python code that uses a deep learning architecture to train on the encrypted
lines from 12 texts. 

------------------------

This code requires keras and numpy modules.

Running the code:
        $ python decryptor_source.py
        
        
If you would like to use already trained weights choose n when prompted and make sure the file 'weights'
is in the same directory as code.

-------------------------

A quick exploration into the encrypted lines reveals that most of the lines are 416 letters in length and the whole training set only contains all the 26 english letters with no missing or corrupt data. In order for the deep learning architecture to work, the code fragments each line into chunks of 416 letters. If the length of the whole line is less than 416 the code pads the fragment with '0' (string zero). If the line is longer than 416 letters the code chops the line into as many fragments with 416 letters as possible and pads the last remaining fragment with '0' to make it 416-long. Then it distributes the labels properly for all fragments. About 48000 fragments are created this way.

For evaluating the model the code uses the last 10000 fragments only.

For prediction the model does the same fragmentation over the test set. In order for the model to predict on lines longer than 416 it votes between the maximum probabilities from fragments produced by each line.

The accuracy of the model from the train test split is about 91%.

This model is a first step and could be improved for speed and accuracy.

Bahman Roostaei.
