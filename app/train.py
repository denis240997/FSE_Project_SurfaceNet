import numpy as np
import sys
import os
import params

sys.path.append("./utils")
#import adapthresh

# import main_train

if __name__ == "__main__":

    print("\ncurrent mode *** {} ***\n".format(params.whatUWant))
    
    if params.whatUWant is 'reconstruct_model':
        print('This is a script for training a model. For evaluation and reconstruction, run reconstruction.py')
    else:
        # main_train.train() # pass
        pass
