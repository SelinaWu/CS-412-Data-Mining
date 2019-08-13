There are four Python script files used in this project.
    utils.py: contains various class definitions and auxiliary functions
    DataPreprocess.py: can be executed with command 'python3 DataPreprocess.py'. Perfrom data imputations on both training and testing datasets.
    DataClassification.py: can be exeucted with command 'python3 DataClassification.py'. Perform data classifications based on imputed datasets.
    PredictDataset.py: can be exeucted with command 'python3 PredictDataset.py'. This script takes 'testing_result.csv' (the complete, or imputed, testing set) as input and uses 'trained_trees.pkl' file (contains the trained model) so that the classification can be performed on this new complete testing set very quickly.



External libraries directly used:
    numpy==1.14.2
    pandas==0.22.0
    scipy==1.0.1

Other external library dependencies:
    cycler==0.10.0
    kiwisolver==1.0.1
    matplotlib==2.2.2
    mkl-fft==1.0.0
    mkl-random==1.0.1
    pyparsing==2.2.0
    python-dateutil==2.7.2
    pytz==2018.4
    six==1.11.0
    tornado==5.0.2
