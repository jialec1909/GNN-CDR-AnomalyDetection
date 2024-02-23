# Transformer-CDR-AnomalyDetection
This is anomaly detection targeted processing on CDR datasets.

To reach CDR datasets, please reach out https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV.

Instructions:
1. To preprocess the data structure, please firstly create a folder "`/datasets`" under the root folder of project folder for the datasets.
2. To observe how the data distribution looks like, please run utils/observe.py.
3. To preprocess the data structure for the transformer model, please run utils/preprocess.py.

   ```python utils/preprocess.py --if_cover --if_transformer```
4. To load data and train the model, please  locate to the scripts folder and run train.py.

   ```cd scripts```

   ```python train.py```
5. If your environment is not compatible with the current version of the code, please install the required packages by running the following command.

   ```pip install -r requirements.txt```
6. To avoid the error related to unrecognized CDR path, please locate to the root folder and install the package through:

   ```pip install -e .```