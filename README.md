# Transformer-CDR-AnomalyDetection
This is anomaly detection targeted processing on CDR datasets.

To reach CDR datasets, please reach out https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV.

Instructions:
1. To preprocess the data structure, please firstly create a folder "`/datasets`" under the root folder of `CDR/CDR/` folder for the datasets.
2. To observe how the data distribution looks like, please run utils/observe.py.
3. To preprocess the data structure for the transformer model, please run utils/preprocess.py.

   ```python utils/preprocess.py --if_cover --if_transformer```
4. To load data and train the model, please  locate to the scripts folder and run train.py.

   ```cd scripts```

   ```python train.py```