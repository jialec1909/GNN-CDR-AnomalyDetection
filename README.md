# GNN-CDR-AnomalyDetection
This is anomaly detection targeted processing on CDR datasets.

To reach CDR datasets, please reach out https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV.

Instructions:
1. To preprocess the data structure, please create a folder "/datasets" under the same root folder with this project for the dataset and run preprocess.py.
2. To convert the dataset to data object that fits py-geometric models, please run batch_dataload.py, this will read the CDR data files for every day and transfer the datasets to a list of data objects, which is suitable for pyG inputs.
