# train a dominant detector
from pygod.detector import DOMINANT

model = DOMINANT(num_layers=4, epoch=20)  # hyperparameters can be set here
model.fit(train_data)  # input data is a PyG data object

# get outlier scores on the training data (transductive setting)
score = model.decision_score_

# predict labels and scores on the testing data (inductive setting)
pred, score = model.predict(test_data, return_score=True)