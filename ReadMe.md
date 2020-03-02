1. Process data
python extra_feature.py.

All data is in data/ dictionary. After running the script, the generated processed feature files are placed in the same directory data/.

2. Train model and predict 
I have used two methods to train model, SVM and RandomForest.

2.1 SVM:
SVM: The svm method is currently one of the best single model performance methods for classification problems and is widely used. Therefore, it is intended to use the svm method to train the classification model.

The svm files are located under the SVM / folder.
python svm_method.py, the svm model is trained and the results of the prediction data named SVM_output.csv are generated.

python svm_test.py, We can get the accuracy of cross-validation on the training data set. It can be seen from the accuracy rate that the svm method does not achieve ideal classification performance on this problem, and the accuracy rate is not high.


2.2 Random Forest:
RandomForest: The random forest method is an ensemble method after ensemble learning of decision trees, which further develops the performance of decision trees. Through the integrated thought of bagging, Random Forest can learn more parameters and further improve the learning performance. In recent years, it has achieved good results in many classification problems. Therefore, in this problem, the performance of svm is not ideal. Use random forest method to achieve good performance.

The randomforest files are located under the RandomForest / folder.

python make_random_forest_prediction.py, the randomforest model is trained and the results of the prediction data named RF_Output.csv are generated.

python random_forests.py, we can get the accuracy of cross-validation on the training data set. The accuracy of the random forest method cross-validation is 0.8037, and this performance can be ranked around 15 in the kaggle ranking, which is already a good result.
