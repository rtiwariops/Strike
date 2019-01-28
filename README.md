# Strike
<img src="https://github.com/rtiwariops/Strike/blob/master/image/fraud_detection.jpg" width="50" height="50">
Strike is a Credit Card Fraud Detection Project that we will be developing for demonstration purposes showing unsupervised machine learning using anomaly detection from sklearn package.

Download the dataset either from Kaggle or from [here](https://s3-us-west-2.amazonaws.com/strikedataset/creditcard.csv). Notebook with complete code can be found [here](https://github.com/rtiwariops/Strike/blob/master/strike_credit_card_Fraud_detection.ipynb).

## Data Description
### Context
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

### Content
The datasets contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### Acknowledgements
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

Please cite: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

### CSV dataset
CSV dataset is 144Mb structured dataset from Kaggle. Shoutout to Kaggle on getting us this valuable information.
#### Columns
Time: Number of seconds elapsed between this transaction and the first transaction in the dataset<br />
V1: may be result of a PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28)<br />
V2<br />
V3<br />
V4<br />
V5<br />
V6<br />
V7<br />
V8<br />
V9<br />
V10<br />
V11<br />
V12<br />
V13<br />
V14<br />
V15<br />
V16<br />
V17<br />
V18<br />
V19<br />
V20<br />
V21<br />
V22<br />
V23<br />
V24<br />
V25<br />
V26<br />
V27<br />
V28: abc<br />
Amount: Transaction amount<br />
Class:<br />
  1 for fraudulent transactions,<br />
  0 otherwise<br />
