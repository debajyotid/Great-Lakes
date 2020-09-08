### Great Lakes
![Image](https://www.greatlakes.edu.in/e-learning-programs/images/great-learning.jpg)
### Introduction
---
 This is a collection of the different Machine Learning & Deep Learning Projects undertaken by me. Some of these projects are part of the PGP AI-ML course from Great Lakes Institute of Management, to which I am currently enrolled to


### Index
|__Problem__|__Methods__|__Libs__|__Repo__|
|-|-|-|-|
|[Predicting Loan Defualt using Randomforest Classifier](#Predicting-Loan-Defualt-using-Randomforest-Classifier)|`Ensemble Techniques` |`KFold`, `sklearn.utils`, `RandomForestClassifier`, `sklearn.preprocessing`, `sklearn.preprocessing`, `sklearn.model_selection`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_LoanDefault_using_RandomForestClassifier.ipynb)|
|[Predicting onset of Parkinson's disease by analyzing voice sample using Ensemble Techniques](#Predicting-onset-of-Parkinson's-disease-by-analyzing-voice-sample-using-Ensemble-Techniques)|`Ensemble Techniques` |`DecisionTreeClassifier`, `RandomForestClassifier`, `PCA`, `sklearn.preprocessing`, `GridSearchCV`, `RandomizedSearchCV`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_Parkinsons_onset_analyzing_voicesamples_using_EnsembleTechniques.ipynb)|
|[Classifying vehicles by analysing silhouettes](#Classifying-vehicles-by-analysing-their-silhouettes)|`Unsupervised Learning` |`StandardScaler`, `SVM`, `PCA`, `GridSearchCV`, `Cross-Val`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Vehicle_Classification_analyzing_silhouettes.ipynb)|
|[Predicting mileage for city vehicles using Cluster Analysis](#Cluster-Analysis-on-Vehicle-data-for-vehicles-for-better-prediction-of-miles/gallon-figures-for-each-class-of-vehicle)|`Unsupervised Learning` |`KMeans`, `AgglomerativeClustering`, `LinearRegression`, `GridSearchCV`, 'sklearn.preprocessing'|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_mileage_using_Cluster_Analysis.ipynb)|
|[Campaign to sell Personal Loans](#Using-Supervised-Machine-Learning-techniques-to-create-a-successful-targetted-Perosnal-Loan-Campaign)|`Supervised Learning` |`LogisticRegression`, `KNeighborsClassifier`, `train_test_split`, `GridSearchCV`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Campaign_for_Personal_Loans.ipynb)|
|[Classifying patients based on their orthopdeic biomechanical features](#Patient-Classification-using-orthopaedic-biomechanical-features)|`Supervised Learning` |`KNeighborsClassifier`, `MinMaxScaler`, `train_test_split`, `metrics`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Classifying_Patients_using_biomechanical_features.ipynb)|
|[Building a Student Performance Prediction System](#Building-a-system-to-predict-Student's-performance-using-Regression-techniques)|`Supervised Learning` |`LogisticRegression`, `GaussianNB`, `train_test_split`, `seaborn`, `labelencoder`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Building_Student_Performace_Prediction_System.ipynb)|
|[Analyzing Cost of Insurance using Statistical Techniques](#Analyzing-Insurance-Cost-using-Statistical-Techniques)|`Hypothesis Testing` |`t-tests`, `Students t-Test`, `EDA`, `Anova`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Analyzing_Insurance_costs_using_Statistical_techniques.ipynb)|
|[Hypothesis Testing Questions](#Hypothesis-Testing-Questions)|`Hypothesis Testing` |`t-tests`, `ANOVA`, `Type-I & Type-II Errors`, `Chi-Squared Tests`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Hypothesis_Testing_Questions.ipynb)|

### Predicting Loan Defualt using Randomforest Classifier
---
Based on different attributes of a bank's customers, predict whether a customer will default or not, using different ML techniques like K-Fold Cross-Validation, RandomForestClassifier, etc. Also the model's performance is explained using a ROC-AUC plot

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_LoanDefault_using_RandomForestClassifier.ipynb)

#### Skills and Tools
K-Fold Cross-Validation, RandomForestClassifier, GridSearchCV, LabelEncoder, OneHotEncoder, ROC-AUC

### Predicting onset of Parkinson's disease by analyzing voice sample using Ensemble Techniques
---
Parkinson’s Disease (PD) is a degenerative neurological disorder marked by decreased dopamine levels in the brain. It manifests itself through a deterioration of movement, including the presence of tremors and stiffness. There is commonly a marked effect on speech, including dysarthria (difficulty articulating sounds), hypophonia (lowered volume), and monotone (reduced pitch range). Additionally, cognitive impairments and changes in mood can occur, and risk of dementia is increased.
Traditional diagnosis of Parkinson’s Disease involves a clinician taking a neurological history of the patient and observing motor skills in various situations. Since there is no definitive laboratory test to diagnose PD, diagnosis is often difficult, particularly in the early stages when motor effects are not yet severe.
Monitoring progression of the disease over time requires repeated clinic visits by the patient. An effective screening process, particularly one that doesn’t require a clinic visit, would be beneficial. Since PD patients exhibit characteristic vocal features, voice recordings are a useful and non-invasive tool for diagnosis. If machine learning algorithms could be applied to a voice recording dataset to accurately diagnosis PD, this would be an effective screening step prior to an appointment with a clinician.
The data & attributes information for this project is available at https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/ The data consists of those diagnosed with Parkinson Disease and those who do not.
Data Set Information:
This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.
The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.
This dataset is courtesy the below & maybe copyrighted by the same.
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering (to appear).

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_Parkinsons_onset_analyzing_voicesamples_using_EnsembleTechniques.ipynb)

#### Skills and Tools
DecisionTreeClassifier, RandomForestClassifier, PCA, sklearn.preprocessing, GridSearchCV, RandomizedSearchCV

### Classifying vehicles by analysing their silhouettes
---
The purpose of the case study is to classify a given silhouette as one of three different types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.

Four "Corgie" model vehicles were used for the experiment: a double-decker bus, Chevrolet van, Saab 9000 and an Opel Manta 400 cars.This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Vehicle_Classification_analyzing_silhouettes.ipynb)

#### Skills and Tools
K-fold Cross Validation, StandardScaler, SVM, GridSearchCV, PCA

### Classifying vehicles by analysing their silhouettes
---
The purpose of the case study is to classify a given silhouette as one of three different types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.

Four "Corgie" model vehicles were used for the experiment: a double-decker bus, Chevrolet van, Saab 9000 and an Opel Manta 400 cars.This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Vehicle_Classification_analyzing_silhouettes.ipynb)

#### Skills and Tools
K-fold Cross Validation, StandardScaler, SVM, GridSearchCV, PCA

### Cluster Analysis on Vehicle data for vehicles for better prediction of miles/gallon figures for each class of vehicle
---
The dataset was used in the 1983 American Statistical Association Exposition. The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 2 multivalued discrete and 4 continuous variables.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Predicting_mileage_using_Cluster_Analysis.ipynb)

#### Skills and Tools
KMeans, AgglomerativeClustering, LinearRegression, GridSearchCV

### Using Supervised Machine Learning techniques to create a successful targetted Perosnal Loan Campaign
---
This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans.
In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors).
A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.
The department wants to build a model that will help them identify the potential customers who have higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.
The dataset is readily available in Kaggle & has also been included in the repo. The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Campaign_for_Personal_Loans.ipynb)

#### Skills and Tools
LogisticRegression, KNeighborsClassifier, train_test_split, GridSearchCV

### Patient Classification using orthopaedic biomechanical features
---
In this assignment we look to classify whether a patient has started noticing onset of Rheumatoid Arthritis based on the biomechanical features like pelvic_incidence, lumbar_lordosis_angle, pelvic_radius, etc. The dataset has 2 parts: 1 having 2 classes-Normal/Abnormal, while the other having 3 classes-Normal/Spondylolisthesis/Hernia. The dataset is part of UCI Machine Learning repository. We use a popular supervised machine learning algorithn KNeighborsClassifier for our classification task. This algorithm works by classifying a datapoint to a particular class based on the proximity of each indvidual feature in the datapoint with other datapoints. The datapoints is assigned the class of the nearest datapoint(s). Number of nearest neighbours to be used for this learning is an extremely important hyper-parameter as is the method used to calculate the distance of a point from it's nearest neighbours.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Classifying_Patients_using_biomechanical_features.ipynb)

#### Skills and Tools
KNeighborsClassifier, MinMaxScaler, train_test_split, metrics

### Building a system to predict Student's performance using Supervised Machine Learning algorithms
---
The dataset consists of student achievements in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in Mathematics.
Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance

We use LogisticRegression & Gaussian Naive-Baye's Classifiers for calculating the probability of a student passing & predicting the same based on the calculation

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Building_Student_Performace_Prediction_System.ipynb)

#### Skills and Tools
LogisticRegression, GaussianNB, train_test_split, seaborn, labelencoder


### Analyzing Insurance Cost using Statistical Techniques
---
In the case of an insurance company, attributes of customers like Age, Gender, BMI, No. of Children, Smoking habits, etc. can be crucial in making business decisions. Hence, knowing to explore and generate value out of such data can be an invaluable skill to have.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Analyzing_Insurance_costs_using_Statistical_techniques.ipynb)

#### Skills and Tools
t-Test, Student's t-Test, ANOVA, EDA

### Hypothesis Testing Questions
---
In this assignment, we look at various statistical techniques like t-Tests, ANOVA, Chi-Square Tests, etc. and try answer various questions statistically using Hypothesis Testing

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Hypothesis_Testing_Questions.ipynb)

#### Skills and Tools
t-Tests, ANOVA, Type-I & Type-II Errors, Chi-Squared Tests
