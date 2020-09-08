### Great Lakes
![Image](https://www.greatlakes.edu.in/e-learning-programs/images/great-learning.jpg)
### Introduction
---
 This is a collection of the different Machine Learning & Deep Learning Projects undertaken by me. Some of these projects are part of the PGP AI-ML course from Great Lakes Institute of Management, to which I am currently enrolled to


### Index
|__Problem__|__Methods__|__Libs__|__Repo__|
|-|-|-|-|
|[Classifying patients based on their orthopdeic biomechanical features](#Patient Classification using orthopaedic biomechanical features)|`Supervised Learning` |`KNeighborsClassifier`, `MinMaxScaler`, `train_test_split`, `metrics`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Classifying_Patients_using_biomechanical_features.ipynb)|
|[Analyzing Cost of Insurance using Statistical Techniques](#Analyzing Insurance Cost using Statistical Techniques)|`Hypothesis Testing` |`t-tests`, `Students t-Test`, `EDA`, `Anova`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Analyzing_Insurance_costs_using_Statistical_techniques.ipynb)|
|[Hypothesis Testing Questions](#Hypothesis Testing Questions)|`Hypothesis Testing` |`t-tests`, `ANOVA`, `Type-I & Type-II Errors`, `Chi-Squared Tests`|[Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Hypothesis_Testing_Questions.ipynb)|

### Patient Classification using orthopaedic biomechanical features
---
In this assignment we look to classify whether a patient has started noticing onset of Rheumatoid Arthritis based on the biomechanical features like pelvic_incidence, lumbar_lordosis_angle, pelvic_radius, etc. The dataset has 2 parts: 1 having 2 classes-Normal/Abnormal, while the other having 3 classes-Normal/Spondylolisthesis/Hernia. The dataset is part of UCI Machine Learning repository. We use a popular supervised machine learning algorithn KNeighborsClassifier for our classification task. This algorithm works by classifying a datapoint to a particular class based on the proximity of each indvidual feature in the datapoint with other datapoints. The datapoints is assigned the class of the nearest datapoint(s). Number of nearest neighbours to be used for this learning is an extremely important hyper-parameter as is the method used to calculate the distance of a point from it's nearest neighbours.

Repo Link: [Click](https://nbviewer.jupyter.org/github/debajyotid/AI-ML-Projects/blob/master/Classifying_Patients_using_biomechanical_features.ipynb)

#### Skills and Tools
KNeighborsClassifier, MinMaxScaler, train_test_split, metrics


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
