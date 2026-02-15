# 🔤 Letter Recognition -- Multi-Class Classification

## Problem Statement

The objective of this project is to build and compare multiple machine
learning classification models to recognize capital English alphabets
(A--Z) from numerical image-based features.

The dataset represents a multi-class classification problem consisting
of 26 classes corresponding to each uppercase English letter.

The goals of this project are:

-   To implement multiple classification algorithms
-   To evaluate each model using standard performance metrics
-   To compare model performances
-   To deploy an interactive Streamlit web application

------------------------------------------------------------------------

## Dataset Description

-   Dataset Name: Letter Recognition Dataset
-   Source: Kaggle
-   Total Instances: 20,000
-   Number of Features: 16 numerical features
-   Target Variable: `letter` (A--Z)
-   Problem Type: Multi-class Classification (26 classes)

Each instance contains statistical attributes extracted from images of
handwritten capital letters.

The dataset satisfies the assignment requirements:

-   Feature size > 12
-   Instance size > 500

------------------------------------------------------------------------

## Machine Learning Models Implemented

The following six classification models were implemented on the same
dataset:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest (Ensemble Model)
6.  XGBoost (Ensemble Model)

------------------------------------------------------------------------

## 📈 Model Performance Comparison

  -------------------------------------------------------------------------------
  ML Model Name      Accuracy   AUC Score Precision   Recall   F1 Score MCC Score
  ------------------ ---------- --------- ----------- -------- -------- ---------
  Logistic           0.7742     0.9805    0.7748      0.7730   0.7728   0.7653
  Regression                                                            

  Decision Tree      0.8828     0.9389    0.8828      0.8825   0.8824   0.8781

  KNN                0.9433     0.9963    0.9442      0.9430   0.9432   0.9410

  Naive Bayes        0.6522     0.9573    0.6641      0.6512   0.6479   0.6391

  Random Forest      0.6730     0.9617    0.6998      0.6709   0.6621   0.6624

  XGBoost            0.9075     0.9977    0.9105      0.9071   0.9078   0.9039
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📌 Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name             Observation about Model Performance
  ------------------------- ---------------------------------------------
  Logistic Regression       Performs moderately well but struggles with
                            complex non-linear patterns in multi-class
                            classification.

  Decision Tree             Performs better than Logistic Regression but
                            may overfit due to hierarchical splitting.

  KNN                       Achieves very high accuracy and AUC. Performs
                            extremely well because similar letters share
                            similar feature distributions.

  Naive Bayes               Shows comparatively lower performance.
                            Independence assumption between features
                            reduces effectiveness for this dataset.

  Random Forest             Performs better than a single Decision Tree
                            but does not outperform KNN or XGBoost in
                            this implementation.

  XGBoost                   Achieves the best overall performance with
                            high Accuracy, AUC, and MCC. It effectively
                            captures complex relationships among
                            features.
  -----------------------------------------------------------------------
##  Conclusion

Among all the implemented models, KNN and XGBoost achieved the highest
performance on the Letter Recognition dataset.

Ensemble and instance-based learning approaches significantly outperform
linear and probabilistic classifiers for this multi-class classification
problem.

------------------------------------------------------------------------
