# ML Assignment 2: Adult Income Classification

## a. Problem Statement
The objective of this assignment is to implement five different machine learning classification models to predict whether an individual's income exceeds $50K/yr based on census data. The project involves end-to-end ML deployment including data preprocessing, model training, evaluation, and creating an interactive web application using Streamlit for live inference.

## b. Dataset Description
* **Dataset Name:** Adult Census Income
* **Source:** Kaggle - Adult Census Income
* **Instances:** 32,561 rows (subset used for deployment optimization)
* **Features:** 14 attributes (including age, workclass, education, marital.status, occupation, race, sex, capital.gain/loss, hours.per.week, native.country)
* **Target:** `income` (Binary classification: >50K or <=50K)

## c. Github Repository Link
https://github.com/AvinashDuvvuru/Adult-Income-Classification

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| :--- |:---------|:-------|:----------|:-------|:-------|:-------|
| **Logistic Regression** | 0.858    | 0.9096 | 0.7150    | 0.6538 | 0.6830 | 0.5928 |
| **Decision Tree** | 0.817    | 0.7633 | 0.5985    | 0.6624 | 0.6288 | 0.5089 |
| **KNN** | 0.804    | 0.8291 | 0.5792    | 0.5940 | 0.5865 | 0.4581 |
| **Naive Bayes** | 0.294    | 0.5585 | 0.2473    | 0.9872 | 0.3955 | 0.1184 |
| **Support Vector Machine** | 0.847    | 0.9040 | 0.7213    | 0.5641 | 0.6331 | 0.5448 |

## e. Observations
| ML Model Name | Observation about model performance                                                                                                                                                                                                                                        |
| :--- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | This is the most balanced model. It has the highest overall accuracy (85.8%), the highest F1 Score, and the best AUC. It is very good at correctly separating the high and low-income groups without making too many false positive mistakes.                              |
| **Decision Tree** | This model is fairly average. It has a decent recall (meaning it catches a good number of high-income earners), but its precision is lower than Logistic Regression, meaning it incorrectly guesses that some low-income earners are high-income.                          |
| **KNN** | KNN performed adequately but slightly worse than the Decision Tree. It struggles slightly with precision and recall, likely because census data has many complex categorical variables that make distance-based calculations less effective.                               |
| **Naive Bayes** | This model performed very poorly on accuracy (29.4%). It has an extremely high recall (98.7%), meaning it almost always guesses ">50K", but its precision is terrible. It basically classifies almost everyone as a high-income earner, making many false positive errors. |
| **Support Vector Machine** | SVM performed very well, acting as a close runner-up to Logistic Regression. It has excellent precision and a very high AUC (0.904), but its recall is a bit low, meaning it is very cautious and misses several actual high-income earners.                               |
| **Overall Winner** | **Logistic Regression**. It achieved the highest Accuracy, highest F1 Score, and highest MCC, providing the most reliable and balanced predictions across both income classes.                                                                                             |

