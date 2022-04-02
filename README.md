# Neural_Network_Charity_Analysis

## Overview

For this project, we are working with Alphabet Soup, a non-profit organization that has raised and donated $10 billion over the last 20 years. They would like to analyze the impact of each of their donations, but sometimes the businesses that receive the funding disappear. Since Alphabet Soup's goal is to protect people's health and the environment, they would like to ensure that their contributions are making the impacts that will help them reach these goals. It is therefore necessary to determine if a neural network can help the company predict whether a donation should be made to an organization, and when the endeavor is too high-risk. Alphabet Soup has provided a dataset of over 34,000 donations that they have made, and the dataset can be found [here](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv).

## Resources

* Software: Python 3.7.6, Jupyter Notebook 7.29.0, TensorFlow r2.7, scikit-learn 0.24
* Data Source: [charity_data.csv](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
* All code can be found at the following links: 
  * [Neural Network with TensorFlow](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) 
  * [Optimized Neural Network with TensorFlow](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization_Final.ipynb)

## Preprocessing the Data

Our first step in preparing the data for our neural network after reading in our dataset was to drop any column not considered as either a target or feature for our model. Upon reading in our dataset, our target array was our "IS_SUCCESSFUL" column. Our features were the following columns from our dataset:

* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT

Another important part of preprocessing was to limit the amount of "noise" in our dataset, to allow for smoother processing by our neural network (supervised machine learning) model. We looked at our categorical data points to determine if any of these had more than 10 unique values. We found that our APPLICATION_TYPE dimension had 17, while our CLASSIFICATION dimension had 71 unique values. We then checked the distribution of these values, and "binned" these columns into six categories each:

<p float="left">
  <img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/APP.png" title="APPLICATION_TYPE" width="400" height="175" />
  <img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/CLASS.png" title="cLASSIFICATION" width="400" height="175" />
</p> 

