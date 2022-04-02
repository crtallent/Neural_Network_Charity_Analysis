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

We then used OneHotEncoder to encode our categorical varibles in a new DataFrame, and then merged these new variables into our DataFrame and dropped the original categorical variables:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/merged%20df.png" />

We then split our preprocessed data into our features and target arrays and the split it into our training and testing datasets:

~~~
y = app_df.IS_SUCCESSFUL.values
X = app_df.drop(["IS_SUCCESSFUL"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

~~~

Finally, we created our StandardScaler instances, fit it, and then scaled our data:

~~~
# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
~~~

Once we had our data processed, fit, and scaled, it was time to Compile, Train, and Evalate our Model, as seen in our Results below.

## Results

The following steps were performed to compile, train, and evaluate our neural network model:

1. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow Keras.
2. Create the first hidden layer and choose an appropriate activation function.
3. If necessary, add a second hidden layer with an appropriate activation function.
4. Create an output layer with an appropriate activation function.
5. Check the structure of the model.

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Training.png" />

6. Compile and train the model, and create a callback that saves the model's weights every 5 epochs.

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Callback.png" />

7. Evaluate the model using the test data to determine the loss and accuracy and save and export results to HDF5 file named "AlphabetSoupCharity".

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Evaluate.png" />

As we can see from our results above, our accuracy of our predictive model was not as high as we would have hoped, at 72.56%. Our goal was 75% accuracy, so we needed to optimize our model to see if could achieve a higher accuracy score. For our first attempt at optimization, we increased our hidden nodes layers to 6, which actually decreased the accuracy of our model:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Attempt%201.png" />

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Result%201.png" />

For our second attempt, we decreased our hidden layers to 3, but increased the amount of nodes in each layer, and increased our epochs to 100:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Attempt%202.png" />

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Result%202.png" />

Not only did this method greatly increase our run time as had so many [parameters](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Params.png), our accuracy was only enhanced by 0.02%, still far under our 75% mark. Other methods tried included the following:

1. Removing columns such as APPLICATION_TYPE, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS AND ASK_AMT did not increase our accuracy, but in most cased, actually lowered our accuracy percentage.
2. Re-binning our APPLICATION_TYPE and CLASSIFICATION columns - this greatly reduced our accuracy to 46-58%.
3. Adding more and less hidden layers, as well as more and less nodes - this produced varying results, but always lowered our accuracy.
4. Changing our activation types with varying results.
5. Using TensorFlow's keras tuner method to create a function to check for the best method for higher accuracy:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/keras.png" />

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/keras%20results.png" />

With the keras tuner method, we were able to see that the highest predicted accuracy using the data in its structure was 72.79% - the highest accuracy thus far, but still below our 75% benchmark. This indicated that additional processing work to the original dataset may be needed.

With that in mind, we returned to our dataset and analyzed information about our "NAME" column that we had previously dropped. Upon review of unique values, there were quite a few donations made to the same companies:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Name_counts.png" />

We decided to keep that feature in our dataset, and after tuning our hidden layers, nodes, and activation types again, we were finally able to produce results above the 75% benchmark:

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Attempt%203.png" />

<img src="https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Attempt%203%20Result.png" />
          
## Summary

As we can see from our final result, we were able to increase our accuracy's model to 78.3%.  However, our training loss was 0.48, and our training accuracy was 96.63%, showing that we may have overfitted our data. We also had a total of 588,779 [parameters](https://github.com/crtallent/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization_Final.ipynb), which was likely more than what we needed. Before using this model, it would be advisable to perform TensorFlow's keras tuner function with the updated features to determine the most accurate model to use. 


