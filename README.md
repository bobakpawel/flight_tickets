# flight_tickets
A Linear Regression model was built to predict flight ticket prices.

### The main goal of the project is to build a Linear Regression model to predict flight ticket prices as a part of data science learning. 

### The data used in the project was gathered via web scraping from the 'Ease My Trip' website. It contains basic information about flight tickets on sale during February and March 2022 for routes to/from six Indian cities. The dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction .

### The data gathered from Kaggle has two versions: clean and raw. The author used the raw files. Basic data cleaning steps, such as removing duplicates, checking for NA values, and more, were performed. The data in the dataframe columns was visually inspected. A few rows with inconsistent formats for transformation were removed, and some data type transformations were applied.

### The next step the project focused on was performing feature engineering. During this activity, the author used a few basic functions, such as pd.cut(), np.where(), and map(), to create new features with categorical values or flags. The features created during this step may be used in the linear regression model. This phase concluded with the use of Seaborn's heatmap and a specially prepared function to identify the correlation between the dependent variable and the independent features in the dataframe.

### A function was created to automate the steps taken during Linear Regression using functions from the sklearn libraries. This approach allows changing a single parameter and observing potential changes to the model without producing too many lines of code. Although not a very creative approach to Linear Regression, it enables achieving an R2 of 0.915 and a MAPE of 45.014.

### Finally, a more creative approach was taken to the Linear Regression. Most importantly, an approach similar to pd.get_dummies() was applied, but each categorical value from the feature received not one column, but two (one for Economy Class ticket and one for Business Class ticket). This more tailored approach helped improve the R² to 0.950 and significantly reduce MAPE to 22.403.
