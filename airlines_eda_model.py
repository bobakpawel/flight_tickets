import pandas as pd 
import numpy as np 
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns',None)
warnings.filterwarnings("ignore")

economy = pd.read_csv('C:/PYTHON/regr_project/economy.csv', parse_dates = ['date'])
business = pd.read_csv('C:/PYTHON/regr_project/business.csv', parse_dates = ['date'])
economy['class'] = 'Eco'
business['class'] = 'Bus'
tickets = pd.concat(objs=[economy, business]).reset_index(drop = True)


##### EDA PHASE ##### 

tickets = tickets.drop_duplicates()
### Two rows in the database are exact duplicates (same values in all columns). These rows have been removed to avoid redundancy. The database is now confirmed to be free of duplicates.

#tickets.info()
tickets.isna().sum()
### There are 300 259 non-null values in each column. This satisfies one of the key conditions for performing Linear Regression, which requires no missing values in the dataframe.

tickets['price'] = tickets['price'].str.replace(',','').astype(int)
### In the original file, the 'price' column is stored as an object datatype and contains ',' as a separator. This needs to be removed (replaced with an empty string '') in order to process the data correctly and convert the column's datatype to integer.

tickets['date'] = pd.to_datetime(tickets['date'], errors = 'coerce', format='%d-%m-%Y')
### The data format in the 'date' column has been successfully converted to the datetime64[ns] type. The values in this column range from 2022-02-11 to 2022-03-31. Approximately 6000 rows exist for each day (tickets['date'].value_counts().sort_index().plot(kind='bar');). Additionally, there are no missing dates within the analyzed date range.
### A quick look at the average ticket prices by flight date shows that prices increase as the flight date approaches. This trend holds true until around 2022-02-26, after which the average ticket prices appear to level off. This pattern is particularly noticeable for economy class tickets. (tickets.groupby('date')['price'].mean().plot(kind='bar'); / tickets[tickets['class'] == 'Eco'].groupby('date')['price'].mean().plot(kind='bar'); / tickets[tickets['class'] == 'Bus'].groupby('date')['price'].mean().plot(kind='bar');).

tickets['airline'].value_counts(dropna = False)
tickets = tickets[~tickets['airline'].isin(['StarAir', 'Trujet'])]
tickets[['airline', 'class']] = tickets[['airline', 'class']].astype('category')
### The dataset contains 8 airlines. Two airlines (StarAir and Trujet) have only 40-60 observations, which is significantly fewer than the remaining 6 airlines (with 9k to 128k observations). Due to this, the 'airline' column will be converted to integer values using the pd.get_dummies() function. It may be worth considering removing these two airlines from further analysis, as their sample sizes are very small, especially compared to the other airlines.
### StarAir operates 61 flights between Hyderabad and Bangalore (both directions), while there are over 16000 flights between these two cities in total. Since the airline brand is likely to be an important factor in the model, StarAir flights could be excluded from further analysis. The situation with Trujet is similar, and it may also be appropriate to exclude that airline from the analysis for the same reason (tickets[tickets['airline'] == 'StarAir'].groupby(['from', 'to'])['stop'].count() // tickets [ ((tickets['from'] == 'Bangalore') & (tickets['to'] == 'Hyderabad')) | ((tickets['to'] == 'Bangalore') & (tickets['from'] == 'Hyderabad')) ].groupby(['from', 'to'])['stop'].count()).
### The average ticket prices for Vistara and Air India are significantly higher than those of the other airlines. This is because these two airlines are the only ones offering Business Class tickets, which are, on average, 8 times more expensive than Economy Class tickets. This suggests that the class of service will likely be a strong predictor of ticket prices (tickets.groupby('class')['price'].mean())). Secondly, even when considering only Economy Class tickets, Vistara and Air India are still noticeably more expensive on average compared to the other airlines (tickets[tickets['class'] == 'Eco'].groupby('airline')['price'].mean().sort_values().plot(kind = 'bar'); ).
### Converting the 'airline' feature, which has only 6 possible text values, to the 'category' data type reduced memory usage from 175.5 MB to 157.3 MB. A similar procedure applied to the 'class' feature further reduced memory usage to 140.4 MB.

tickets.groupby(['airline', 'ch_code'])['time_taken'].count()
tickets.drop('ch_code', axis = 1, inplace = True)
### The variables 'ch_code' and 'airline' provide the same information, as 'ch_code' serves as an ID for the airline name. One of these columns should be omitted from further analysis, so the column 'ch_code' has been dropped from the analyzed dataframe.

num_code_df = tickets['num_code'].value_counts(dropna=False).to_frame().reset_index()
### There are 1254 unique 'num_code' values. 83 of them have more than 1000 occurrences (num_code_df[num_code_df['count'] >= 1000]), while 200 have fewer than 10 occurrences (num_code_df[num_code_df['count'] <= 10]). The author is uncertain about the type of information this feature provides. If the model's predictive power is low, it will be tested for potential regression with both dependent and independent features. Although the values appear numeric, it seems to be a categorical feature.

dep_time_df = tickets['dep_time'].value_counts(dropna = False).to_frame().reset_index().sort_values('dep_time')
### There are 251 possible departure times in the database. After sorting the values, departure times range from 00:10 to 23:55. The data appears to be very clean, and these departure times will be used in the model creation phase.
### The average ticket price per hour doesn't appear to be a strong predictor of ticket price. It is clear that late-night tickets tend to have lower prices compared to tickets with departure times during typical working hours, especially for Economy class tickets (tickets.groupby('dep_time')['price'].mean().sort_index().plot(kind='bar'); / tickets[tickets['class'] == 'Eco'].groupby('dep_time')['price'].mean().sort_index().plot(kind='bar'); / tickets[tickets['class'] == 'Bus'].groupby('dep_time')['price'].mean().sort_index().plot(kind='bar');). 

arr_time_df = tickets['arr_time'].value_counts(dropna = False).to_frame().reset_index().sort_values('arr_time')
### A similar analysis was performed for the 'arr_time' feature as was done for 'dep_time'. The results are similar to those obtained in the 'dep_time_df'.
### The chart presenting the average ticket price per arrival time hour shows a more even distribution, with a slight increase in price from 5 or 6 o'clock. This contrasts with the bar charts for departure time vs ticket price, which displayed more noticeable trends. Additionally, the trend for Economy class tickets is also easier to identify in this case (tickets.groupby('arr_time')['price'].mean().sort_index().plot(kind='bar');).

tickets = tickets[~tickets['time_taken'].str.contains('\.')] 
### There are 4 problematic rows with a different structure in the 'time_taken' column, which presents flight duration. Since the database contains more than 300k rows, these 4 rows were excluded. While flight duration could be calculated using algorithms, these rows were excluded from further analysis due to the database size and other potential issues related to these rows.
### An increasing trend in ticket prices can be observed as the value in the 'time_taken' feature increases. Generally, this is not surprising, as longer journeys tend to have higher ticket prices (tickets[tickets['class'] == 'Eco'].groupby('time_taken')['price'].mean().plot(); / tickets[tickets['class'] == 'Bus'].groupby('time_taken')['price'].mean().plot();).

tickets['from'].value_counts(dropna=False)
### There are 6 departure points, with the distribution among them not being equal, but each having more than 38k occurrences. Therefore, all of them are suitable targets for applying the pd.get_dummies() function.
### The average ticket price from Delhi appears slightly lower than from other cities. However, for the remaining 5 cities, the differences are small. The 'from' feature is likely not a very strong predictor of ticket price. It should also be noted that another trend is observed, which is true only for Economy class tickets: tickets to/from Kolkata are more expensive than those to/from other destinations (tickets.groupby('from')['price'].mean().sort_values().plot(kind = 'bar'); / tickets[tickets['class'] == 'Eco'].groupby('from')['price'].mean().sort_values().plot(kind = 'bar');).

tickets['to'].value_counts(dropna=False)
### The findings here are similar to those for the 'from' variable. There are only 6 arrival points, and each is well represented in the database, with more than 40k rows per 'to' point. Similarly to the 'from' feature, the average ticket price to Delhi is slightly lower than to other cities. As noted previously, Economy class tickets to/from Kolkata are more expensive than those to/from other destinations.

tickets['stop'].value_counts()
tickets['stop'] = tickets['stop'].str.replace('\t','').str.replace('\n','')
tickets['stop_ed'] = np.where(tickets['stop'] == 'non-stop ', '0s', np.where(tickets['stop'] == '2+-stop', '2s','1s'))
### The lack of unification in the values in the 'stop' column makes analysis more challenging. There are many characters, such as '\t', that do not add value to the analysis and will be excluded. The same exclusion rules will be applied to the '\n' character.
### There are over 243k flights with the value '1-stop'. Additionally, approximately 7.3k entries with '1-stop' contain extra information about the airport where the flight made a stop (mid-landing). Since the information standard for '1-stop' flights is inconsistent, all values will be standardized into 3 categories: '0s' (non-stop), '1s' (1-stop), and '2s' (2+-stop). This information will be added to a new column to preserve data about the airport where passengers changed planes, in case it proves useful for the model (to be checked later if this decision is made).
### It is clearly visible that non-stop ('0s') ticket prices are on average the lowest (around 4k for Economy Class and 28k for Business Class). 1-stop ('1s') ticket prices are approximately 7k (Economy Class) and 54k (Business Class). The most expensive tickets on average are for flights with two or more mid-landings: around 9k for Economy Class and 70k for Business Class. This will likely be a strong ticket price predictor in the model (tickets[tickets['class'] == 'Eco'].groupby('stop_ed')['price'].mean().plot(kind = 'bar'); / tickets[tickets['class'] == 'Bus'].groupby('stop_ed')['price'].mean().plot(kind = 'bar');).

tickets[['from', 'to', 'stop_ed']] = tickets[['from', 'to', 'stop_ed']].astype('category')
### Three column data types were switched to 'category' type to reduce memory usage (from 134.2 MB to 81.5 MB).

features = ['date', 'class', 'airline', 'from', 'to', 'dep_time', 'arr_time', 'time_taken', 'stop_ed', 'price']
tickets_eda = tickets[features].copy()


##### FEATURE ENGINEERING PHASE ##### 

tickets_eda['days_left'] = (tickets_eda['date'] - pd.to_datetime('2022-02-10')).dt.days
labels_days_left = ['1', '2-3','4-10','11-15','15+']
bins_days_left = [-1, 1, 3, 10, 15, 1000] 
tickets_eda['days_left_gr'] = pd.cut(tickets_eda['days_left'], bins = bins_days_left, labels = labels_days_left)
### According to the dataset description, the data frame was prepared via web scraping performed on '2022-02-10' and contains ticket prices for the upcoming 49 days (until the end of March 2022). Previous analysis has demonstrated that the number of days before the flight is a strong ticket price predictor, particularly for Economy class tickets.
### The relationship between ticket price and the number of days until the flight was analyzed for both Economy and Business Class tickets. It was observed that the most expensive tickets are for flights the next day (in both classes). Then, there is a noticeable drop in average ticket prices for the following two days. Purchasing tickets 4 to 10 days in advance corresponds to another price range, especially visible for Economy class, though less so for Business Class. For Economy class tickets, a distinct price range appears for purchases made 11 to 15 days in advance. Beyond that, average ticket prices remain relatively flat in both classes (tickets_eda[tickets_eda['class'] == 'Eco'].groupby('days_left')['price'].mean().plot(kind = 'bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('days_left')['price'].mean().plot(kind = 'bar');).
### Both approaches (categorizing days into groups and using the exact number of days in advance for ticket purchases) will be tested during the model-building phase. The results of grouping by days_left_gr and calculating the mean ticket price suggest that this approach may yield good predictive scores (tickets_eda[tickets_eda['class'] == 'Eco'].groupby('days_left_gr')['price'].mean().plot(kind = 'bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('days_left_gr')['price'].mean().plot(kind = 'bar'); ).

tickets_eda['day_of_week'] = (tickets_eda['date'].dt.strftime('%a')).astype('category')
tickets_eda['weekend_flg'] = np.where(tickets_eda['day_of_week'].isin(['Fri', 'Sat', 'Sun']), 1, 0)
### The day of the week might significantly influence ticket prices prediction and therefore requires examination during the model-building phase. (tickets_eda[tickets_eda['class'] == 'Eco'].groupby('day_of_week')['price'].mean().plot(kind = 'bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('day_of_week')['price'].mean().plot(kind = 'bar');).
### A quick glance at the charts above reveals that Economy Class tickets are more expensive for weekends (Fri-Sun) compared to other days. Meanwhile, Business Class tickets show only a minimal increase in price during the weekend. While expectations are modest, this factor will be tested in the model.

tickets_eda['dep_hour'] = pd.to_datetime(tickets_eda['dep_time'], format = '%H:%M').dt.hour
tickets_eda['arr_hour'] = pd.to_datetime(tickets_eda['arr_time'], format = '%H:%M').dt.hour
### Two new columns containing the integer hour values for departure and arrival times were created. The author suspects that the departure and arrival hours might correlate with the ticket price.
### A chart for Economy Class tickets reveals that average ticket prices vary based on the departure hour. Tickets departing at 22, 23, 0 to 4 are, on average, cheaper than those departing during other hours. A similarly strong trend was not observed for Business Class tickets. Nonetheless, a flag indicating departure hours in the range of 5 to 21 will be created. (tickets_eda[tickets_eda['class'] == 'Eco'].groupby('dep_hour')['price'].mean().plot(kind='bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('dep_hour')['price'].mean().plot(kind='bar');).
### Regarding the arrival hours, a similar trend is observed for Economy Class tickets: average ticket prices differ significantly for arrivals between 1 and 5. However, no clear trend is visible for Business Class tickets. As a result, a flag will be created to differentiate more popular arrival hours.

tickets_eda['dep_hour_non_sleep'] = np.where( ((tickets_eda['dep_hour'] > 4) & (tickets_eda['dep_hour'] < 22)), 1, 0)
### The hint provided earlier with the newly created column dep_hour was used to create a new column. The price difference between the two groups appears to be significant, both overall and in each ticket class separately (tickets_eda.groupby('dep_hour_non_sleep')['price'].mean().plot(kind='bar'); / tickets_eda[tickets_eda['class'] == 'Eco'].groupby('dep_hour_non_sleep')['price'].mean().plot(kind='bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('dep_hour_non_sleep')['price'].mean().plot(kind='bar');).
tickets_eda['arr_hour_non_sleep'] = np.where( ((tickets_eda['arr_hour'] == 0) | (tickets_eda['arr_hour'] >= 6)), 1, 0)
### Similarly, the observation from the arr_hour analysis was applied here. A flag was created to differentiate two groups of rows based on the arrival hour: 1-6 and 7-24 (0). A distinct difference in average ticket prices was noted between these groups overall and within the Economy class, although no such trend was observed in the Business class (tickets_eda.groupby('arr_hour_non_sleep')['price'].mean().plot(kind='bar'); / tickets_eda[tickets_eda['class'] == 'Eco'].groupby('arr_hour_non_sleep')['price'].mean().plot(kind='bar'); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('arr_hour_non_sleep')['price'].mean().plot(kind='bar');).

tickets_eda['dep_min'] = pd.to_datetime(tickets_eda['dep_time'], format = '%H:%M').dt.minute
tickets_eda['arr_min'] = pd.to_datetime(tickets_eda['arr_time'], format = '%H:%M').dt.minute
### As expected, the minute of departure or arrival does not show a correlation with the ticket price. Still, it was better to quickly verify this hypothesis.

tickets_eda['dur_h'] = np.where(tickets_eda['time_taken'].str.split(' ').str[0].str.replace('h','').str[0] == '0', 
       tickets_eda['time_taken'].str.split(' ').str[0].str.replace('h','').str.replace('0','', n=1),
       tickets_eda['time_taken'].str.split(' ').str[0].str.replace('h','')).astype(int)

tickets_eda['dur_m'] = np.where(tickets_eda['time_taken'].str.split(' ').str[1].str.replace('m','').str[0] == '0',
        tickets_eda['time_taken'].str.split(' ').str[1].str.replace('m','').str.replace('0','', n=1),
       tickets_eda['time_taken'].str.split(' ').str[1].str.replace('m','')).astype(int)

tickets_eda['dur'] = tickets_eda['dur_h'] * 60 + tickets_eda['dur_m']
### Firstly, two new columns were engineered. The first contains flight duration in hours (up to 49 hours, which is possible: anything longer than 4 hours is associated with mid-landing, as indicated by the 'stop_ed' column). The second column shows flight duration in minutes (ranging from 0 to 59 minutes, as expected). Both of these values were extracted from the 'time taken' field, which, except for one exception in 4 rows (previously excluded), has a consistent format that allows such values to be extracted using the np.where() function. The values were then converted to the integer data type.
### The charts showing the average ticket price versus trip duration in hours do not exhibit a clear linear relationship. Economy Class tickets show sharp price growth for trips lasting between 0 and 4 hours, followed by gradual, consistent growth up to 40 hours. For trips exceeding 40 hours, a rapid price increase is observed. However, this trend is based on only 13 flights, making it statistically insignificant as a predictive metric. Business Class tickets show rapid price growth for trips longer than 4 hours. Between 10 and 30 hours, a slight decrease in price with each additional hour is observed, followed by rapid price growth after 30 hours.
### In the model construction phase, the author will begin by using the dur_h metric. If the results are unsatisfactory, alternative approaches such as incorporating squared terms or creating new features that combine the ticket class flag with trip duration will be considered. Constructing custom bins for the number of hours a trip lasted will be used as a last resort  (tickets_eda.groupby('dur_h')['price'].mean().sort_index().plot(); / tickets_eda[tickets_eda['class'] == 'Eco'].groupby('dur_h')['price'].mean().sort_index().plot(); / tickets_eda[tickets_eda['class'] == 'Bus'].groupby('dur_h')['price'].mean().sort_index().plot(); / tickets_eda.loc[ (tickets_eda['class'] == 'Eco') & (tickets_eda['dur_h'] > 40), :]).

tickets_eda['road'] = (tickets_eda['from'].str[:3] + '-' + tickets_eda['to'].str[:3]).astype('category')
### The point of departure and the point of arrival might be useful variables for further investigation during the model-building phase, but the hypothesis suggests that analyzing the full route could provide more value. There are 30 routes, each with over 6.1k rows of occurrences, and the difference in the average ticket price depending on the route is clearly visible on the chart (both overall and separately for each ticket class).

distance_dict = {'Del-Ban' : 2138, 'Del-Mum' : 1365, 'Ban-Del' : 2138, 'Mum-Del' : 1365, 'Mum-Kol' : 1885, 'Ban-Mum' : 986,
       'Mum-Ban' : 986, 'Del-Kol' : 1633, 'Kol-Mum' : 1885, 'Kol-Del' : 1633, 'Del-Che' : 2274, 'Che-Del' : 2274,
       'Hyd-Mum' : 704, 'Mum-Hyd' : 704, 'Kol-Ban' : 1874, 'Ban-Kol' : 1874, 'Mum-Che' : 1321, 'Del-Hyd' : 1581,
       'Che-Mum' : 1321, 'Hyd-Del' : 1581, 'Ban-Hyd' : 574, 'Hyd-Kol' : 1490, 'Kol-Hyd' : 1490, 'Hyd-Ban' : 574,
       'Che-Kol' : 1668, 'Kol-Che' : 1668, 'Hyd-Che' : 632, 'Ban-Che' : 333, 'Che-Hyd' : 632, 'Che-Ban' : 333}

tickets_eda['distance'] = tickets_eda['road'].map(distance_dict)
### Using Google Maps, the author verified the distance (by road) between the cities. This information will likely be incorporated into the model using the variable road. However, for learning purposes, the author will assess whether adding this variable improves the model's metrics.

model_features = ['days_left', 'days_left_gr', 'day_of_week', 'weekend_flg', 'class', 'airline', 'from', 'to', 'road', 'distance', 'stop_ed', 'dep_hour', 'arr_hour', 'dep_hour_non_sleep', 'arr_hour_non_sleep', 'dur', 'price']

tickets_mdl_rd = tickets_eda[model_features].copy()
tickets_mdl_rd = tickets_mdl_rd.sample(frac=1, random_state = 999).reset_index(drop=True)

#sns.heatmap(tickets_mdl_rd.corr(numeric_only=True), annot = True, fmt = '.2f', cmap = 'RdYlGn', vmin = -1, vmax = 1); / sns.heatmap(tickets_mdl_rd[tickets_mdl_rd['class'] == 'Eco'].corr(numeric_only=True), annot = True, fmt = '.2f', cmap = 'RdYlGn', vmin = -1, vmax = 1); / sns.heatmap(tickets_mdl_rd[tickets_mdl_rd['class'] == 'Bus'].corr(numeric_only=True), annot = True, fmt = '.2f', cmap = 'RdYlGn', vmin = -1, vmax = 1);
### The Seaborn heatmap method helps to identify the best ticket price predictors from all numeric features in the dataframe. The feature days_left has a strong correlation with price, with a value of -0.56. However, this strong correlation exists only for Economy Class tickets. For Business Class, the relationship between these two features is much weaker, with a correlation of only -0.09.

def corr_checker(correlation_strength = 0.15, df = tickets_mdl_rd, dependent_feature = 'price'):
    for column in pd.get_dummies(df, drop_first=True, dtype=int).columns:
        if column == dependent_feature:
            pass
        else:
            corr = pd.get_dummies(df, drop_first=True, dtype=int)[dependent_feature].corr(pd.get_dummies(df, drop_first=True, dtype=int)[column])
            if abs(corr) > correlation_strength:
                print(column, round(pd.get_dummies(tickets_mdl_rd, drop_first=True, dtype=int)[dependent_feature].corr(pd.get_dummies(df, drop_first=True, dtype=int)[column]),3))
                
#corr_checker(df = tickets_mdl_rd[tickets_mdl_rd['class'] == 'Eco'])
### Another approach to identifying features that are strongly correlated with the dependent variable is the specially constructed corr_checker() function. The function iterates through all model features and returns only those that are sufficiently correlated with the dependent variable. Key metrics are parameterized to allow individual control over their thresholds.


##### MODEL CONSTRUCTION PHASE - APPROACH A ##### 

X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class']], dtype = int, drop_first = True))
y = tickets_mdl_rd['price']

def linear_regr_func(X = X, y = y):
    X_train_main, X_test, y_train_main, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_main, y_train_main, test_size=0.25, random_state=1234)

    model = sm.OLS(y_train, X_train).fit()

    train_df = pd.DataFrame({'actual' : y_train, 'predict' : model.predict(X_train)})
    valid_df = pd.DataFrame({'actual' : y_valid, 'predict' : model.predict(X_valid)})
    test_df = pd.DataFrame({'actual' : y_test, 'predict' : model.predict(X_test)})

    print('TRAIN R2:', round(r2_score(y_train, model.predict(X_train)),3), 'VALID R2:', round(r2_score(y_valid, model.predict(X_valid)),3), 'TEST R2:', round(r2_score(y_test, model.predict(X_test)),3) )
    print('TRAIN MAE:',  round(mae(train_df['actual'], train_df['predict']),3), 'VALID MAE:', round(mae(valid_df['actual'], valid_df['predict']),3), 'TEST MAE:', round(mae(test_df['actual'], test_df['predict']),3))
    print('TRAIN MAPE:', round(100*mape(train_df['actual'], train_df['predict']),3), 'VALID MAPE:', round(100*mape(valid_df['actual'], valid_df['predict']),3), 'TEST MAPE:', round(100*mape(test_df['actual'], test_df['predict']),3))
    print()
    
    durbin_watson_value = round(sm.stats.stattools.durbin_watson(model.resid),3)
    
    if durbin_watson_value > 2.05 or durbin_watson_value < 1.95:
        print('--- Durbin - Watson value outside expected range of [1.95,2.05], DW:', durbin_watson_value)
    else:
        print('+++ Durbin - Watson value is in the expected range, DW:', durbin_watson_value)
    
    if model.f_pvalue > 0.01:
        print('--- Prob (F-statistic) is higher than 0.01:', model.f_pvalue)
    else:
        print('+++ Prob (F-statistic) value is ok:', model.f_pvalue)
    
    x = 0
    for var, p in model.pvalues.items():
        if p > 0.01:
            print(f"--- {var}: {round(p,3)}")
            x += 1
    if x == 0:
        print("+++ The p-values for all model coefficients are below 0.01, indicating statistical significance at the 0.01 level.")

### The author created the linear_regr_func() function, which performs several tasks related to linear regression. First, using train_test_split(), the function divides the given dataset into three subsets: train, validation, and test. The model is then created and fitted based on the training data, and several accuracy metrics—such as MAE, MAPE, and R²—are returned. The function also checks whether other important metrics, such as the Prob (F-statistic) or p-values for all model coefficients, are below the specified thresholds.
        
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class']], dtype = int, drop_first = True)) ### [TRAIN|VALID] R2: 0.879, 0.880 | MAPE: 43.998, 43.689 ---> Building a model based on only one feature, class, yields an R2 of 0.879 and a MAPE of approximately 44.0. Adding more features to the model will surely increase its accuracy.
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class', 'days_left']], dtype = int, drop_first = True))   ### R2:  0.886, 0.887 | MAPE: 37.939, 37.772 ---> Adding the second feature, days_left, significantly improved the model's predictive power. Before proceeding with other features, the author will check whether days_left_gr yields better results in the model.
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class', 'days_left_gr']], dtype = int, drop_first = True)) ### R2: 0.889, 0.890 | MAPE: 34.756, 34.395 ---> The addition of the days_left_gr feature significantly improves the main model metrics (R2 and MAPE). Furthermore, the improvement contributed by days_left_gr to the model is greater—particularly in the MAPE metric—than the improvement contributed by the days_left feature.
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class', 'days_left_gr', 'airline', 'dur']], dtype = int, drop_first = True)) ### R2: 0.900, 0.901 | MAPE: 32.350, 32.364 ---> Two new features were added, contributing to a slight improvement in the model. As a result, the R2 value reached 0.900 for the first time, and the MAPE is slightly higher than 32.3.
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[['class', 'days_left_gr', 'airline', 'dur', 'stop_ed']], dtype = int, drop_first = True)) ### R2: 0.910, 0.911 | MAPE: 44.361, 44.288 ---> Adding the feature stop_ed to the model helps significantly increase R2, but has a negative impact on the MAPE metric, causing it to rise sharply to 44.361. This suggests that an alternative approach might be worth trying in this situation.
X = sm.add_constant(pd.get_dummies(tickets_mdl_rd.drop('price', axis = 1), dtype = int, drop_first = True)) ### R2: 0.915, 0.915 | MAPE: 45.014, 45.012 ---> Building a model using all features results in an R2 of 0.915, but the MAPE is 45. The model seems to be influenced by the ticket class, which suggests that the dataset may behave differently for each class. This implies that a specialized approach to feature engineering for each class could improve the model's predictive accuracy. By treating each ticket class (Economy and Business) as a separate entity or tailoring the feature engineering process to account for the differences between them, the model might better capture the nuances that influence ticket prices for each class, potentially leading to better performance.

linear_regr_func(X = X, y = y)
print()


##### MODEL CONSTRUCTION PHASE - APPROACH B #####

features_for_model = []    

tickets_mdl_rd['Eco_F'] = np.where(tickets_mdl_rd['class'] == 'Eco', 1, 0)
tickets_mdl_rd['Bus_F'] = np.where(tickets_mdl_rd['class'] == 'Bus', 1, 0)   ### Two new columns were created to serve as flags for the ticket classes. The first column will flag Economy Class tickets, and the second column will flag Business Class tickets. These flags will allow the model to differentiate between the two classes, making it easier to tailor the feature engineering and modeling process for each class, potentially improving the overall predictive accuracy of the model.

tickets_mdl_rd['dur_e'] = tickets_mdl_rd['dur'] * tickets_mdl_rd['Eco_F']
tickets_mdl_rd['dur_b'] = tickets_mdl_rd['dur'] * tickets_mdl_rd['Bus_F']
features_for_model.extend(tickets_mdl_rd.columns[-2:].tolist())
#X = sm.add_constant(tickets_mdl_rd[features_for_model])  
#linear_regr_func(X = X, y = y)  ### R2: 0.731, 0.730 | MAPE: 95.985, 95.217 ---> The model with only the feature 'dur,' which represents the number of minutes the trip will last and approaches the two ticket classes differently, is able to reach an R2 of 0.731 and a MAPE slightly lower than 96.

tickets_mdl_rd['dur_e_2'] = tickets_mdl_rd['dur_e'] ** 2
tickets_mdl_rd['dur_b_2'] = tickets_mdl_rd['dur_b'] ** 2
features_for_model.extend(tickets_mdl_rd.columns[-2:].tolist())
#X = sm.add_constant(tickets_mdl_rd[features_for_model])
#linear_regr_func(X = X, y = y)   ### R2: 0.888, 0.889 | MAPE: 50.536, 50.065 ---> Two columns created in the previous step were incorporated into the model in a second-degree transformation. This operation increased the model's R2 to 0.888 and significantly reduced the MAPE to slightly above 50. The author confirmed that further transformations of these features yielded minimal improvements to the model's predictive power.

# tickets_mdl_rd['days_left_e'] = tickets_mdl_rd['days_left'] * tickets_mdl_rd['Eco_F']
# tickets_mdl_rd['days_left_b'] = tickets_mdl_rd['days_left'] * tickets_mdl_rd['Bus_F']
# tickets_mdl_rd['days_left_e_2'] = tickets_mdl_rd['days_left_e'] ** 2
# tickets_mdl_rd['days_left_b_2'] = tickets_mdl_rd['days_left_b'] ** 2
# tickets_mdl_rd['days_left_e_3'] = tickets_mdl_rd['days_left_e'] ** 3
# tickets_mdl_rd['days_left_b_3'] = tickets_mdl_rd['days_left_b'] ** 3
# tickets_mdl_rd['days_left_e_4'] = tickets_mdl_rd['days_left_e'] ** 4
# tickets_mdl_rd['days_left_b_4'] = tickets_mdl_rd['days_left_b'] ** 4
# features_for_model.extend(tickets_mdl_rd.columns[-8:].tolist())
# X = sm.add_constant(tickets_mdl_rd[features_for_model]) 
# linear_regr_func(X = X, y = y)   ### R2: 0.912, 0.913 | MAPE: 30.031, 29.925 ---> Another feature added to the model is days_left, representing the number of days before the flight. This feature was also tailored to separately address tickets from different classes. It was expanded to include terms up to the fourth power, which improved the model's R2 to 0.912 and reduced the MAPE to slightly above 30. However, the author hypothesizes—based on previously created charts—that days_left_gr, a feature calculated differently for each ticket class, might yield even better results.

def function_editor(df = tickets_mdl_rd, feat_column = 'road'):
    x = 0
    for element in df[feat_column].value_counts().index.to_list():
        eko = '_e'
        bus = '_b'
        df[element+eko] = np.where( ( (df['class'] == 'Eco') & (df[feat_column] == element)), 1, 0)
        df[element+bus] = np.where( ( (df['class'] == 'Bus') & (df[feat_column] == element)), 1, 0)
        x += 2
        
    #df.drop(df.columns[-1], axis=1, inplace = True)
    features_for_model.extend(tickets_mdl_rd.columns[-(x):].tolist())   ### Function that works similarly to pd.get_dummies(). It takes a column with categorical values and creates new columns with 0/1 flags for each category, separated by ticket class. In other words, while pd.get_dummies() creates one column for each category, this function creates two columns for each category: one for Economy Class tickets and the other for Business Class tickets. Additionally, the function extends the list of columns for model creation.

function_editor(feat_column = 'days_left_gr')  
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[features_for_model], drop_first=True, dtype=int))
#linear_regr_func(X = X, y = y)   ### R2: 0.918, 0.919 | MAPE: 27.891, 27.731 ---> As the author expected, the feature days_left_gr performs better in the model compared to days_left. It achieves an R2 of 0.918 (an improvement of +0.006) and a MAPE of 27.891 (an improvement of more than 2 points). Therefore, this feature will be used in the model, and previous attempts with the days_left feature, even taken to the 4th power, will be commented out.

function_editor(feat_column = 'airline')
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[features_for_model], drop_first=True, dtype=int))
#linear_regr_func(X = X, y = y)   ### R2: 0.929, 0.929 | MAPE: 25.261, 25.145 ---> The model's predictive power continues to improve in both metrics. Three of the columns created from days_left_gr were reported as statistically insignificant. With the approach based on pd.get_dummies(), this is not that alarming, but it’s better to observe with the addition of other features.

function_editor(feat_column = 'stop_ed')
#X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[features_for_model], drop_first=True, dtype=int)) 
#linear_regr_func(X = X, y = y)    ### R2: 0.940, 0.940 | MAPE: 23.959, 23.819 ---> The R2 and MAPE of the model are still improving. However, the p-values of two metrics—dur_e (0.917) and dur_e (0.072)—were reported as statistically insignificant. This is definitely concerning and suggests that the model might be overfitting.

function_editor(feat_column = 'road')
X = sm.add_constant(pd.get_dummies(tickets_mdl_rd[features_for_model], drop_first=True, dtype=int)) 
linear_regr_func(X = X, y = y)     ### R2: 0.949, 0.950 | MAPE: 22.403, 22.258 ---> Adding another feature enabled the model to achieve an R2 of 0.95 and a MAPE lower than 22.5. Furthermore, the p-value of dur_e is now 'only' 0.02 (still not ideal, but much better). Five (out of 60 newly added columns) are reported as statistically insignificant. After applying a function similar to pd.get_dummies() to the features generated from that column, this doesn't seem overly concerning.

### The author also attempted adding other unused features to the model, such as weekend_flg, day_of_the_week, dep_hour, and more. Additionally, the author experimented with raising the terms dur_e and dur_b to higher powers. Unfortunately, the best result observed was an R2 of 0.951 and a MAPE of 22.19. Due to the minimal improvement in the model's accuracy, the author decided to conclude the model creation phase after adding the feature road to the model. It is worth noting that, with a more tailored approach, it was possible to increase the R2 from 0.915 to 0.95 and reduce the MAPE from 45.027 to 22.403.

