# Data Science Project 1: Real Estate Price Prediction
import joblib
import pandas
from matplotlib import pyplot
import numpy
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------
# df = original dataframe
# df1 = only required columns dataframe
# df2 = no empty cells dataframe
# df3 = cleaning out wrong cell entries
# df4 = cleaning out abnormal cell entries even if they seem correct but aren't
# df5 = cleaning out those dataframes with abnormal bedroom size
# df6 = removing those entries with weird bhk values in same location
# df7 = dataframe ready for modeling
# -------------------------------------------

# STEP 1: CLEANING OUT THE DATASET:
df = pandas.read_csv('Pune house data.csv')
print(df.head())

print(df.columns)
print(df['area_type'].unique())
print(df['availability'].unique())
print(df['size'].unique())
print(df['society'].unique())
print(df['total_sqft'].unique())
print(df['bath'].unique())
print(df['balcony'].unique())
print(df['site_location'].unique())

# Looking at above, we can drop: ['availability', 'society', 'site_location']
df1 = df.drop(['availability', 'society'], axis='columns')
print(df1.head())

# To see how many of them have empty cells:
print(df1.isna().sum())

# For bath and balcony:
df2 = df1.copy()
print(df2['bath'].unique())
df2['bath'] = df2['bath'].fillna(round(df2['bath'].mean()))
print(df2['balcony'].unique())
df2['balcony'] = df2['balcony'].fillna(round(df2['balcony'].mean()))
print(df2.isna().sum())

# For size:
print(df2['size'].unique())
# For such a big dataset, you can just drop them:
df2 = df2.dropna()
print(df2.isna().sum())

# Cleaning out size column:
df3 = df2.copy()
# .copy() function makes a deep copy
df3['size'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3['size'].unique())
# Cleaning out the area column:
print(df3['total_sqft'].unique())


def cleaning_total_sqft(cell):
    tokens = cell.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(cell)
    except:
        return None


df3['total_sqft'] = df3['total_sqft'].apply(lambda x: cleaning_total_sqft(x))
print(df3['total_sqft'].unique())
print(f'SIZE:  {df3.shape}')
print(df3.dropna(axis=0, inplace=True))
print(f'SIZE:  {df3.shape}')
print(df3)
# Cleaning out area_type column:
dummy_columns = pandas.get_dummies(df3['area_type'])
df3 = pandas.concat([df3, dummy_columns], axis='columns')
df3.drop(['area_type'], axis='columns', inplace=True)
print(df3.head())
print(df3.columns)

# IMPORTANT!! Cleaning out site_location column:
# First of all, see, how many unique locations you have:
print(df3['site_location'].nunique())
# So, 97 is a lot of rows. This is basically a dimensional curse.
print(df3.groupby(['site_location']).size().sort_values(ascending=True))
# However, after checking, each location has a significant number of housing.

# ----------------------------------------------------------------------------------
# # If above wasn't the condition, we would've done:
# location_stats = df3.groupby(['site_location']).size().sort_values(ascending=False)
# print(location_stats)
# # Basically, location_stats is a series. So it acts like a dataframe right.
# # Now, we can just say that the locations with only one entry/house/space are 'other'
# locations_with_one_space = location_stats[location_stats <= 1]
# print(locations_with_one_space)
# df3['site_location'] = df3['site_location'].apply(lambda x: x.strip())
# df3['site_location'] = df3['site_location'].apply(lambda x: 'other' if x in locations_with_one_space else x)
# print(df3['site_location'].nunique())
# ----------------------------------------------------------------------------------------
print(df3.shape)
print(df3.isna().sum())
print(f'SIZE:  {df3.shape}')


# --------------------------------------------------------------------------------------
# STEP 2: CLEANING OUT THE ABNORMAL ENTRIES(OUTLIER DETECTION):
df4 = df3.copy()
df4['price_per_sqft'] = (df4['price'] * 100000) / df4['total_sqft']
print(df4.head())


# ABNORMALITY 1:
# Let's say size of a bedroom less than 250 sqft is abnormal:
print(df4[df4['price_per_sqft'] / df4['size'] < 250])
# Thus, we can just remove these rows by:
df4 = df4[~(df4['price_per_sqft'] / df4['size'] < 250)]
# '~' negates the statement.


# ABNORMALITY 2:
# Now, check those flats where number of bathrooms is greater than number of bedrooms+2, and remove them:
print(df4[df4['bath'] > df4['size'] + 2])
df4 = df4[~(df4['bath'] > df4['size'] + 2)]
print(df4.shape)


# ABNORMALITY 3:
# Now, some costs can be abnormal too.
print(df4['price_per_sqft'].min())
print(df4['price_per_sqft'].max())
# Thus, there is a huge difference in the cost here. Atleast location-wise, the cost shouldn't deviate this much.


def abnormal_sqft_area(dataframe):
    new_dataframe = pandas.DataFrame()
    for location, sub_df in dataframe.groupby('site_location'):
        mean = sub_df['price_per_sqft'].mean()
        std = sub_df['price_per_sqft'].std()
        reduced_df = sub_df[(sub_df['price_per_sqft'] >= (mean - std)) & (sub_df['price_per_sqft'] <= (mean + std))]
        new_dataframe = pandas.concat([new_dataframe, reduced_df], axis=0, ignore_index=True)
    return new_dataframe


df5 = abnormal_sqft_area(df4)
print(df5.shape)


# ABNORMALITY 4:
# Now, in a particular locality, ideally, the 2 BHK flats should cost less than 3 BHK:


def compare_costs(dataframe, location):
    bhk_2 = dataframe[(dataframe['site_location'] == location) & (dataframe['size'] == 2)]
    bhk_3 = dataframe[(dataframe['site_location'] == location) & (dataframe['size'] == 3)]
    pyplot.scatter(bhk_2['total_sqft'], bhk_2['price'], marker='+', label='2 BHK')
    pyplot.scatter(bhk_3['total_sqft'], bhk_3['price'], marker='*', label='3 BHK')
    pyplot.xlabel('sqft')
    pyplot.ylabel('Price')
    pyplot.legend()
    # pyplot.show()


compare_costs(df5, 'Yerawada')
# As you can see, for some same square feet area, 2 BHK is costlier than 3 BHK. To filter these conditions out:


def less_bhk_higher_cost(df):
    exclude_indices = numpy.array([])
    for location, location_df in df.groupby('site_location'):
        bhk_stats = {}
        for bhk, sub_bhk_df in location_df.groupby('size'):
            bhk_stats[bhk] = {
                'mean': numpy.mean(sub_bhk_df['price_per_sqft']),
                'std': numpy.std(sub_bhk_df['price_per_sqft']),
                'count': sub_bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('size'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = numpy.append(exclude_indices,
                                               bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)
        # for bhk, sub_bhk_df in sub_df.groupby('size'):
        #     if bhk-1 != 0:
        #         smaller_bhk_data = bhk_data.get(bhk-1)
        #         if smaller_bhk_data is not None and smaller_bhk_data['count'] > 5:
        #             unnecessary_df = sub_bhk_df[sub_bhk_df['price_per_sqft'] < smaller_bhk_data['mean']]
        #             new_dataframe = pandas.concat([new_dataframe, unnecessary_df], axis=0, ignore_index=True)
    return df.drop(exclude_indices, axis='index')


df6 = less_bhk_higher_cost(df5)
print(df6.shape)

compare_costs(df5, 'Yerawada')

# Now check how your data looks:
pyplot.hist(df6.price_per_sqft, rwidth=0.8)
pyplot.xlabel("Price Per Square Feet")
pyplot.ylabel("Count")
# pyplot.show()
# Thus, data is nice bell curve. Data is cleaned. Remove unnecessary columns now:
df6['bhk'] = df6['size']
df6.drop(['size', 'price_per_sqft'], axis=1, inplace=True)

# Now, you have a nice, clean, dataset; ready for modelling.
# ----------------------------------------------------------------------------------------
# STEP 3: MODELING:

print(df6.columns)
# One hot encoding for site location:
new_df = pandas.get_dummies(df6['site_location'], drop_first=True)
df7 = pandas.concat([df6.drop(['Super built-up  Area', 'site_location'], axis='columns'), new_df],
                    axis='columns')
print(df7.columns)
print(df7.dtypes)

# Now, training the model:
X = df7.drop(['price'], axis=1)
y = df7['price']
print(X)
print(X.columns)
print(y)


# # Now, we will use GridSearchCV() to get the best model for our prediction:
model_dict = {
    'Linear Regression': {
        'model': LinearRegression(),
        'parameters': {
            'positive': [True, False]
            }
        },
    'Lasso': {
        'model': Lasso(),
        'parameters': {
            'alpha': [1, 2],
            'selection': ['random', 'cyclic']
            }
        },
    'Decision Tree Regressor': {
        'model': DecisionTreeRegressor(),
        'parameters': {
            'criterion': ['mse', 'friedman_mse'],
            'splitter': ['best', 'random']
            }
        }
    }


def best_model(dict):
    scores = []
    for model_name, model_info in dict.items():
        model = model_info['model']
        params = model_info['parameters']

        grid_model = GridSearchCV(model, params, cv=10, return_train_score=False)
        grid_model.fit(X, y)

        scores.append({
            'Model': model_name,
            'Best Score': grid_model.best_score_,
            'Best Parameters': grid_model.best_params_
        })
    return scores


scores_list = best_model(model_dict)
best_score_df = pandas.DataFrame(scores_list)
print(best_score_df[best_score_df['Best Score'] == best_score_df['Best Score'].max()])

# Now, looking at above score, we can say that Lasso is the best model for our predictions.
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

final_model = Lasso(alpha=1, selection='random')
final_model.fit(x_train, y_train)
print(final_model.score(x_test, y_test))

print(X.columns)


def predict(total_sqft, bhk, bath, balcony, location, area_type):
    loc_index = numpy.where(X.columns == location)[0][0]
    try:
        area_type_index = numpy.where(X.columns == area_type)[0][0]
    except IndexError:
        area_type_index = -1

    x = numpy.zeros(len(X.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = balcony
    x[6] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    if area_type_index >= 0:
        x[area_type_index] = 1

    return final_model.predict([x])[0]


print(predict(1600, 2, 2, 1, 'Aundh', 'Carpet  Area'))
# -------------------------------------------------------------------------------------------
# STEP 4: SAVING THE MODEL FOR FUTURE USE
with open('punepriceprediction.pickle', 'wb') as file:
    pickle.dump(final_model, file)

joblib.dump(final_model, 'punepriceprediction_joblib')

columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns_price_prediction.json", "w") as file:
    file.write(json.dumps(columns))
    