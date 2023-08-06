import json
import pickle

import joblib
import numpy
import sklearn
import warnings
warnings.filterwarnings('ignore')

__locations_ = None
__area_type_ = None
__data_columns_ = None
__model_ = None


def load_saved_artifacts():
    print('loading saved artifacts..start')
    global __data_columns_, __locations_, __model_, __area_type_

    with open('./artifacts/columns_price_prediction.json', 'r') as file:
        __data_columns_ = json.load(file)['data_columns']
        __locations_ = __data_columns_[7:]
        __area_type_ = __data_columns_[3:6]

    # with open('./artifacts/punepriceprediction.pickle', 'rb') as file:
    #     MODEL = pickle.load(file)

    __model_ = joblib.load('./artifacts/punepriceprediction_joblib')
    print('loading saved artifacts...done')


def get_location_names():
    return __locations_


def get_area_type_names():
    return __area_type_


def predict(total_sqft, bhk, bath, balcony, location, area_type):
    global __data_columns_, __model_

    try:
        loc_index = __data_columns_.index(location.lower())
    except IndexError:
        loc_index = -1
    try:
        area_type_index = __data_columns_.index(area_type.lower())
    except IndexError:
        area_type_index = -1

    x = numpy.zeros(len(__data_columns_))
    x[0] = total_sqft
    x[1] = bath
    x[2] = balcony
    x[6] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    if area_type_index >= 0:
        x[area_type_index] = 1

    print(loc_index)
    print(area_type_index)
    print(x)
    return round(__model_.predict([x])[0], 2)


if __name__ == '__main__':
    load_saved_artifacts()
    print(__data_columns_)
    print(get_location_names())
    print(get_area_type_names())
    print(predict(2000, 3, 5, 2, 'balaji nagar', 'Carpet  Area'))



