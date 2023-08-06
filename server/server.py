from flask import Flask, request, jsonify
import sklearn
import util

app = Flask(__name__)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict_price', methods=['GET', 'POST'])
def predict_price():
    total_sqft = float(request.form['total_sqft'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    balcony = int(request.form['balcony'])
    location = request.form['location']
    area_type = request.form['area_type']

    response = jsonify({
        'estimated_price': util.predict(total_sqft, bhk, bath, balcony, location, area_type)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# The given Python code sets up a Flask web server for a Home Price Prediction application. The server has two routes
# defined: one for getting location names and another for predicting home prices. Let's go through the code step by
# step:
# from flask import Flask, request, jsonify: Importing necessary modules from Flask for creating a web server, handling
# requests, and returning JSON responses.
# import sklearn and import util: Importing the required machine learning-related libraries (not explicitly used in the
# code provided) and a custom module named util that contains functions for location names retrieval and home price
# prediction.
# app = Flask(__name__): Creating a Flask application instance.
# @app.route('/get_location_names', methods=['GET']): This is a decorator used to associate the /get_location_names URL
# path with the get_location_names function. The methods=['GET'] argument specifies that this route should only handle
# HTTP GET requests.
# def get_location_names(): This function is defined to handle the /get_location_names route. It returns a JSON
# response containing a list of location names by calling the util.get_location_names() function.
# @app.route('/predict_price', methods=['GET', 'POST']): This is another decorator used to associate the /predict_price
# URL path with the predict_price function. The methods=['GET', 'POST'] argument indicates that this route should handle
# both HTTP GET and POST requests.
# def predict_price():: This function is defined to handle the /predict_price route. It retrieves input parameters
# (total_sqft, bhk, bath, balcony, location, and area_type) from the request's form data, converts them to appropriate
# data types, and then calls the util.predict() function to make a prediction for the home price. The prediction result
# is returned as a JSON response.
# if __name__ == '__main__': This block of code is executed when the script is run as the main module (i.e., not
# imported as a module in another script).
# print('Starting Python Flask Server For Home Price Prediction...'): A message indicating that the server is starting
# is printed.
# util.load_saved_artifacts(): This line calls the load_saved_artifacts() function from the util module. It is expected
# that this function loads pre-trained machine learning model artifacts and any necessary data required for prediction.
# Since this function is not provided in the code, its functionality is not explicitly visible here.
# app.run(): This starts the Flask development server, serving the application. The server listens for incoming requests
# and routes them to the appropriate functions defined for different URL paths.
# In summary, this Python code sets up a Flask web server that has two routes: /get_location_names for obtaining
# location names and /predict_price for predicting home prices. The server uses the util module to retrieve location
# names and make price predictions based on the provided input parameters.


if __name__ == '__main__':
    print('Starting Python Flask Server For Home Price Prediction...')
    util.load_saved_artifacts()
    app.run()
