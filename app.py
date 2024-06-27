# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np

# # Initialize the Flask application
# app = Flask(__name__)

# # Load the model
# model = joblib.load('salary_prediction_all_countries.h5')

# # Define the mapping dictionaries
# job_titles_dict = {'Data Scientist':0, 'Machine Learning Engineer':1, 'Front End Developer':2, 'Back End Developer':3, 'Data Engineer':4,
#               'Full Stack Developer':5, 'Data Analyst':6, 'Designer':7, 'AI Engineer':8, 'Computer Vision Engineer':9}

# company_name_dict = {'DeepMind': 0, 'Faculty': 1, 'Ocado Group': 2, 'ARM Holdings': 3, 'Darktrace': 4,
#                      'FiveAI': 5, 'StatusToday': 6, 'Tractable': 7, 'Sage Group': 8, 'Graphcore': 9,
#                      'Dataiku': 10, 'Shift Technology': 11, 'DreamQuark': 12, 'Snips': 13,
#                      'Dassault Systèmes': 14, 'Thales Group': 15, 'Meero': 16, 'Craft AI': 17,
#                      'Atos': 18, 'Capgemini': 19, 'Coveo': 20, 'Integrate.ai': 21, 'Shopify': 22,
#                      'OpenText': 23, 'CGI Inc': 24, 'Kira Systems': 25, 'BenchSci': 26, 'BlueDot': 27,
#                      'Element AI': 28, 'RBC': 29, 'Bosch': 30, 'Arago': 31, 'Deutsche': 32,
#                      'Siemens': 33, 'Celonis': 34, 'Merantix': 35, 'SAP': 36, 'Twenty Billion Neurons': 37,
#                      'German Autolabs': 38, 'Konux': 39, 'Health': 40, 'useReady': 41, 'Microsoft': 42,
#                      'Centro': 43, 'Facebook': 44, 'openAI': 45, 'Tempus Labs': 46,
#                      'Software Engineering Institute': 47, 'Porch': 48, 'Google': 49}

# countries_dict = {'England': 0, 'France': 1, 'Canada': 2, 'Germany': 3, 'United States': 4}

# company_size_dict = {'S':0, 'M':1, 'L':2}

# english_level_dict = {'Intermediate':0, 'Professional':1, 'Full Professional':2}


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
#     job_title = data['job_title']
#     country = data['country']
#     years_of_experience = data['years_of_experience']
#     company_size = data['company_size']
#     company_name = data['company_name']
#     english_level = data['english_level']
    
#     # Validate input
#     if not job_title or not country or not years_of_experience or not company_size or not company_name:
#         return jsonify({'error': 'All fields are required!'}), 400
    
#     try:
#         years_of_experience = int(years_of_experience)
#     except ValueError:
#         return jsonify({'error': 'Years of experience must be a valid integer!'}), 400
    
#     # Map the inputs
#     job_title = job_titles_dict.get(job_title)
#     country = countries_dict.get(country)
#     company_size = company_size_dict.get(company_size)
#     company_name = company_name_dict.get(company_name)
#     english_level = english_level_dict.get(english_level)
    
#     if job_title is None or country is None or company_size is None or company_name is None:
#         return jsonify({'error': 'Invalid input values!'}), 400
    
#     # Create the input array for the model
#     input_data = np.array([[job_title, country,company_name, years_of_experience, company_size,  english_level]])    
#     # Predict the salary
#     prediction = model.predict(input_data)[0]
    
#     # # Round the prediction to the nearest multiple of 500
#     # rounded_prediction = 500 * round(prediction / 500)
    
#     # Function to round to the nearest number divisible by 10
#     def round_to_nearest_10(value):
#         return np.round(value / 10) * 10

#     # Round the predicted salary
#     rounded_salary = int(round_to_nearest_10(prediction))
    
#     # Return the prediction as a JSON response
#     return render_template('prediction.html', salary=rounded_salary)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# # Example: Saving the model
# model = XGBRegressor()
# Assuming you've trained your model
# Load the model
model = joblib.load('salary_prediction_all_countries_dt.h5')

# Define the mapping dictionaries
job_titles_dict = {'Data Scientist': 0, 'Machine Learning Engineer': 1, 'Front End Developer': 2, 'Back End Developer': 3, 'Data Engineer': 4,
                   'Full Stack Developer': 5, 'Data Analyst': 6, 'Designer': 7, 'AI Engineer': 8, 'Computer Vision Engineer': 9}

company_name_dict = {'DeepMind': 0, 'Faculty': 1, 'Ocado Group': 2, 'ARM Holdings': 3, 'Darktrace': 4,
                     'FiveAI': 5, 'StatusToday': 6, 'Tractable': 7, 'Sage Group': 8, 'Graphcore': 9,
                     'Dataiku': 10, 'Shift Technology': 11, 'DreamQuark': 12, 'Snips': 13,
                     'Dassault Systèmes': 14, 'Thales Group': 15, 'Meero': 16, 'Craft AI': 17,
                     'Atos': 18, 'Capgemini': 19, 'Coveo': 20, 'Integrate.ai': 21, 'Shopify': 22,
                     'OpenText': 23, 'CGI Inc': 24, 'Kira Systems': 25, 'BenchSci': 26, 'BlueDot': 27,
                     'Element AI': 28, 'RBC': 29, 'Bosch': 30, 'Arago': 31, 'Deutsche': 32,
                     'Siemens': 33, 'Celonis': 34, 'Merantix': 35, 'SAP': 36, 'Twenty Billion Neurons': 37,
                     'German Autolabs': 38, 'Konux': 39, 'Health': 40, 'useReady': 41, 'Microsoft': 42,
                     'Centro': 43, 'Facebook': 44, 'openAI': 45, 'Tempus Labs': 46,
                     'Software Engineering Institute': 47, 'Porch': 48, 'Google': 49}

countries_dict = {'England': 0, 'France': 1, 'Canada': 2, 'Germany': 3, 'United States': 4}

company_size_dict = {'S': 0, 'M': 1, 'L': 2}

english_level_dict = {'Intermediate': 0, 'Professional': 1, 'Full Professional': 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type == 'application/json':
        data = request.json
    else:  # Assume form data
        data = request.form

    job_title = data.get('job_title')
    country = data.get('country')
    years_of_experience = data.get('years_of_experience')
    company_size = data.get('company_size')
    company_name = data.get('company_name')
    english_level = data.get('english_level')

    # Validate input
    if not job_title or not country or not years_of_experience or not company_size or not company_name:
        return jsonify({'error': 'All fields are required!'}), 400

    try:
        years_of_experience = int(years_of_experience)
    except ValueError:
        return jsonify({'error': 'Years of experience must be a valid integer!'}), 400

    # Map the inputs
    job_title = job_titles_dict.get(job_title)
    country = countries_dict.get(country)
    company_size = company_size_dict.get(company_size)
    company_name = company_name_dict.get(company_name)
    english_level = english_level_dict.get(english_level)

    if job_title is None or country is None or company_size is None or company_name is None:
        return jsonify({'error': 'Invalid input values!'}), 400

    # Create the input array for the model
    input_data = np.array([[job_title, country, company_name, years_of_experience, company_size, english_level]])

    # Predict the salary
    prediction = model.predict(input_data)[0]

    # Round the predicted salary
    def round_to_nearest_10(value):
        return np.round(value / 10) * 10

    rounded_salary = int(round_to_nearest_10(prediction))

    # Return the prediction as a JSON response
    return jsonify({'prediction': rounded_salary})

if __name__ == '__main__':
    app.run(debug=True)
