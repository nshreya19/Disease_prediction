from flask_mysqldb import MySQL
import os
from flask import Flask, request,session,url_for, render_template, redirect, flash,  jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.svm import SVC
import mysql.connector
from flask_mysqldb import MySQL
import logging

# flask app
app = Flask(__name__)
app.secret_key = 'Spandana'  # Set a secret key for flashing messages

app.config['MYSQL_HOST'] = 'localhost'  
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Sbv@1018'
app.config['MYSQL_DB'] = 'spandana'
mysql = MySQL(app)

logging.basicConfig(level=logging.DEBUG)

# Define file paths
# Define file paths
sym_des_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\symtoms_df.csv"
precautions_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\precautions_df.csv"
workout_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\workout_df.csv"
description_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\description.csv"
medications_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\medications.csv"
diets_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\diets.csv"
model_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\models\svc.pkl"
doctors_path = r"C:\Users\acer\Desktop\medicine_rec\Medicine-recommendation\datasets\Doctors.csv"

# Function to load CSV files
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Load CSV files
sym_des = load_csv(sym_des_path)
precautions = load_csv(precautions_path)
workout = load_csv(workout_path)
description = load_csv(description_path)
medications = load_csv(medications_path)
diets = load_csv(diets_path)
doctors = load_csv(doctors_path)

# Load model
if os.path.exists(model_path):
    svc = pickle.load(open(model_path, 'rb'))
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Define helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    # Filter doctors for the given disease
    doctor_recommendations = doctors[doctors['Disease'] == dis]
    doctors_list = []
    for _, row in doctor_recommendations.iterrows():
        doctors_list.append({
            "name": row['Doctor_Name'],
            "specialization": row['Specialty'],
            "contact": row['Contact_Info'],
            "is_available": row['Is_Available_Now'],
            "availability": row['When_Available_If_Not'],
            "photo": row['Doctor_photo']
        })

    return desc, pre, med, die, wrkout, doctors_list

# Define symptoms and diseases
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# BMI calculation
def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

# Period Tracker
def predict_next_period(last_period_date):
    last_period = datetime.strptime(last_period_date, '%Y-%m-%d')
    next_period = last_period + timedelta(days=28)
    return next_period.strftime('%Y-%m-%d')

# Creating routes
@app.route("/")
def welcome():
    return render_template("welcome.html")

# Index route
@app.route('/index')
def index():
    if 'user_id' in session:
        user_name = session['user_name']
        return render_template('index.html', user_name=user_name)
    else:
        return redirect(url_for('log_in'))

# Define a route for the home page and prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        # Check if the user is logged in
        if 'user_id' in session:
            symptoms = request.form.get('symptoms')
            print(symptoms)
            if symptoms == "Symptoms":
                message = "Please either write symptoms or you have written misspelled symptoms"
                return render_template('index.html', message=message)
            else:
                # Split the user's input into a list of symptoms (assuming they are comma-separated)
                user_symptoms = [s.strip() for s in symptoms.split(',')]
                # Remove any extra characters, if any
                user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
                predicted_disease = get_predicted_value(user_symptoms)
                dis_des, precautions, medications, rec_diet, workout, doctors = helper(predicted_disease)

                my_precautions = []
                for i in precautions[0]:
                    my_precautions.append(i)

                return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                       my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                       workout=workout, doctors=doctors, user_name=session.get('user_name'))

        else:
            # If the user is not logged in, redirect to the login page
            return redirect(url_for('login'))
    else:
        # Render the home page for GET requests
        return render_template('welcome.html')

# Route for patient registration form
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        age = int(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        password = request.form['password']
        
        try:
            # Create a cursor object to execute SQL queries
            cursor = mysql.connection.cursor()

            # SQL query to insert data into user table
            insert_query = """INSERT INTO user (Name, Email, PhoneNumber, Age, Gender, Location, Password)
                VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            
            # Execute the SQL query to insert data
            cursor.execute(insert_query, (name, email, phone, age, gender, location, password))
            
            # Commit changes to the database
            mysql.connection.commit()

            # Close cursor
            cursor.close()

            # Flash a success message
            flash("Registration successful!", "success")

            # Redirect to a success page or do whatever you want
            return redirect('/login')  # Redirect to the registration page to clear the form

        except Exception as e:
            # Rollback changes in case of any error
            mysql.connection.rollback()
            flash(f"Error: {e}", "danger")
            print(f"Error: {e}")  # Print the error message for debugging purposes

    # Render the registration form template for GET requests
    return render_template('register.html')
# about view function and path
@app.route('/about')
def about():
    return render_template("about.html")

# Register form function and path
@app.route('/reg_form')
def reg():
    return render_template("register.html")

# contact view function and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view function and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view function and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

# login function and path
@app.route('/login')
def login():
    return render_template("login.html")
# Login and index combined route

@app.route('/log_in', methods=['GET', 'POST'])
def log_in():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM user WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()

            if user and password == user[7]:  
                session['user_id'] = user[0]  
                session['user_name'] = user[1]  
                return redirect(url_for('index'))
            else:
                flash('Invalid email or password')
                return redirect(url_for('log_in'))
        except Exception as e:
            logging.error("An error occurred during authentication:", exc_info=True)
            flash('An error occurred. Please try again later.')
            return redirect(url_for('log_in'))

    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    # Clear the session or any other authentication tokens
    session.pop('user_name', None)
    # Add any other session keys to clear
    session.clear()  # Optionally clear all session data
    return render_template("welcome.html")



if __name__ == '__main__':
    app.run(debug=True)
