from flask import Flask, jsonify, render_template, request, url_for
import joblib
import os
import numpy as np

app = Flask(__name__)

def isGermany(geo):
    return 1 if geo == 'germany' else 0

def isSpain(geo):
    return 1 if geo == 'spain' else 0

def isHumanResources(role):
    return 1 if role == 'human_resources' else 0 

def isLaboratoryTechnician(role):
    return 1 if role == 'laboratory_technician' else 0 

def isManager(role):
    return 1 if role == 'manager' else 0 

def isManufacturingDirector(role):
    return 1 if role == 'manufacturing_director' else 0 

def isResearchDirector(role):
    return 1 if role == 'research_director' else 0

def isResearchScientist(role):
    return 1 if role == 'research_scientist' else 0

def isSalesExecutive(role):
    return 1 if role == 'sales_executive' else 0

def isSalesRepresentative(role):
    return 1 if role == 'sales_representative' else 0

def isMarried(mar):
    return 1 if mar == 'married' else 0

def isSingle(mar):
    return 1 if mar == 'single' else 0

def isYes(ot):
    return 1 if ot == 'yes' else 0

def result_of_prediction(pred, mode):
    if mode == 1:
        if pred == 1:
            return "Yes, the customer may cancel the subscription to service"
        else:
            return "No, the customer is not going to cancel the subscription"
    elif mode == 2:
        if pred == 1:
            return "Yes, the employee may leave the organization"
        else:
            return "No, the employee will not leave the organization"

@app.route("/")
def index():
    return render_template(
        "index.html",
        css_url=url_for('static', filename='style.css')
        )

@app.route("/customer")
def home1():
    return render_template(
        "customer.html",
        css_url=url_for('static', filename='style.css')
        )

@app.route("/customer-eda")
def customer_eda():
    return render_template(
        "customer_eda.html",
        css_url=url_for('static', filename='style2.css'),
        jointplot1 = url_for('static', filename='customer_jointplot_1.png'),
        jointplot2 = url_for('static', filename='customer_jointplot_2.png'),
        pairplot = url_for('static', filename='customer_pairplot.png')
        )

@app.route("/employee")
def home2():
    return render_template(
        "employee.html",
        css_url=url_for('static', filename='style.css')
        )

@app.route("/employee-eda")
def employee_eda():
    return render_template(
        "employee_eda.html",
        css_url=url_for('static', filename='style2.css'),
        employee_kdeplots = url_for('static', filename='employee_kdeplot.png'),
        employee_pairplots = url_for('static', filename='employee_pairplot.png'),
        scatterplot = url_for('static', filename='feature_importance_scatterplot.png')
        )

@app.route('/customer-prediction',methods=['POST','GET'])
def result1():
    credit_score= int(request.form['credit_score'])
    age=int(request.form['age'])
    tenure= int(request.form['tenure'])
    balance= float(request.form['balance'])
    no_of_products = int(request.form['no_of_products'])
    has_cr_card= int(request.form['has_cr_card'])
    is_active_member= int(request.form['is_active_member'])
    estimated_salary= float(request.form['estimated_salary'])
    geogrphy_germany = isGermany(request.form['geography'])
    geography_spain = isSpain(request.form['geography'])
    gender_male= int(request.form['gender'])

    X= np.array([[credit_score, age, tenure, balance, no_of_products,
                  has_cr_card, is_active_member, estimated_salary, geogrphy_germany, geography_spain, gender_male]])

    scaler_path= r"C:\Users\vikram\Downloads\Customer-Churn-and-Employee-Attrition-Prediction\models\sc.sav"
    model_path= r"C:\Users\vikram\Downloads\Customer-Churn-and-Employee-Attrition-Prediction\models\rf.sav"
    sc=joblib.load(scaler_path)
    X_std= sc.transform(X)
    model= joblib.load(model_path)
    Y_pred=model.predict(X_std)

    prediction = int(Y_pred)
    result = result_of_prediction(prediction, 1)
    return render_template(
        'output.html', 
        css_url=url_for('static', filename='style.css'),
        prediction=prediction, 
        result=result,
        title='Customer Churn Prediction'
        )

@app.route('/employee-prediction',methods=['POST','GET'])
def result2():
    age= int(request.form['age'])
    daily_rate=int(request.form['daily_rate'])
    environment_satisfaction= int(request.form['environment_satisfaction'])
    job_level= int(request.form['job_level'])
    job_satisfaction = int(request.form['job_satisfaction'])
    monthly_income = int(request.form['monthly_income'])
    companies_worked= int(request.form['companies_worked'])
    salary_hike= int(request.form['salary_hike'])
    stock_option= int(request.form['stock_option'])
    total_working_years = int(request.form['total_working_years'])
    work_life_balance = int(request.form['work_life_balance'])
    years_at_company = int(request.form['years_at_company'])
    years_since_last_promotion = int(request.form['years_since_last_promotion'])
    department_research = 1 if request.form['department'] == 'research' else 0
    department_sales = 1 if request.form['department'] == 'sales' else 0
    gender_male = 1 if request.form['gender'] == 'male' else 0
    jobrole_human_resources = isHumanResources(request.form['job_role'])
    jobrole_laboratory_technician = isLaboratoryTechnician(request.form['job_role'])
    jobrole_manager = isManager(request.form['job_role'])
    jobrole_manufacturing_director = isManufacturingDirector(request.form['job_role'])
    jobrole_research_director = isResearchDirector(request.form['job_role'])
    jobrole_research_scientist = isResearchScientist(request.form['job_role'])
    jobrole_sales_executive = isSalesExecutive(request.form['job_role'])
    jobrole_sales_representative = isSalesRepresentative(request.form['job_role'])
    marital_status_married = isMarried(request.form['marital_status'])
    marital_status_single = isSingle(request.form['marital_status'])
    overtime_yes = isYes(request.form['overtime'])

    X2= np.array([[age, daily_rate, environment_satisfaction, job_level, job_satisfaction,
                    monthly_income, companies_worked, salary_hike, stock_option,
                    total_working_years, work_life_balance, years_at_company, years_since_last_promotion,
                    department_research, department_sales, gender_male, jobrole_human_resources,
                    jobrole_laboratory_technician, jobrole_manager, jobrole_manufacturing_director,
                    jobrole_research_director, jobrole_research_scientist, jobrole_sales_executive,
                    jobrole_sales_representative, marital_status_married, marital_status_single,
                    overtime_yes]])

    scaler_path2= r"C:\Users\vikram\Downloads\Customer-Churn-and-Employee-Attrition-Prediction\models\sc_employee.sav"
    model_path2=r"C:\Users\vikram\Downloads\Customer-Churn-and-Employee-Attrition-Prediction\models\rf_employee.sav"
    sc2=joblib.load(scaler_path2)
    X_std2= sc2.transform(X2)
    model2= joblib.load(model_path2)
    Y_pred2=model2.predict(X_std2)

    prediction = int(Y_pred2)
    result = result_of_prediction(prediction, 2)
    return render_template(
        'output.html', 
        css_url=url_for('static', filename='style.css'),
        prediction=prediction, 
        result=result,
        title='Employee Attrition Prediction'
        )


if __name__ == "__main__":
    app.run(debug=True, port=4123)
