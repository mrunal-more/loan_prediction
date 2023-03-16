from flask import Flask,render_template,request,jsonify,redirect,url_for
import numpy as np
import pickle
import json

with open("artifacts/dict_file.json","r") as file:
    dict_file = json.load(file)

with open("artifacts/Loan_Pred_Model.pkl","rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_data",methods=["POST","GET"])
def get_data():

    data                = request.form
    Gender              = data["html_gender"]
    Married             = data["html_married"]
    Dependents          = data["html_dependents"]
    Education           = data["html_education"]
    Self_Employed       = data["html_self_employed"]
    ApplicantIncome     = data["html_applicantincome"]
    CoapplicantIncome   = data["html_coapplicantincome"]
    LoanAmount          = data["html_loanamount"]
    Loan_Amount_Term    = data["html_loan_amount_term"]
    Credit_History      = data["html_credit_history"]
    Property_Area       = data["html_property_area"]

    user_data       = np.zeros(len(dict_file["Column_Names"]))
    user_data[0]    = dict_file["Gender"][Gender]
    user_data[1]    = dict_file["Married"][Married]
    user_data[2]    = dict_file["Dependents"][Dependents]
    user_data[3]    = dict_file["Education"][Education]
    user_data[4]    = dict_file["Self_Employed"][Self_Employed]
    user_data[5]    = ApplicantIncome
    user_data[6]    = CoapplicantIncome
    user_data[7]    = LoanAmount
    user_data[8]    = Loan_Amount_Term
    user_data[9]    = Credit_History
    user_data[10]   = dict_file["Property_Area"][Property_Area]

    result = model.predict([user_data])[0]
    if result == 1:
        result = "Congratulation You Are Eligible For Loan"
    else: 
        result = " Sorry! You Are Not Eligible For Loan"
    print(result)
    return render_template('index.html',Loan_Status=result) 


if __name__ == "__main__":
    app.run(host = '0.0.0.0')