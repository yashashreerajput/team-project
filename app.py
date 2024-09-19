from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import time
import logging
from Pipeline_Handler import PipelineHandler, PreprocessingTransformer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained (MortgagePipelineModel) model from the pickle file 
logger.debug("Start loading the model...")
with open('model\MortgagePipelineModel.pkl', 'rb') as f:
    model = joblib.load(f)
logger.debug("Model loaded")

# Define features for the form input 

cat_cols = {
    'Channel': ['Retail', 'Broker', 'Correspondent'],
    'FirstTimeHomebuyer': ['Y', 'N']
}

num_cols = {
    'MonthsDelinquent': 0,
    'CreditScore': 700,
    'MonthsInRepayment': 12,
    'NumBorrowers': 1,
    'LTV': 80,
    'OCLTV': 75,
    'DTI': 36,
    'OrigInterestRate': 3.5,
    'OrigUPB': 300000,
    'OrigLoanTerm': 360
}

@app.route('/')
def home():
    logger.debug("HomePage route accessed.")
    return render_template('index.html', cat_cols=cat_cols, num_cols=num_cols)

@app.route('/predict', methods=['POST'])
def predict():
    try:
       logger.debug("Prediction route accessed.")
       start_time = time.time()

       form_data = {
        "MonthsDelinquent": [int(request.form['MonthsDelinquent'])],
        "CreditScore": [int(request.form['CreditScore'])],
        "MonthsInRepayment": [int(request.form['MonthsInRepayment'])],
        "NumBorrowers": [int(request.form['NumBorrowers'])],
        "Channel": [request.form['Channel']],
        "LTV": [int(request.form['LTV'])],
        "FirstTimeHomebuyer": [request.form['FirstTimeHomebuyer']],
        "OCLTV": [int(request.form['OCLTV'])],
        "DTI": [int(request.form['DTI'])],
        "OrigInterestRate": [float(request.form['OrigInterestRate'])],
        "OrigUPB": [int(request.form['OrigUPB'])],
        "OrigLoanTerm": [int(request.form['OrigLoanTerm'])]
        }

       logger.debug(f"Form data received: {form_data}")
      
       # Input validation form 
       errors = []

       if not (0 <= form_data['MonthsDelinquent'][0] <= 120):
            errors.append("Please select MonthsDelinquent between 0 and 120.")
       if not (300 <= form_data['CreditScore'][0] <= 850):
            errors.append("Please select CreditScore between 300 and 850.")
       if not (0 <= form_data['MonthsInRepayment'][0] <= 360):
            errors.append("Please select MonthsInRepayment between 0 and 360.")
       if not (1 <= form_data['NumBorrowers'][0] <= 10):
            errors.append("Please select NumBorrowers between 1 and 10.")
       if not (0 <= form_data['LTV'][0]  <= 100):
            errors.append("Please select LTV between 0 and 100.")
       if not (0 <= form_data['OCLTV'][0]  <= 100):
            errors.append("Please select OCLTV between 0 and 100.")
       if not (0 <= form_data['DTI'][0]  <= 100):
            errors.append("Please select DTI between 0 and 100.")
       if not (0 <= form_data['OrigInterestRate'][0]  <= 20):
            errors.append("Please select OrigInterestRate between 0 and 20.")
       if not (0 <= form_data['OrigUPB'][0]  <= 1_000_000):
            errors.append("Please select OrigUPB between 0 and 1,000,000.")
       if not (0 <= form_data['OrigLoanTerm'][0]  <= 360):
            errors.append("Please select OrigLoanTerm between 0 and 360.")

       if errors:
            error_message = "Attention: " + ", ".join(errors)
            logger.error(error_message)
            return render_template('index.html', error_message=error_message,cat_cols=cat_cols, num_cols=form_data)

       # Convert form data to DataFrame
       df = pd.DataFrame(form_data)

       # Perform prediction 
       try:
            logger.debug("Making predictions.")
            y_class_pred, y_reg_pred_new = model.predict(df[['MonthsDelinquent', 'CreditScore', 'MonthsInRepayment',
                                                         'NumBorrowers', 'Channel', 'LTV', 'FirstTimeHomebuyer', 
                                                         'OCLTV', 'DTI', 'OrigInterestRate', 'OrigUPB', 'OrigLoanTerm']])
       except Exception as e:
            logger.error(f"Error During Prediction: {e}")
            return render_template('index.html', error_message=f"An error occurred: {str(e)}",cat_cols=cat_cols, num_cols=form_data)
      
       end_time = time.time()
       elapsed_time = end_time - start_time
       logger.debug(f"Prediction completed in {elapsed_time:.2f} seconds.")
       if len(y_reg_pred_new) > 0:
           return render_template('result.html', 
                           classification_prediction=y_class_pred[0], 
                           regression_prediction=y_reg_pred_new[0],
                           time_taken=elapsed_time)
       else:
           error_message = "Delinquent borrowers have no prepayment"
           return render_template('result.html', 
                           classification_prediction=y_class_pred[0],
                           time_taken=elapsed_time,error_message=error_message)
    except Exception as e:
          
           logger.error(f"Error Occurred During Prediction Process: {e}")
           return render_template('index.html', error_message=f"An error occurred: {str(e)}",cat_cols=cat_cols, num_cols=form_data)

if __name__ == '__main__':
    logger.debug("Flask app Started")
    app.run(debug=True)
