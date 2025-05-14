from django.shortcuts import render
from django.conf import settings
from .models import UserRegistrationModel
from .forms import UserRegistrationForm
from django.contrib import messages
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegister.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def DatasetView(request):
    from django.conf import settings
    import pandas as pd 
    path = settings.MEDIA_ROOT + "//" + 'dynamic_pricing.csv'
    d = pd.read_csv(path)   
    # Drop the last column
    if not d.empty:
        d = d.iloc[:]  
    # d = d.head(50)  
    print(d)
    return render(request,'users/DatasetView.html', {'d': d})
DATASET_PATH = os.path.join(settings.MEDIA_ROOT, 'dynamic_pricing.csv')

def Training(request):
    df = pd.read_csv(DATASET_PATH)
    X = df.drop('Historical_Cost_of_Ride', axis=1)
    y = df['Historical_Cost_of_Ride']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    xgb_model = XGBRegressor()
    rf_model = RandomForestRegressor()

    xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    xgb_grid_search.fit(X_train_encoded, y_train)

    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    rf_grid_search.fit(X_train_encoded, y_train)

    best_xgb_model = xgb_grid_search.best_estimator_
    best_rf_model = rf_grid_search.best_estimator_

    joblib.dump(best_xgb_model, os.path.join(settings.MEDIA_ROOT, 'xgb_model.pkl'))
    joblib.dump(best_rf_model, os.path.join(settings.MEDIA_ROOT, 'rf_model.pkl'))
    joblib.dump(X_train_encoded, os.path.join(settings.MEDIA_ROOT, 'X_train_encoded.pkl'))
    time_encoder = LabelEncoder()
    time_encoder.fit(X_train['Time_of_Booking'])
    joblib.dump(time_encoder, os.path.join(settings.MEDIA_ROOT, 'time_encoder.pkl'))

    y_pred_xgb = best_xgb_model.predict(X_test_encoded)
    y_pred_rf = best_rf_model.predict(X_test_encoded)

    xgb_score = best_xgb_model.score(X_test_encoded, y_test)
    rf_score = best_rf_model.score(X_test_encoded, y_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # Metrics for XGBRegressor
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Metrics for RandomForestRegressor
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("XGBRegressor Best Score:", xgb_score)
    print("RandomForestRegressor Best Score:", rf_score)

    return render(request, 'users/Training.html', {
        'xgb_score': xgb_score,
        'rf_score': rf_score,
        'mae_xgb': mae_xgb,
        'mse_xgb': mse_xgb,
        'rmse_xgb': rmse_xgb,
        'r2_xgb': r2_xgb,
        'mae_rf': mae_rf,
        'mse_rf': mse_rf,
        'rmse_rf': rmse_rf,
        'r2_rf': r2_rf})

def Prediction(request):
    if request.method == 'POST':
        # Collect input data from the form
        new_input = {
            'Number_of_Riders': float(request.POST.get('Number_of_Riders')),
            'Number_of_Drivers': float(request.POST.get('Number_of_Drivers')),
            'Location_Category': request.POST.get('Location_Category'),
            'Customer_Loyalty_Status': request.POST.get('Customer_Loyalty_Status'),
            'Number_of_Past_Rides': float(request.POST.get('Number_of_Past_Rides')),
            'Average_Ratings': float(request.POST.get('Average_Ratings')),
            'Time_of_Booking': request.POST.get('Time_of_Booking'),
            'Vehicle_Type': request.POST.get('Vehicle_Type'),
            'Expected_Ride_Duration': float(request.POST.get('Expected_Ride_Duration'))
        }

        # Load the trained models and encoders
        try:
            xgb_model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'xgb_model.pkl'))
            X_train_encoded = joblib.load(os.path.join(settings.MEDIA_ROOT, 'X_train_encoded.pkl'))
            
            # Load encoders
            location_encoder = joblib.load(os.path.join(settings.MEDIA_ROOT, 'location_encoder.pkl'))
            loyalty_encoder = joblib.load(os.path.join(settings.MEDIA_ROOT, 'loyalty_encoder.pkl'))
            vehicle_encoder = joblib.load(os.path.join(settings.MEDIA_ROOT, 'vehicle_encoder.pkl'))
            time_encoder = joblib.load(os.path.join(settings.MEDIA_ROOT, 'time_encoder.pkl'))

        except Exception as e:
            return render(request, 'users/Prediction.html', {'error': f"Error loading model or encoder: {e}"})

        # Encode categorical features
        new_input['Location_Category'] = location_encoder.transform([new_input['Location_Category']])[0]
        new_input['Customer_Loyalty_Status'] = loyalty_encoder.transform([new_input['Customer_Loyalty_Status']])[0]
        new_input['Vehicle_Type'] = vehicle_encoder.transform([new_input['Vehicle_Type']])[0]
        new_input['Time_of_Booking'] = time_encoder.transform([new_input['Time_of_Booking']])[0]

        # Create a DataFrame from the input
        new_input_df = pd.DataFrame([new_input])

        # One-hot encode the new input DataFrame
        new_input_df = pd.get_dummies(new_input_df, drop_first=True)

        # Align columns with the encoded training set
        new_input_df = new_input_df.reindex(columns=X_train_encoded.columns, fill_value=0)

        # Predict using XGBoost
        prediction_xgb = xgb_model.predict(new_input_df)

        return render(request, 'users/Prediction.html', {
            'prediction_xgb': prediction_xgb[0] 
        })

    return render(request, 'users/Prediction.html')