from django.shortcuts import render
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
# Create your views here.
def home(request):
    return render(request,'home.html')


def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from catboost import CatBoostClassifier
import pandas as pd
import joblib
def prediction(request):
    clf=joblib.load('model/catBoost_model.pkl')
    feature_labels = {
        "Gender": "Gender",
        "Use phone for class notes": "Use phone for class notes",
        "Buy books from mobile": "Buy books from mobile",
        "Phone battery lasts a day": "Phone battery lasts a day",
        "Run for charger when battery dies": "Run for charger when battery dies",
        "Worry about losing phone": "Worry about losing phone",
        "Take phone to bathroom": "Take phone to bathroom",
        "Use phone in social gathering": "Use phone in social gathering",
        "Check phone before sleep/after waking up": "Check phone before sleep/after waking up",
        "Keep phone next to you while sleeping": "Keep phone next to you while sleeping",
        "Check emails/missed calls/texts during class time": "Check emails/missed calls/texts during class time",
        "Rely on phone when things get awkward": "Rely on phone when things get awkward",
        "On phone while watching TV/eating": "On phone while watching TV/eating",
        "Panic attack if phone left elsewhere": "Panic attack if phone left elsewhere",
        "Respond to messages/check phone on date": "Respond to messages/check phone on date",
        "Use phone for playing games": "Use phone for playing games",
        "Can live a day without phone": "Can live a day without phone"
    }

    if request.method == "POST":
        data_dict = {feature: int(request.POST.get(feature, 0)) for feature in feature_labels}
        df = pd.DataFrame([data_dict])
        predict = clf.predict(df)[0]
        print(predict)
        ot=predict[-1]
        labels=["HIGH","LOW","MODERATE"]
        outcome = labels[ot]
        return render(request, 'outcome.html', {'outcome': outcome})

    return render(request, 'prediction.html', {"feature_labels": feature_labels})


from django.core.files.storage import default_storage
le=LabelEncoder()
dataloaded=False
global X_train,X_test,y_train,y_test
global df
def Upload_data(request):
    load=True
    global df,dataloaded
    global X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path))
        df["Gender"]=le.fit_transform(df["Gender"])
        df["Phone Dependency Level"]=le.fit_transform(df["Phone Dependency Level"])
        sns.set(style="darkgrid")  # Set the style of the plot
        plt.figure(figsize=(8, 6))  # Set the figure size
        ax = sns.countplot(x='Phone Dependency Level', data=df)
        plt.title("Count Plot")  # Add a title to the plot
        plt.xlabel("Categories")  # Add label to x-axis
        plt.ylabel("Count")  # Add label to y-axis
        # Annotate each bar with its count value
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.xticks(rotation=90)
        plt.show()
        x=df.iloc[:,2:19]
        y=df.iloc[:,-1]
        X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=42)
        default_storage.delete(file_path)
        outdata=df.head(100)
        dataloaded=True
        return render(request,'train.html',{'temp':outdata.to_html()})
    return render(request,'train.html',{'upload':load})
labels=["LOW","MEDIUM","HIGH"]
#defining global variables to store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
def calculateMetrics(algorithm, testY,predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    
def DTC(request):
    if dataloaded == False:
        return redirect('upload')
    if os.path.exists('model/DT_model.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/DT_model.pkl')
        print("Model loaded successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("DT_model", predict, y_test)
    else:
        clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/DT_model.pkl')
        print("Model saved successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("DT_model", predict, y_test)
    return render(request,'train.html',
                  {'algorithm':'Decision Tree classification',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
def RFC(request):
    if dataloaded == False:
        return redirect('upload')
    if os.path.exists('model/RF_model.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/RF_model.pkl')
        print("Model loaded successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("RF_model", predict, y_test)
    else:
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/RF_model.pkl')
        print("Model saved successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("RF_model", predict, y_test)
    return render(request,'train.html',
                  {'algorithm':'Random Forest',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
    
def catboost_view(request):
    if dataloaded == False:
        return redirect('upload')
    # Check if the model file exists
    if os.path.exists('model/CatBoost_model.pkl'):
        # Load the trained model from the file
        clf = joblib.load('model/CatBoost_model.pkl')
        print("Model loaded successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("CatBoost_model", predict, y_test)
    else:
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, 
                                loss_function='MultiClass', verbose=0)  # Use MultiClass for multiple categories
        clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'model/CatBoost_model.pkl')
        print("Model saved successfully.")
        predict = clf.predict(X_test)
        calculateMetrics("CatBoost_model", predict, y_test)
    return render(request,'train.html',
                  {'algorithm':'CatBoostClassifier',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})