from flask import Flask, render_template,request,make_response, session,jsonify
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import sys

import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import geocoder
from random import randint
from math import sqrt
from math import pi
from math import exp

app = Flask(__name__)
app.secret_key = "secret key"





@app.route('/')
def index():       
    return render_template('index.html')

@app.route('/index')
def indexnew():    
    return render_template('index.html')

@app.route('/register')
def register():    
    return render_template('register.html')

@app.route('/login')
def login(): 
    session.pop('graph', None) 
    return render_template('login.html')




""" REGISTER CODE  """

@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
    connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
    uname = request.args['uname']
    name = request.args['name']
    pswd = request.args['pswd']
    email = request.args['email']
    phone = request.args['phone']
    addr = request.args['addr']
    value = randint(123, 99999)
    uid="User"+str(value)
    print(addr)
        
    cursor = connection.cursor()
    sql_Query = "insert into userdata values('"+uid+"','"+uname+"','"+name+"','"+pswd+"','"+email+"','"+phone+"','"+addr+"')"
        
    cursor.execute(sql_Query)
    connection.commit() 
    connection.close()
    cursor.close()
    msg="Data stored successfully"
    #msg = json.dumps(msg)
    resp = make_response(json.dumps(msg))
    
    print(msg, flush=True)
    #return render_template('register.html',data=msg)
    return resp




"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    connection=mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['pswd']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
        
   




@app.route('/dashboard')
def dashboard():   
    try: 
        connection=mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
        cursor = connection.cursor()
        sq_query="select count(*) from userdata"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        rcount = int(data[0][0])
        print(rcount, flush=True)

        sq_query="select count(TypeofAccident) from accdata where TypeofAccident='Over Speed'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        regcount = int(data[0][0])
        print(regcount, flush=True)

        sq_query="select count(TypeofAccident) from accdata where TypeofAccident='Drink & Drive'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        ccount = int(data[0][0])
        print(ccount, flush=True)

        sq_query="select count(*) from accdata"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        dscount = int(data[0][0])
        print(dscount, flush=True)


        
        connection.commit() 
        connection.close()
        cursor.close()
        return render_template('dashboard.html',rcount=rcount,regcount=regcount,ccount=ccount,dscount=dscount)
    except:
        print("No Data to be Displayed")
        return render_template('dashboard.html')

@app.route('/dataloader')
def dataloader():
    return render_template('dataloader.html')





@app.route('/predict')
def predict():
    connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
    cursor = connection.cursor()
    query="SELECT DISTINCT Location FROM accdata;"
    cursor.execute(query)
    locations = [location[0] for location in cursor.fetchall()]    
    connection.commit()      
    connection.close()
    cursor.close()
    return render_template('predict.html',locations=locations)






@app.route('/getaccidents', methods=['POST'])
def get_accident_types():
    location = request.form['location']
    connection = mysql.connector.connect(host='localhost', database='flaskradb', user='root', password='')
    cursor = connection.cursor()
    query = "SELECT DISTINCT TypeofAccident FROM accdata WHERE Location = '"+location+"';"
    cursor.execute(query)
    accident_types = [accident[0] for accident in cursor.fetchall()]
    connection.commit()
    connection.close()
    cursor.close()
    return jsonify(accident_types)






@app.route('/get_accident_count', methods=['POST'])
def get_accident_count():
    location = request.form['location']
    accident_type = request.form['accidentType']
    connection = mysql.connector.connect(host='localhost', database='flaskradb', user='root', password='')
    cursor = connection.cursor()
    query = "SELECT COUNT(*) FROM accdata WHERE TypeofAccident = '"+accident_type+"' AND Location = '"+location+"' AND YEAR(STR_TO_DATE(Date, '%d-%m-%Y')) = ( SELECT MAX(YEAR(STR_TO_DATE(Date, '%d-%m-%Y'))) FROM accdata WHERE TypeofAccident = '"+accident_type+"' );"
    print(query)
    cursor.execute(query)
    accident_count = cursor.fetchone()[0]
    connection.commit()
    connection.close()
    cursor.close()
    
    return jsonify({'count': (round(accident_count+(accident_count*0.22))), 'type': accident_type, 'location': location})







# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated





#Fuzzy 

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    
        try:
                separated = separate_by_class(dataset)
                summaries = dict()
                for class_value, rows in separated.items():
                        summaries[class_value] = summarize_dataset(rows)
                return summaries
        except:
            print('')

# Calculate the Tree probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent




@app.route('/cleardataset', methods = ['POST'])
def cleardataset():
    connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
    cursor = connection.cursor()
    query="delete from accdata"
    cursor.execute(query)
    connection.commit()      
    connection.close()
    cursor.close()
    return render_template('dataloader.html')

@app.route('/cpred')
def cpred():
    print('reached')
    os.system('python Classifier.py')
    try: 
        connection=mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
        cursor = connection.cursor()
        sq_query="select count(*) from userdata"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        rcount = int(data[0][0])
        print(rcount, flush=True)

        sq_query="select count(TypeofAccident) from accdata where TypeofAccident='Over Speed'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        regcount = int(data[0][0])
        print(regcount, flush=True)

        sq_query="select count(TypeofAccident) from accdata where TypeofAccident='Drink & Drive'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        ccount = int(data[0][0])
        print(ccount, flush=True)

        sq_query="select count(*) from accdata"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        dscount = int(data[0][0])
        print(dscount, flush=True)


        
        connection.commit() 
        connection.close()
        cursor.close()
        session["graph"] = "yes"
        return render_template('dashboard.html',rcount=rcount,regcount=regcount,ccount=ccount,dscount=dscount)
    except:
        print("No Data to be Displayed")
        return render_template('dashboard.html')
    #return render_template('dashboard.html')


@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
        cursor = connection.cursor()
    
        prod_mas = request.files['prod_mas']
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("C:\\Upload\\", filename))

        #csv reader
        fn = os.path.join("C:\\Upload\\", filename)

        # initializing the titles and rows list 
        fields = [] 
        rows = []
        
        with open(fn, 'r') as csvfile:
            # creating a csv reader object 
            csvreader = csv.reader(csvfile)  
  
            # extracting each data row one by one 
            for row in csvreader:
                rows.append(row)
                print(row)

        try:     
            #print(rows[1][1])       
            for row in rows[1:]: 
                # parsing each column of a row
                if row[0][0]!="":                
                    query="";
                    query="insert into accdata values (";
                    for col in row: 
                        query =query+"'"+col+"',"
                    query =query[:-1]
                    query=query+");"
                print("query :"+str(query), flush=True)
                cursor.execute(query)
                connection.commit()
        except:
            print("An exception occurred")
        csvfile.close()
        
        print("Filename :"+str(prod_mas), flush=True)       
        
        
        connection.close()
        cursor.close()
        return render_template('dataloader.html',data="Data loaded successfully")



@app.route('/planning')
def planning():
    connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
    sql_select_Query = "SELECT * FROM ( SELECT *, ROW_NUMBER() OVER (PARTITION BY Location ORDER BY Date DESC) AS row_num FROM accdata ) AS ranked_data WHERE row_num <= 10;"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()

    return render_template('planning.html', data=data)




@app.route('/forecast')
def forecast():
    g = geocoder.ip('me')
    print(g.latlng[0])
    print(g.latlng[1])
    print(g)
    
    abc=str(g[0])
    xyz=abc.split(', ')
    print(xyz[0][1:])
    print(xyz[1])
    prediction="Heart Attack"
    loc=xyz[0][1:]+", "+xyz[1]
    
    return render_template('forecast.html',glat=g.latlng[0],glon=g.latlng[1],curloc=loc,pred=prediction)






@app.route('/predictdata' ,methods = ['POST','GET'])
def predictdata():
    if request.method == 'GET':
        patid = request.args['patid']
        print(patid)
        patname = request.args['patname']
        print(patname)
        age = request.args['age']
        gender = request.args['gender']
        date = request.args['date']
        doctor = request.args['doctor']
        cs = request.args['cs']
        ah = request.args['ah']
        cc = request.args['cc']
        diags = request.args['diags']
        diag = request.args['diag']
        hb = request.args['hb']
        plt = request.args['plt']
        print(plt)
        
        #g = geocoder.ip('me')
        #print(g.latlng[0])
        #print(g.latlng[1])
        #print(g)
        
        #abc=str(g[0])
        #xyz=abc.split(', ')
        #print(xyz[0][1:])
        #print(xyz[1])


        connection = mysql.connector.connect(host='localhost',database='flaskradb',user='root',password='')
        sql_select_Query = "select * from accdata"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        data = cursor.fetchall()
        

        #22222222
        X=[]
        Y=[]
        for i in range(len(data)):
            X.append(len(data[11]))
            Y.append(len(data[8]))


        symptoms=[]
        for i in range(len(data)):
            symptoms.append(data[11])

        symptomval=[]
        for i in range(len(symptoms)):
            symptomval.append(symptoms[i][11])
            
        print(symptomval)

        diagsplit=diags.split(',')
        print(diagsplit)

        counter=0
        ts=len(diagsplit)
        for j in range(ts):
            for i in range(len(symptoms)):
                if diagsplit[j].lower() in symptomval[i].lower():
                    counter=counter+1
                    

        print(counter)
        stag=''
        pred=''

        naiveresult=summarize_by_class(Y)

        stag,pred=Process.predict(counter)
            
        
        print(stag)
        #for i in range(len()):

        #33333333       
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        

        df = pd.DataFrame({
            'x': X,
            'y': Y
        })

        
        np.random.seed(200)
        k = 3
        # centroids[i] = [x, y]
        centroids = {
            i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
            for i in range(k)
        }
            
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(df['x'], df['y'], color='k')
        colmap = {1: 'r', 2: 'g', 3: 'b'}
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i])
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()
        sql_Query = "insert into accdata values('"+patid+"','"+patname+"','"+age+"','"+date+"','"+doctor+"','"+gender+"','"+cs+"','"+cc+"','"+ah+"','"+stag+"','"+diag+"','"+diags+"','"+pred+"')"
        cursor = connection.cursor()   
        cursor.execute(sql_Query)
        connection.commit() 
        connection.close()
        cursor.close()
        
        knn=Process.Knnpr()
        svm=Process.svmpr()
        rf=Process.rfpr()
        #ComparisonGraph(knn,svm,rf)
        barWidth = 0.4
        bars1 = [int(knn),int(svm),int(rf)]
        bars4 = bars1 

        # The X position of bars
        r1 = [1,2,3]
        r4=r1

        # Create barplot
        plt.bar(r1, bars1, width = barWidth)
        # Note: the barplot could be created easily. See the barplot section for other examples.

        # Create legend
        plt.legend()

        # Text below each barplot with a rotation at 90°
        plt.xticks([r +0.6+ barWidth for r in range(len(r4))], ['KNN','SVM','RF'], rotation=0)
        # Create labels
        
        
        label = [str(knn)+' %',str(svm)+' %',str(rf)+' %']

        # Text on the top of each barplot
        for i in range(len(r4)):
            plt.text(x = r4[i]-0.2 , y = bars4[i]+0.4, s = label[i], size = 6)

        # Adjust the margins
        plt.subplots_adjust(bottom= 0.2, top = 0.98)

        # Show graphic    
        #plt.show(block=False)
        plt.show()
        #plt.pause(15)
        #plt.savefig('./static/assets/fctgrp6.png')
        #plt.close()
        #plt.close()
        
        
                     
        
        
        return render_template('forecast.html', msg=msg)
    
        

    
if __name__ == '__main__':
    UPLOAD_FOLDER = 'D:/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.debug = True
    app.run()
