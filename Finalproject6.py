import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import csv
import re

headers = {
    "accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "accept-language":"en-US,en;q=0.9",
    "cache-control":"max-age=0",
    "upgrade-insecure-requests":"1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36"
    }

baseurl = 'http://www.planecrashinfo.com/'
years = range(2012,2018)
valid_pages = range(1,100)
Accidents = {"Dates":[], "Times":[], "Location":[], "Operator":[], "FlightNo":[], "Route":[], "ACType":[], "Registration":[], "Aboard":[], "Fatalities":[], "Ground":[], "Summary":[]}
for n in years:
    for j in valid_pages:
        try:
            url = f'{baseurl}{n}/{n}-{j}'
            #print(url)
            resp = requests.get(url)
            if resp.status_code == 404 :
                break;
            soup = BeautifulSoup(resp.text, 'html.parser')
            AccidentData = list(map(lambda item:item.get_text(),soup.find_all("font")))
            Accidents["Dates"].append(AccidentData[3])
            Accidents["Times"].append(AccidentData[5])
            Accidents["Location"].append(AccidentData[7])
            Accidents["Operator"].append(AccidentData[9])
            Accidents["FlightNo"].append(AccidentData[11])
            Accidents["Route"].append(AccidentData[13])
            Accidents["ACType"].append(AccidentData[15])
            Accidents["Registration"].append(AccidentData[17])
            Accidents["Aboard"].append(AccidentData[19])
            Accidents["Fatalities"].append(AccidentData[21])
            Accidents["Ground"].append(AccidentData[23])
            Accidents["Summary"].append(AccidentData[27])
            #print(AccidentData)
        except IndexError:
            print("Error")
            break;

#print(Accidents)
AccidentTable = pd.DataFrame(Accidents)
AccidentTable['FatalitiesInt'] = AccidentTable['Fatalities'].str.split(' ').str[0]
AccidentTable['AboardInt'] = AccidentTable['Aboard'].str.split(' ').str[0]
AccidentTable['GroundInt'] = AccidentTable['Ground'].str.split(' ').str[0]
AccidentTable['Departure'] = AccidentTable['Route'].str.split(' ').str[0]
AccidentTable['Destination'] = AccidentTable['Route'].str.split(' ').str[1]
AccidentTable['ACType'] = AccidentTable['ACType'].str.split(' ').str[0]
AccidentTable['ACType'] = [1 if x=="Boeing" else x for x in AccidentTable['ACType']]
AccidentTable['ACType'] = [1 if x=="Airbus" else x for x in AccidentTable['ACType']]
AccidentTable['ACType'] = [0 if x!= 1 else x for x in AccidentTable['ACType']]
AccidentTable['Dates'] = pd.to_datetime(AccidentTable.Dates)
AccidentTable['Month']=''
AccidentTable['Month'] = pd.to_datetime(AccidentTable.Month)
AccidentTable['Month'] = AccidentTable['Dates'].dt.strftime('%m')
AccidentTable['Year']=''
AccidentTable['Year'] = pd.to_datetime(AccidentTable.Year)
AccidentTable['Year'] = AccidentTable['Dates'].dt.strftime('%Y')
AccidentTable['DeathRate']=''
AccidentTable['DeathRate'] = (pd.to_numeric(AccidentTable['GroundInt']))/(pd.to_numeric(AccidentTable['FatalitiesInt']))
AccidentTable['DeathRate'] = AccidentTable['DeathRate'].round(4)
AccidentTable['Jan']=''
AccidentTable['Feb']=''
AccidentTable['Mar']=''
AccidentTable['Apr']=''
AccidentTable['May']=''
AccidentTable['June']=''
AccidentTable['July']=''
AccidentTable['Aug']=''
AccidentTable['Sep']=''
AccidentTable['Oct']=''
AccidentTable['Nov']=''
AccidentTable['Dec']=''
AccidentTable['Jan'] = [1 if x=="01" else 0 for x in AccidentTable['Month']]
AccidentTable['Feb'] = [1 if x=="02" else 0 for x in AccidentTable['Month']]
AccidentTable['Mar'] = [1 if x=="03" else 0 for x in AccidentTable['Month']]
AccidentTable['Apr'] = [1 if x=="04" else 0 for x in AccidentTable['Month']]
AccidentTable['May'] = [1 if x=="05" else 0 for x in AccidentTable['Month']]
AccidentTable['June'] = [1 if x=="06" else 0 for x in AccidentTable['Month']]
AccidentTable['July'] = [1 if x=="07" else 0 for x in AccidentTable['Month']]
AccidentTable['Aug'] = [1 if x=="08" else 0 for x in AccidentTable['Month']]
AccidentTable['Sep'] = [1 if x=="09" else 0 for x in AccidentTable['Month']]
AccidentTable['Oct'] = [1 if x=="10" else 0 for x in AccidentTable['Month']]
AccidentTable['Nov'] = [1 if x=="11" else 0 for x in AccidentTable['Month']]
AccidentTable['Dec'] = [1 if x=="12" else 0 for x in AccidentTable['Month']]
AccidentTable['Weather']=''
AccidentTable['Weather']=AccidentTable['Summary'].str.contains("weather|Weather")
AccidentTable['Weather']=AccidentTable['Weather'].astype(int)
print(AccidentTable)

#Save data table to csv
AccidentTable.to_csv("output2012-2018.csv", index=False, sep="\t")

#Geocoding
from geopy.geocoders import Nominatim
import geopy
import certifi
import ssl
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
geolocator = Nominatim(user_agent="Finalproject3.py")
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=5)

AccidentTable['DepartureLocations'] = AccidentTable['Departure'].apply(geolocator.geocode)
AccidentTable['DestinationLocations'] = AccidentTable['Destination'].apply(geolocator.geocode)

tmp = AccidentTable.head().copy()
latlon = tmp.DepartureLocations.apply(lambda addr: geolocator.geocode(addr))
tmp["Latitude"] = [x.latitude for x in latlon]
tmp["Longitude"] = [x.longitude for x in latlon]
AccidentTable['DepartureLat']=tmp['Latitude']
AccidentTable['DepartureLon']=tmp['Longitude']

tmp = AccidentTable.head().copy()
latlon = tmp.DestinationLocations.apply(lambda addr: geolocator.geocode(addr))
tmp["Latitude"] = [x.latitude for x in latlon]
tmp["Longitude"] = [x.longitude for x in latlon]
AccidentTable['DestinationLat']=tmp['Latitude']
AccidentTable['DestinationLon']=tmp['Longitude']

#Plot flights on map using plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import webbrowser

plotly.tools.set_credentials_file(username='ValentinHilbig', api_key='W6xWqmBpwzwn44FfRxJr')

mapbox_access_token = 'pk.eyJ1Ijoic2h1bXdheTE5OTMiLCJhIjoiY2pwZmdoa3Y5MDMwNzNrcHU4dXRvdTdqYyJ9.WrVGBM9GsYgGKaiyW-KsSg'

data = [
    go.Scattermapbox(
        lat=AccidentTable["DepartureLat"],
        lon=AccidentTable["DepartureLon"],
        mode='markers',
        marker=dict(
            size=17,
            
        ),
        text=Accidents["Route"],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38.92,
            lon=-77.07
        ),
        pitch=0,
        zoom=0
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')

webbrowser.open('https://plot.ly/~ValentinHilbig/2.embed')

#Build 2 regression models
#Build regression Model for time series
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

#visualize the relationship between the features and the response using scatterplots
#sns.pairplot(AccidentTable, x_vars=['ACType','Weather','Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'], y_vars='DeathRate', height=7, aspect=0.8)
#plt.show() 

#Buile the model
x = AccidentTable[['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']]
y = AccidentTable['DeathRate']
#print(x.head(5))
#print(y.head(5))

#Setting training data
x_train,x_test, y_train, y_test = train_test_split(x, y, random_state=1)#default:25%testing,75%training
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#model
linreg = LinearRegression()
model=linreg.fit(x_train, y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)

# pair the feature names with the coefficients
feature_cols = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']
B=list(zip(feature_cols,linreg.coef_))
print(B)

# prediction
y_pred = linreg.predict(x_test)
print(y_pred)
print(type(y_pred))

# Model evaluation(Root Mean Squared Error, RMSE)
print(type(y_pred),type(y_test))
print(len(y_pred),len(y_test))
print(y_pred.shape,y_test.shape)

sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/39)

# calculate RMSE by hand
print("RMSE by hand:",sum_erro)


#####
#####
#Regression model for Weather & ACtype
a = AccidentTable[['Weather','ACType']]
b = AccidentTable['DeathRate']

#print(x.head(5))
#print(y.head(5))

#Setting training data
a_train,a_test, b_train, b_test = train_test_split(a, b, random_state=1)#default:25%testing,75%training
print(a_train.shape)
print(b_train.shape)
print(a_test.shape)
print(b_test.shape)

#model
linreg2 = LinearRegression()
model2=linreg2.fit(a_train, b_train)
print(model2)
print(linreg2.intercept_)
print(linreg2.coef_)

# pair the feature names with the coefficients
feature_cols2 = ['Weather','ACType']
C=list(zip(feature_cols2,linreg2.coef_))
print(C)

# prediction
b_pred = linreg2.predict(a_test)
print(b_pred)
print(type(b_pred))

# Model evaluation(Root Mean Squared Error, RMSE)
print(type(b_pred),type(b_test))
print(len(b_pred),len(b_test))
print(b_pred.shape,b_test.shape)

sum_mean=0
for i in range(len(b_pred)):
    sum_mean+=(b_pred[i]-b_test.values[i])**2
sum_erro=np.sqrt(sum_mean/39)

# calculate RMSE by hand
print("RMSE by hand:",sum_erro)

# ROC Line1
plt.figure(1)
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #show mark
plt.xlabel("Death Cause")
plt.ylabel('DeathRate')

# ROC Line2
plt.figure(2)
plt.plot(range(len(b_pred)),b_pred,'b',label="predict")
plt.plot(range(len(b_pred)),b_test,'r',label="test")
plt.legend(loc="upper right") #show mark
plt.xlabel("Death Cause2")
plt.ylabel('DeathRate')

#print out
plt.show()



