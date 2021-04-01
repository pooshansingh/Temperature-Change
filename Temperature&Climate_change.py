#importing required libraries
import pandas as pd 
import datetime
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,roc_curve,auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn import metrics

#Loading the Dataset
df = pd.read_csv("Temp_change.csv",encoding='latin-1')
#Cleaning
df.dropna()
df.head()
df = df.rename(columns = {"Area":"Country"})
df = df.drop(columns = ['Area Code',"Months Code","Element Code",'Unit'])

temperature_change = df.loc[df.Months.isin(['January',"February","March","April","May","June","July","August",'September','October',"November","December"])]

temperature_change = temperature_change.melt(id_vars=['Country','Months','Element'],var_name = 'Year',value_name ='temperature_change')
temperature_change['Year']= temperature_change['Year'].str[1:].astype('str')

gb = df.groupby("Country")

#Antarctica as a case study
Antarctica_df = gb.get_group("Antarctica")
Antarctica_df.replace(0,np.NaN).dropna(axis=1)

Antarctica_df.head()
plt.figure(figsize =(30,10))
plt.subplot(111)
sns.lineplot(x=Antarctica_df.Months.loc[Antarctica_df.Element=="Temperature change"],y=Antarctica_df.Y1970.loc[Antarctica_df.Element=='Temperature change'],label = "Y1970")
sns.lineplot(x=Antarctica_df.Months.loc[Antarctica_df.Element=="Temperature change"],y=Antarctica_df.Y1980.loc[Antarctica_df.Element=='Temperature change'],label = "Y1980")
sns.lineplot(x=Antarctica_df.Months.loc[Antarctica_df.Element=="Temperature change"],y=Antarctica_df.Y1990.loc[Antarctica_df.Element=='Temperature change'],label = "Y1990")
sns.lineplot(x=Antarctica_df.Months.loc[Antarctica_df.Element=="Temperature change"],y=Antarctica_df.Y2000.loc[Antarctica_df.Element=='Temperature change'],label = "Y2000")
sns.lineplot(x=Antarctica_df.Months.loc[Antarctica_df.Element=="Temperature change"],y=Antarctica_df.Y2010.loc[Antarctica_df.Element=='Temperature change'],label = "Y2010")
plt.xlabel("Months")
plt.ylabel("Temperature change ")
plt.title('Temperature change in Antarctica')
#plt.show()

Antarctica_df = Antarctica_df.melt(id_vars = ['Country','Months','Element'],var_name = 'Year',value_name ='temperature_change')
Antarctica_df['Year'] = Antarctica_df['Year'].str[1:].astype('str')
print(Antarctica_df.info())

plt.figure(figsize= (15,15))
plt.subplot(211)
for i in Antarctica_df.Year.unique():
	plt.plot(Antarctica_df.Months.loc[Antarctica_df.Year==str(i)].loc[Antarctica_df.Element=='Temperature change'],Antarctica_df.temperature_change.loc[Antarctica_df.Year ==str(i)].loc[Antarctica_df.Element=='Temperature change'],linewidth = 0.5)
plt.plot(Antarctica_df.Months.unique(),Antarctica_df.loc[Antarctica_df.Element=='Temperature change'].groupby(['Months']).mean(),'r',linewidth=2.0,label = 'Average')
plt.xlabel('Months')
plt.xticks(rotation = 45)
plt.ylabel('Temperature change')
plt.title('Temperature change in Antarctica')
plt.legend()

plt.subplot(212)
plt.plot(Antarctica_df.Months.loc[Antarctica_df.Year == '2010'].loc[Antarctica_df.Element=='Standard Deviation'],Antarctica_df.temperature_change.loc[Antarctica_df.Year=='2010'].loc[Antarctica_df.Element =='Standard Deviation'],linewidth=2.0)
plt.xlabel('Year')
plt.xticks(rotation = 45) 
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of temperature change in Antarctica (2010)')

plt.subplots_adjust(hspace = 0.3)
#plt.show()

plt.figure(figsize = (15,15))
plt.scatter(Antarctica_df['Year'].loc[Antarctica_df.Element =='Temperature change'],Antarctica_df['temperature_change'].loc[Antarctica_df.Element=='Temperature change'])
plt.plot(Antarctica_df.loc[Antarctica_df.Element == 'Temperature change'].groupby(['Year']).mean(),'r',label = "Average")
plt.axhline(y=0.0,color = 'k',linestyle = '-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation=45)
plt.ylabel('Temperature change')
plt.legend()
plt.title('Temperature change in Antarctica')
#plt.show()

plt.figure(figsize = (15,10))
sns.histplot(Antarctica_df.temperature_change.loc[Antarctica_df.Element=='Temperature change'],kde = True,stat = 'density')
plt.axvline(x=0.0,color='b',linestyle = '-')
plt.xlabel('Temperature change')
plt.title('Temperature change in Antarctica')
#plt.show()


regions=temperature_change[temperature_change.Country.isin(['World', 'Africa',
       'Eastern Africa', 'Middle Africa', 'Northern Africa',
       'Southern Africa', 'Western Africa', 'Americas',
       'Northern America', 'Central America', 'Caribbean',
       'South America', 'Asia', 'Central Asia', 'Eastern Asia',
       'Southern Asia', 'South-Eastern Asia', 'Western Asia', 'Europe',
       'Eastern Europe', 'Northern Europe', 'Southern Europe',
       'Western Europe', 'Oceania', 'Australia and New Zealand',
       'Melanesia', 'Micronesia', 'Polynesia', 'European Union',
       'Least Developed Countries', 'Land Locked Developing Countries',
       'Small Island Developing States',
       'Low Income Food Deficit Countries',
       'Net Food Importing Developing Countries', 'Annex I countries',
       'Non-Annex I countries', 'OECD'])]

temperature_change = temperature_change[~temperature_change.Country.isin(['World', 'Africa',
       'Eastern Africa', 'Middle Africa', 'Northern Africa',
       'Southern Africa', 'Western Africa', 'Americas',
       'Northern America', 'Central America', 'Caribbean',
       'South America', 'Asia', 'Central Asia', 'Eastern Asia',
       'Southern Asia', 'South-Eastern Asia', 'Western Asia', 'Europe',
       'Eastern Europe', 'Northern Europe', 'Southern Europe',
       'Western Europe', 'Oceania', 'Australia and New Zealand',
       'Melanesia', 'Micronesia', 'Polynesia', 'European Union',
       'Least Developed Countries', 'Land Locked Developing Countries',
       'Small Island Developing States',
       'Low Income Food Deficit Countries',
       'Net Food Importing Developing Countries', 'Annex I countries',
       'Non-Annex I countries', 'OECD'])]

plt.figure(figsize = (15,15))
sns.histplot(temperature_change.temperature_change.loc[temperature_change.Element =='Temperature change'],kde = True,stat = 'density')
plt.axvline(x = 0.0,color = 'b',linestyle = '-')
plt.xlabel("Temperature change")
plt.title('Temperature change distribution in the world')
plt.xlim(-5,5)
#plt.show()

average_temp = temperature_change.loc[temperature_change.Element == 'Temperature change'].groupby(['Year'],as_index = False).mean()
avg_temp_country = temperature_change.loc[temperature_change.Element=='Temperature change'].groupby(['Country','Year'],as_index = False).mean()

plt.figure(figsize=(15,10))
plt.scatter(temperature_change["Year"].loc[temperature_change.Element=='Temperature change'],temperature_change['temperature_change'].loc[temperature_change.Element=='Temperature change'])
plt.plot(average_temp.Year,average_temp.temperature_change,'r',label = 'Average')
plt.axhline(y=0.0,color = 'k',linestyle = '-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation =45)
plt.ylabel('Temperature change')
plt.legend()
plt.title('Temperature change across the world')
#plt.show()

plt.figure(figsize = (15,10))
for i in avg_temp_country.Country.unique():
	plt.plot(avg_temp_country.Year.loc[avg_temp_country.Country == str(i)],avg_temp_country.temperature_change.loc[avg_temp_country.Country == str(i)],linewidth = 0.5 )
plt.plot(average_temp.Year,average_temp.temperature_change,'r',linewidth = 2.0)
plt.axhline(y = 0.0,color = 'b',linestyle ='-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation =45)
plt.ylabel('Average temperature change')
plt.title('Average temperature change of the world')
#plt.show()

Month_v = {'January':'1', 'February':'2', 'March':'3', 'April':'4', 'May':'5', 'June':'6', 'July':'7','August':'8', 'September':'9', 'October':'10', 'November':'11', 'December':'12'}
temperature_change = temperature_change.replace(Month_v)

Austria_df = gb.get_group('Austria')
Austria_df = Astria_df.melt(id_vars = ['Country','Months','Element'],var_name = 'Year',value_name ='temperature_change')
Austria_df['Year'] = Astria_df['Year'].str[1:].astype(np.float64)

df.drop(Astria_df.index,axis=0)

y= temperature_change['temperature_change'].loc[temperature_change.Element=='Temperature change']

X = temperature_change.drop(columns =['temperature_change','Country','Months','Element']).loc[temperature_change.Element=="Temperature change"]

##$$$$$$##
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size = 0.8,random_state=42)

LR = LinearRegression()
LR.fit(X_train,y_train)
LRpreds = LR.predict(X_valid)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, LRpreds)))
plt.figure(figsize = (15,8))
plt.plot(y_valid-LRpreds,'o')
plt.axhline(y=0.0,color='k',linestyle = '-' )
plt.ylabel('Actual Value - Predicted value')

LR.fit(X,y)
LR_test = pd.DataFrame({'Year':np.random.randint(1980,2060,size =1000)})
LR.test = LR_test.sort_values(by=['Year']).reset_index(drop = True).astype(str)

preds_test = LR.predict(LR_test)
LR_test['temperature_change'] = pd.Series(preds_test,index = LR_test.index)

PR2_mod = Pipeline([('poly',PolynomialFeatures(degree=2)),('linear',LinearRegression(fit_intercept = False))])

PR3_mod = Pipeline([('poly',PolynomialFeatures(degree=5)),('linear',LinearRegression(fit_intercept = False))])

PR2_mod.fit(X,y)
PR3_mod.fit(X,y)

PR2_test = pd.DataFrame({'Year':np.random.randint(1980,2060,size =1000)})
PR2_test = PR2_test.sort_values(by=['Year']).reset_index(drop=True).astype(str)

PR3_test = pd.DataFrame({'Year':np.random.randint(1980,2060,size = 1000)})
PR3_test = PR3_test.sort_values(by = ['Year']).reset_index(drop = True).astype(str)

pred2_test = PR2_mod.predict(PR2_test)
pred3_test = PR3_mod.predict(PR3_test)

PR2_test['temperature_change'] = pd.Series(pred2_test,index = PR2_test.index)

PR3_test['temperature_change'] = pd.Series(pred3_test,index = PR2_test.index)

plt.figure(figsize = (15,10))
for i in avg_temp_country.Country.unique():
	plt.plot(avg_temp_country.Year.loc[avg_temp_country.Country == str(i)],avg_temp_country.temperature_change.loc[avg_temp_country.Country==str(i)],linewidth = 0.5)
plt.plot(average_temp.Year,average_temp.temperature_change,'r',linewidth = 2.0)
plt.plot(LR_test.Year.unique(),LR_test.groupby('Year').mean(),'b',linewidth = 2.0,label = 'Linear Model')
plt.plot(PR2_test.Year.unique(),PR2_test.groupby('Year').mean(),'g',linewidth=2.0,label = 'Poly-2 Model')
plt.plot(PR3_test.Year.unique(),PR3_test.groupby('Year').mean(),'c',linewidth=2.0,label = 'Poly-5 Model')
plt.axhline(y=0.0,color = 'k',linestyle = '-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,100,40),rotation=45)

plt.ylabel('Avg temperatue change')
plt.title('Avg temperature change across the world')
plt.legend()
#