import pandas as pd 
import numpy as np
from numpy import arange
import openpyxl


import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF


import matplotlib.pyplot as plt
import seaborn as sns




from feature_engine.transformation import ReciprocalTransformer
from sklearn.preprocessing import StandardScaler ,RobustScaler, QuantileTransformer ,PowerTransformer,MaxAbsScaler,MinMaxScaler,Normalizer
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,LarsCV,ridge_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from scipy.stats import zscore,iqr
from scipy.stats.mstats import winsorize


import streamlit as st






choice=st.text_input("Sheet's Name")

if choice:

    upload_file=st.file_uploader(" Upload your Excel File",type='xlsx')
    if upload_file is not None:
        df=pd.read_excel(upload_file,sheet_name=choice)
        

        


        wish=st.checkbox("**Want to add month column in the dataset**")
        if wish:
            df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
            df['Month']=df.Date.dt.month
            df['Year']=df.Date.dt.year
            df['Qarter']=df.Date.dt.quarter

            choise=st.checkbox("**Want to see the dataset**")
            if choise:

                st.write(df.shape,df)
                st.write(df.describe().T)
                st.write(df.corr())


        wish=st.checkbox("**Want to remove row from last**")
        if wish:
            number=st.text_input("Number of rows that you want to drop from last:",1)
            if number:
                
                df0=df.tail(1)
                df=df.head(-int(number))
                
                choise=st.checkbox("**Want to see the dataset** ")
                if choise:
                    st.write(df0.shape,df0)
                    st.write(df.shape,df)
        else:
            choise=st.checkbox("**Want to see the dataset** ")
            if choise:
                df0=df.tail(1)
                st.write(df0.shape,df0)
                st.write(df.shape,df)


        
        wish=st.checkbox("**Want to remove outliers from the dataset**")
        if wish:
            # Remove Outliers
            "### Remove outliers"
            choise=st.selectbox("",['None','Z-Score Method','IQR Method','Winsorize Method'])

            if choise=='None':
                pass                                         
            if choise=='Z-Score Method':
                df00=df
                # df00=df.drop('Date',axis=1)

                with st.sidebar:
                    col1,col2=st.columns(2)
                    with col2:
                        pass
                        col=st.multiselect("Select Columns that you want to remove outliers from",df00.columns)
                        col=list(col)

                    with col1:
                        num=st.number_input("Wants to keep rows less than zscore values",1)

                    z=np.abs(zscore(df00[col]))
                    df00=df00[(z<num).all(axis=1)]
                    # df00=df00[(z<num).all(axis=1)]
                    df=df00
                choise=st.checkbox("**Want to see the dataset**  ")
                if choise:
                    st.write(df.shape,df)
                    st.write(df.describe().T)
                
            if choise=='IQR Method':
                col=st.multiselect("Select Columns that you want to remove outliers from",df.columns)
                col=list(col)
                if col:
                    df00=df
                    # df00=df.drop('Date',axis=1)
                    #find Q1, Q3, and interquartile range for each column
                    Q1 = df00[col].quantile(q=.25)
                    Q3 = df00[col].quantile(q=.75)
                    IQR = df00[col].apply(iqr)

                    #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
                    df= df00[~((df00[col]< (Q1-1.5*IQR)) | (df00[col]> (Q3+1.5*IQR))).any(axis=1)]

                    choise=st.checkbox("Want to see the dataset  ")
                    if choise:
                        
                        
                        st.write(df.describe().T)
                
            if choise=='Winsorize Method':
                pass


            # list=['Standard Scaler','Mini-Max Scaler']
        # wish=st.sidebar(" Choose Dependent and Independent variables")
        # if wish:
        # "### Choose Dependent and Independent variables"
    # Dependent
        with st.sidebar:
            y=st.selectbox("**Dependent Variable**",df.columns)
            y=df[y]
            

            # Independent
            col=st.multiselect("**Independent Variable** ",df.columns)
            if col:
                
                x_test=df0[col]
                x=list(col)
                
                # df
                x=df[x]
                
                
                
                

            

        wish=st.checkbox("**Want to Standarize the Variables** ")
        if wish:
            "### Standarize Variables"
            choice=st.selectbox("Choose Standarization Method",['None','Standard Scaler','Normalizer','Robust Scaler','Quantile Transformer','Max Abs Scaler','MinMax Scaler','Box-Cox Transformer','Yeo-Johnson','Reciprocal Transformer','Sin Transformation','Cos transformation','Tan Transformation'])


            if choice=='None':
                wish=st.checkbox(" **Want to see the dataset**  ")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Standard Scaler':
                std_scaler=StandardScaler()
                x[col]=std_scaler.fit_transform(x)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                    
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Normalizer':
                std_scaler=Normalizer()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                


            if choice=='Robust Scaler':
                std_scaler=RobustScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                


            if choice=='Quantile Transformer':
                std_scaler=QuantileTransformer()
                x[col]=std_scaler.fit_transform(x.values)

                #method
                # x['Method']=choice
                # method=x['Method']
                # x=x.drop('Method',axis=1)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Max Abs Scaler':
                std_scaler=MaxAbsScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='MinMax Scaler':
                std_scaler=MinMaxScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='Box-Cox Transformer':
                std_scaler=PowerTransformer(method='box-cox')
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='Yeo-Johnson':
                std_scaler=PowerTransformer(method='yeo-johnson')
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Reciprocal Transformer':
                x[col]=1/x[col]
                x_test[col]=1/x_test[col]
                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Sin Transformation':
                x[col]=np.sin(x[col])
                x_test[col]=np.sin(x_test[col])
                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Cos Transformation':
                x[col]=np.cos(x[col])
                x_test[col]=np.cos(x_test[col])
                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Tan Transformation':
                x[col]=np.tan(x[col])
                x_test[col]=np.tan(x_test[col])
                wish=st.checkbox(" **Want to see the dataset**")
                if wish:
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
                
                # std_scaler=ReciprocalTransformer()
                # x[col]=std_scaler.fit_transform(x)
                # x_test[col]=std_scaler.transform(x_test)

                # wish=st.checkbox(" **Want to see the dataset**")
                # if wish:
                    
                #     st.write(x.shape,x)

                #     st.write(x.describe().T)

        choice1=st.checkbox("**Want to see the VIF values**")
        if choice1:

            #calculate VIF for each explanatory variable
            vif = pd.DataFrame()
            vif['variable'] = x.columns
            vif['VIF'] = [VIF(x.values, i) for i in range(x.shape[1])]
            
            #view VIF for each explanatory variable 
            st.write(vif.T)
        wish=st.checkbox("**Want to fit the model**")
        if wish:
            choise=st.selectbox("Please choose the model",['Multiple Linear Regression','ARIMA','ARIMAX','Ridge Regression','Lasso Regression','PLS','Elastic Net'])
            if choise=='Multiple Linear Regression':
                #  # Dependent 
                # y=st.selectbox("**Dependent Variable**",df1.columns)
                # y=df1[y]

                # # Independent
                # col=st.multiselect("**Independent Variable** ",df1.columns)
                # if col:
                #     x_test=df0[col]
                #     x=list(col)
                #     x=df1[x]

                #     choice=st.checkbox("**Want to see the VIF values**")
                #     if choice:

                #         #calculate VIF for each explanatory variable
                #         vif = pd.DataFrame()
                #         vif['variable'] = x.columns
                #         vif['VIF'] = [VIF(x.values, i) for i in range(x.shape[1])]
                        
                #         #view VIF for each explanatory variable 
                #         st.write(vif.T)

                    # Add constant
                    wish=st.checkbox("**Want to add constant**")
                    if wish:
                        x1=x
                        # x2=x_test

                        x=sm.add_constant(x)
                        x_test.insert(0,'const',1)
                        
                        choice=st.checkbox(" **Want to fit the model**")
                        if choice:
                            LR=LinearRegression()
                            model = sm.OLS(y,x).fit()
                            
                            model1=LR.fit(x,y)

                            # Model Residuals

                            wish2=st.checkbox("**Want to See the residual values**")
                            if wish2:
                                df1=pd.DataFrame()
                                df1['Actual value']=y.values
                                df1['Predicted value']=model.predict(x)
                                df1['Residuals']=model.resid
                                df1['Standard Deviation']=model.resid.std()
                                df1['Mean']=model.resid.mean()
                                df1['MAPE']=MAPE(y.values,model.predict(x))
                                st.write("**Residuals**",df1)

                                wish1=st.checkbox("**Want to download data**")
                                if wish1:
                                    text=st.text_input("File name")
                                    
                                    # 
                                    # with pd.ExcelWriter(text) as name:
                                    
                                    data1=df1.to_excel(text+'.xlsx',sheet_name='Residuals')

                            # Model Data        
                            wish0=st.checkbox("**Want to see the Model data**")
                            if wish0:
                                # data=pd.concat([x,x_test],axis=0)
                                x1=pd.concat([x,x_test],axis=0)
                                data=pd.merge(x1,y,how='left',left_index=True,right_index=True)
                                
                                st.write("**Data**",data)
                                
                                
                                wish1=st.checkbox(" **Want to download data**")
                                if wish1:
                                    text1=st.text_input("File name")
                                    if text1:
                                        button=st.button("Download")
                                        if button:
                                            data1=data.to_excel(text1+".xlsx",sheet_name='Data',index=False)
                                    

                                
                                    
                                    
                            #     st.write("Standard Data",x.shape,x)
                            #     st.write("Train Data")
                            #     st.write("Test Data",model)
                            
                        #     df
                        #     x
                            # df.set_index(x.set_index(keys=list(col)))
                            # df2=pd.join([df,x],axis=0,join='outer')
                            # st.dataframe(df2)
                            

                        
                            
                            
                            
                            # prediction    

                            wish=st.checkbox("**Want to see the model prediction**")
                                
                            if wish:
                                col1,col2,col3,col4,col5,col6,col7=st.columns(7)
                                with col1:
                                    st.write("**Test Data**",x_test)

                                with col2:
                                    st.write("**Prediction**",model.predict(x_test))
                                with col3:
                                    st.write("**MAPE**",MAPE(y,model.predict(x)))

                                with col4:
                                    st.write("**Std of Residual**",model.resid.std())

                                with col5:
                                    st.write("**Avg of Residual**",model.resid.mean())

                                with col6:
                                    
                                    pass
                                with col7:
                                    pass
                                    # st.write(model.predict(x))
                                    # Mape=(model.resid)/y
                                    # # Mape=Mape.mean()
                                    # st.write(Mape)
                            # wish=st.checkbox("**Want to see MAPE value**")
                            # if wish:
                            


                            choise=st.selectbox("**Want to see the model results**",["Summary","Linearity","Homoscedasticity","Residuals","Q-Q Plot","Residual Plot"])
                            if choise=="Summary":
                                st.write(model.summary())

                            if choise=="Linearity":
                                pass
                                # ind=st.selectbox("Choose an independent column",x.columns)
                                # ind=x[ind]
                                # ind
                
                                # # ind1=x[ind]
                                
                                
                                
                                # a=plt.scatter(ind,y)
                                
                                # # plt.show()
                                # plt.xlabel(str(ind))
                                # plt.ylabel(str(y))
                                # fig=plt.show()
                                # fig

                            if choise=="Homoscedasticity":
                                bp_test = het_breuschpagan(model.resid,x)
                                bp_value = bp_test[0]
                                bp_pvalue = bp_test[1]

                                

                                st.write("Breusch-Pagan test statistic: ", bp_value)
                                st.write("Breusch-Pagan test p-value: ", bp_pvalue)
                                
                                
                            if choise=="Residuals":
                                col1,col2=st.columns(2)
                                with col1:
                                    st.write("**Total MSE**",model.mse_total)
                                with col2:
                                    st.write(model.resid.T)

                            if choise=="Q-Q Plot":                           
                                #define residuals
                                res = model.resid
                                #create Q-Q plot
                                fig = sm.qqplot(res, fit=True, line="45")
                                fig
                                plt.show() 
                                
                            if choise=="Residual Plot":                           
                                ind=st.selectbox("Choose an independent column",x.columns)
                                ind
                                # dep=st.selectbox("Choose a dependent variable",df1.columns)
                                # modify figure size
                                fig = plt.figure(figsize=(14, 8))
                                
                                # creating regression plots
                                fig = sm.graphics.plot_regress_exog(model, ind, fig=fig)
                                fig
                                
                                

                    else:
                        
                        choice=st.checkbox(" **Want to fit the model**")
                        if choice:
                            model = sm.OLS(y,x).fit()
                            # pred=model.predict()
                            # prediction

                            # Model Residuals

                            wish2=st.checkbox("**Want to See the residual values**")
                            if wish2:
                                df1=pd.DataFrame()
                                df1['Actual value']=y.values
                                df1['Predicted value']=model.predict(x)
                                df1['Residuals']=model.resid
                                df1['Standard Deviation']=model.resid.std()
                                df1['Mean']=model.resid.mean()
                                df1['MAPE']=MAPE(y.values,model.predict(x))
                                st.write("**Residuals**",df1)

                                wish1=st.checkbox("**Want to download data**")
                                if wish1:
                                    text=st.text_input("File name")
                                    
                                    # 
                                    # with pd.ExcelWriter(text) as name:
                                    
                                    data1=df1.to_excel(text,sheet_name='Residuals')

                            # Model Data        
                            wish0=st.checkbox("**Want to see the Model data**")
                            if wish0:
                                data=pd.concat([x,x_test],axis=0)
                                data=data.merge(y,how='left',left_index=True,right_index=True)
                                # data.append(x_test,ignore_index=True)
                                
                                st.write("**Data**",data)
                                
                                
                                wish1=st.checkbox(" **Want to download data**")
                                if wish1:
                                    text1=st.text_input(" File name")
                                    button=st.button("Download")
                                    if button:
                                        data1=data.to_excel(text1+'.xlsx',sheet_name='Data')

                            wish=st.checkbox("**Want to see the model prediction**")
                                
                            if wish:
                                col1,col2,col3,col4,col5,col6,col7=st.columns(7)
                                with col1:
                                    st.write("**Test Data**",x_test)

                                with col2:
                                    
                                    st.write("**Prediction**",model.predict(x_test))
                                with col3:
                                    st.write("**MAPE**",MAPE(y,model.predict(x)))

                                with col4:
                                    st.write("**Std of Residual**",model.resid.std())

                                with col5:
                                    st.write("**Avg of Residual**",model.resid.mean())

                                with col6:
                                    
                                    pass
                                with col7:
                                    pass


                            choise=st.selectbox("**Want to see the model results**",["Summary","Linearity","Homoscedasticity","Residuals","Q-Q Plot","Residual Plot"])
                            if choise=="Summary":
                                st.write(model.summary())

                            if choise=="Linearity":
                                pass
                                # ind=st.selectbox("Choose an independent column",x.columns)
                                # str(ind)
                                # ind1=x[ind]
                                
                                
                                
                                # plt.scatter(ind1,y)
                                # plt.xlabel(str(ind))
                                # plt.ylabel(y)
                                # st.write(plt.show())

                            if choise=="Homoscedasticity":
                                bp_test = het_breuschpagan(model.resid,x)
                                bp_value = bp_test[0]
                                bp_pvalue = bp_test[1]

                                

                                st.write("Breusch-Pagan test statistic: ", bp_value)
                                st.write("Breusch-Pagan test p-value: ", bp_pvalue)
                            
                                
                            if choise=="Residuals":
                                col1,col2=st.columns(2)
                                with col1:
                                    st.write("**Total MSE**",model.mse_total)
                                with col2:
                                    st.write(model.resid.T)

                            if choise=="Q-Q Plot":                           
                                #define residuals
                                res = model.resid
                                #create Q-Q plot
                                fig = sm.qqplot(res, fit=True, line="45")
                                fig
                                plt.show() 
                                
                            if choise=="Residual Plot":                           
                                ind=st.selectbox("Choose an independent column",df.columns)
                                ind
                                # dep=st.selectbox("Choose a dependent variable",df1.columns)
                                # modify figure size
                                fig = plt.figure(figsize=(14, 8))
                                
                                # creating regression plots
                                fig = sm.graphics.plot_regress_exog(model, ind, fig=fig)
                                fig
                
            if choise=='Ridge Regression':
                pass
                #define cross-validation method to evaluate model
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

                #define model
                model = Ridge(alpha=1)
                

                #fit model
                model.fit(x, y)
                
                
                model.predict(x_test) 

                
            if choise=='ARIMA':
                model=ARIMA(x,y,order=(1,1,1)).fit()
                
                st.write(model.summary())
                wish=st.checkbox("**Want to see the model prediction**")
                                
                if wish:
                    col1,col2,col3,col4,col5=st.columns(5)
                    with col1:
                        st.write("**Test Data**",x_test)

                    with col2:
                        pass
                        
                        # st.write("**Prediction**",model.predict(x_test))
                    with col3:
                        st.write("**MAPE**",MAPE(y,model.predict(x)))

                    with col4:
                        st.write("**Std of Residual**",model.resid.std())

                    with col5:
                        st.write("**Avg of Residual**",model.resid.mean())
            if choise=='ARIMAX':

                # Find the best order and then fit model

                import itertools
                p = range(0, 3)
                d = range(0, 2)
                q = range(0, 3)
                pdq = list(itertools.product(p, d, q))

                P = range(0, 3)
                D = range(0, 1)
                Q = range(0, 3)
                seasonal_pdq = list(itertools.product(P, D, Q, [12]))


                import warnings


                best_model = None
                best_aic = float('inf')
                for params in pdq:
                    for seasonal_params in seasonal_pdq:
                        try:
                            model = SARIMAX(x, order=params, seasonal_order=seasonal_params, enforce_stationarity=False, enforce_invertibility=False)
                            results = model.fit()
                            aic = results.aic
                            if aic < best_aic:
                                best_model = model
                                best_aic = aic
                        except Exception as e:
                            continue

                        predictions = best_model.predict(start=len(x), end=len(x)+len(x)-1)













                # SARIMAX


                # result = adfuller(x)
                model=SARIMAX(y,x,order=(0,0,2),trend='ct').fit()
                # model=sm.tsa.SARIMAX(y,x,order=(1, 0, 0), trend='c')
                
                st.write(model.summary())
                wish=st.checkbox("**Want to see the model prediction**")
                                
                if wish:


                    # st.write("**Prediction**",model.predict(starts=len(x),end=len(x)+len(x_test)-1,exog=x_test))

                    
                    st.write("**Prediction**",model.forecast(steps=len(x_test),exog=x_test))

                    st.write("**MAPE**",MAPE(y,model.forecast(steps=len(x),exog=x)))


                    # col1,col2,col3,col4,col5=st.columns(5)
                    # with col1:
                    #     st.write("**Test Data**",x_test)

                    # with col2:
                    #     # pass
                        
                    #     st.write("**Prediction**",model.predict(x_test))
                    # with col3:
                    #     pass
                    #     # st.write("**MAPE**",MAPE(y,model.predict(x)))

                    # with col4:
                    #     pass
                    #     # st.write("**Std of Residual**",model.resid.std())

                    # with col5:
                    #     pass
                    #     # st.write("**Avg of Residual**",model.resid.mean())
                    
            if choise=='Lasso Regression':
                pass
            if choise=='PLS':
                pass
            if choise=='Elastic Net':
                pass

       
                
            


                
                
        


    
                    

                
                      

        



                    

else:
    st.warning("Please write your sheet's name above")







