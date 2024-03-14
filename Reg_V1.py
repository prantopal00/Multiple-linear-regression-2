import pandas as pd 
import numpy as np
from numpy import arange
import openpyxl
from io import BytesIO


import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.tools.eval_measures import rmse


import matplotlib.pyplot as plt
import seaborn as sns




from feature_engine.transformation import ReciprocalTransformer
from sklearn.preprocessing import StandardScaler ,RobustScaler, QuantileTransformer ,PowerTransformer,MaxAbsScaler,MinMaxScaler,Normalizer
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,LarsCV,ridge_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_percentage_error as MAPE,mean_squared_error

from scipy.stats import zscore,iqr
from scipy.stats.mstats import winsorize


import streamlit as st






choice=st.text_input("Sheet's Name")

if choice:

    upload_file=st.file_uploader(" Upload your Excel File",type='xlsx')
    if upload_file is not None:
        df=pd.read_excel(upload_file,sheet_name=choice)

        with st.expander("**Data With Summary Table**"):
                selectbox=st.selectbox("",["Data","Summary Table"])
                
                if selectbox=="Data":
                    st.write(df.shape,df)
                if selectbox=="Summary Table":
                    dfx=df
                    dfx=dfx.drop(df.columns[0],axis=1)
                    st.write(dfx.describe().T)
        
                    
        # if st.checkbox("Want to delete Particular Rows"):


        wish=st.checkbox("**Want to add month column in the dataset**")
        if wish:
            df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
            df['Month']=df.Date.dt.month
            df['Year']=df.Date.dt.year
            df['Qarter']=df.Date.dt.quarter

            # choise=st.checkbox("**Want to see the dataset**")
            # if choise:
            with st.expander("**Data With Summary Table**"):
                selectbox=st.selectbox("",["Data","Summary Table"],key=2)
                
                if selectbox=="Data":
                    st.write(df.shape,df)
                if selectbox=="Summary Table":
                    st.write(df.describe().T)                

        "### »»»Remove outliers"
        wish=st.checkbox("**Want to remove outliers from the dataset**")
        if wish:
            # Remove Outliers
            
            choise=st.selectbox("**Select One Method**",['None','Mean_Standard Deviation and Percentile Method','IQR Method','Winsorize Method'])

            if choise=='None':
                pass                                         
            if choise=='Mean_Standard Deviation and Percentile Method':
                df00=df
                df00=df.drop('Date',axis=1)

                

                # with st.sidebar:
                col1,col2,col3=st.columns(3)

                with col2:
                    percentile=st.number_input("**Percentile**",value=0.95,step=0.01,min_value=0.00,max_value=1.00)
                with col3:
                    
                    col=st.multiselect("**Select Columns that you want to remove outliers from**",df00.columns)
                    col=list(col)


                    
                    

                    # with st.expander("**Box Plot(Before Outlier Treatment**)"):
                    #     plt=df00[col].boxplot()
                    #     plt.plot()
                        # plt.figure(figsize=(8, 6))
                        # plt.boxplot(df00[col])
                        # plt.title(f'Box Plot of {col}')
                        # plt.ylabel('Values')
                        # plt.show()

                with col1:
                    num=st.number_input("**zscore values**",min_value=1,step=1,value=2)

                # with st.expander("**Box Plot(Before Outlier Treatment**)"):
                #     plt.figure(figsize=(8, 6))
                #     plt.boxplot(df00[col])
                #     plt.title(f'Box Plot of {col}')
                #     plt.ylabel('Values')
                #     plt.show()

                

                

                # z=np.abs(zscore(df00[col]))
                # z
                
                

                upper_limit0 = df00[col].mean() + num*df00[col].std()
                lower_limit0 = df00[col].mean() - num*df00[col].std()

                Q1 = df00[col].quantile(q=round((1-percentile),2))
                Q4 = df00[col].quantile(q=percentile)

                upper_limit=min(float(upper_limit0),float(Q4))
                lower_limit=max(float(lower_limit0),float(Q1))

                # if lower_limit != 0:
                #     lower_limit=lower_limit
                # else:

                
                

                # upper_limit=min(num(upper_limit0),num(Q4))
                # lower_limit=max(num(lower_limit0),num(Q1))
                wish=st.selectbox("**Select One Method to replace Outliers**",["Mean Standard Deviation Method","Percentile Method"])

                if wish=="Mean Standard Deviation Method":

                    if st.checkbox("**Concider Mean and Standard Deviation as a Upper and Lower Limit**"):

                        lower_outlier=df00[df00[col]< lower_limit0]
                        upper_outlier=df00[df00[col]> upper_limit0]


                        loc_lower_outlier = lower_outlier[col][lower_outlier[col].notna()]
                        loc_upper_outlier = upper_outlier[col][upper_outlier[col].notna()]

                        # a=upper_outlier[upper_outlier[col].isna()].index.to_list()
                        # a
                        with st.expander("**Lower Limit and Location of Outliers**"):
                            col1,col2=st.columns(2)
                            with col1:
                                st.write("**Lower Limit :**",lower_limit0,"**Lower Quantile**",Q1,"**Final Lower Limit**",lower_limit)
                                st.write("**Number of Lower Outliers :**",loc_lower_outlier.notna().sum())
                                st.write("**Lower Outlier Location**",loc_lower_outlier)
                            with col2:
                                st.write("**Upper Limit :**",upper_limit0,"**Upper Quantile**",Q4,"**Final Upper Limit**",upper_limit)
                                # Q4
                                st.write("**Number of Upper Outliers :**",loc_upper_outlier.notna().sum())
                                st.write("**Upper Outlier Location**",loc_upper_outlier)
                        
                        

                        if st.checkbox("**Want to replace Outlier**"):
                            df00[col] = np.where(df00[col]> upper_limit0, upper_limit,
                            np.where(df00[col]< lower_limit0, lower_limit,
                                df00[col]))
                            
                            df=df00

                if wish=="Percentile Method":

                    if st.checkbox("**Consider Percentile as a Lower and Upper Limit**"):
                        lower_limit0=Q1
                        upper_limit0=Q4

                        upper_limit=Q4
                        lower_limit=Q1



                        lower_outlier=df00[df00[col]< lower_limit0]
                        upper_outlier=df00[df00[col]> upper_limit0]


                        loc_lower_outlier = lower_outlier[col][lower_outlier[col].notna()]
                        loc_upper_outlier = upper_outlier[col][upper_outlier[col].notna()]

                        # a=upper_outlier[upper_outlier[col].isna()].index.to_list()
                        # a
                        with st.expander("**Lower Limit and Location of Outliers**"):
                            col1,col2=st.columns(2)
                            with col1:
                                st.write("**Lower Limit :**",lower_limit0,"**Lower Quantile**",Q1,"**Final Lower Limit**",lower_limit)
                                st.write("**Number of Lower Outliers :**",loc_lower_outlier.notna().sum())
                                st.write("**Lower Outlier Location**",loc_lower_outlier)
                            with col2:
                                st.write("**Upper Limit :**",upper_limit0,"**Upper Quantile**",Q4,"**Final Upper Limit**",upper_limit)
                                # Q4
                                st.write("**Number of Upper Outliers :**",loc_upper_outlier.notna().sum())
                                st.write("**Upper Outlier Location**",loc_upper_outlier)
                        
                        

                        if st.checkbox("**Want to replace Outlier**"):
                            df00[col] = np.where(df00[col]> upper_limit0, upper_limit,
                            np.where(df00[col]< lower_limit0, lower_limit,
                                df00[col]))
                            
                            df=df00

                
                else:
                #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
                    df= df00[~((df00[col]< lower_limit0) | (df00[col]> upper_limit0)).any(axis=1)]




                ##only keep rows in dataframe with all z-scores less than absolute value of 3

                # df00=df00[(z<num).all(axis=1)]
                # # df00=df00[(z<num).all(axis=1)]
                # df=df00
                with st.expander("**Data With Summary Table**"):
                        selectbox=st.selectbox("",["Data","Summary Table"],key=33)
                        
                        if selectbox=="Data":
                            st.write(df.shape,df)
                        if selectbox=="Summary Table":
                            st.write(df.describe().T)

                if st.checkbox("**Want to Download**  "):
                    filename=st.text_input("**Write File Name**")
                    if filename:
                        filename=filename+".csv"

                        st.download_button(
                            label=f"**Download Data as a CSV File**",
                            data=df.to_csv(),
                            file_name=filename,
                            mime='csv',
                        )
                

            if choise=='IQR Method':
                col=st.multiselect("**Select Columns that you want to remove outliers from**",df.columns)
                col=list(col)
                if col:
                    df00=df
                    df00=df.drop('Date',axis=1)
                    #find Q1, Q3, and interquartile range for each column
                    Q1 = df00[col].quantile(q=0.25)
                    Q3 = df00[col].quantile(q=0.75)
                    # IQR = df00[col].apply(iqr)
                    IQR=Q3-Q1

                    lower_limit=Q1-1.5*IQR
                    upper_limit=Q3+1.5*IQR

                    
                    lower_outlier=df00[df00[col]< lower_limit]
                    upper_outlier=df00[df00[col]> upper_limit]


                    loc_lower_outlier = lower_outlier[col][lower_outlier[col].notna()]
                    loc_upper_outlier = upper_outlier[col][upper_outlier[col].notna()]

                    # a=upper_outlier[upper_outlier[col].isna()].index.to_list()
                    # a
                    with st.expander("**Lower Limit and Location of Outliers**"):
                        col1,col2=st.columns(2)
                        with col1:
                            st.write("**Lower Limit :**",lower_limit)
                            st.write("**Number of Lower Outliers :**",loc_lower_outlier.notna().sum())
                            st.write("**Lower Outlier Location**",loc_lower_outlier)
                        with col2:
                            st.write("**Upper Limit :**",upper_limit)
                            st.write("**Number of Upper Outliers :**",loc_upper_outlier.notna().sum())
                            st.write("**Upper Outlier Location**",loc_upper_outlier)
                    
                    

                    if st.checkbox("**Want to replace Outlier**"):
                        df00[col] = np.where(df00[col]> upper_limit, upper_limit,
                        np.where(df00[col]< lower_limit, lower_limit,
                          df00[col]))
                        

                        
                        df=df00
                    
                    else:
                    #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
                        df= df00[~((df00[col]< (Q1-1.5*IQR)) | (df00[col]> (Q3+1.5*IQR))).any(axis=1)]

                    with st.expander("**Data With Summary Table**"):
                        selectbox=st.selectbox("",["Data","Summary Table"],key=1)
                        
                        if selectbox=="Data":
                            st.write(df.shape,df)
                        if selectbox=="Summary Table":
                            st.write(df.describe().T)

                    if st.checkbox("**Want to Download**  "):
                        filename=st.text_input("**Write File Name**")
                        if filename:
                            filename=filename+".csv"

                            st.download_button(
                                label=f"**Download Data as a CSV File**",
                                data=df.to_csv(),
                                file_name=filename,
                                mime='csv',
                            )
                
            if choise=='Winsorize Method':
                pass

        "### »»» Log Transformation and Add Lag of Dependent Variable"

        if st.checkbox("**Want to add Log Transformation and Lag Columns**"):

            col1,col2,col3=st.columns(3)

            with col1:
                lag=st.number_input("**Lag Value**",1)

            with col2:
                col=st.selectbox("**Select a Column that you want to do Log Transformation**",df.columns)
                if col:
                    df[col+"_Log(e)"]=np.log(df[col])
                    df[col+"_Log(10)"]=np.log10(df[col])
                    df[col+"_Log(2)"]=np.log2(df[col])


            with col3:

                col=st.selectbox("**Select a Column that you want to add Lag**",df.columns)
                # col=list(col)
                if col:
                    
                    df[col+"_Lag"]=df[col].shift(periods=lag)



                    # for col in col:
                    #     df[col+"_Lag"]=df[col].shift(periods=lag)

                        # Log transformation
                        # df[col+"_Log(e)"]=np.log(df[col])
                        # df[col+"_Log(10)"]=np.log10(df[col])
                        # df[col+"_Log(2)"]=np.log2(df[col])

                    # Remove first lag number of rows

                    df=df.iloc[lag:]
        df=df

        


        with st.expander("**Data With Summary Table**"):
            selectbox=st.selectbox("",["Data","Summary Table"],key=12)
            
            if selectbox=="Data":
                st.write(df.shape,df)
            if selectbox=="Summary Table":
                st.write(df.describe().T)

        

        if st.checkbox("**Want to Download**   "):
                filename=st.text_input("**Write File Name** ")
                if filename:
                    filename=filename+".csv"

                    st.download_button(
                        label=f"**Download Data as a CSV File** ",
                        data=df.to_csv(),
                        file_name=filename,
                        mime='csv',
                    )

        "#### »»» Remove Rows from last"
        wish=st.checkbox("**Want to remove row from last**")
        if wish:
            number=st.text_input("Number of rows that you want to drop from last:",1)
            if number:
                
                df0=df.tail(int(number))
                df=df.head(-int(number))

                #Correlation

                with st.expander("**Correlation Matrix**"):
                    df_corr=df.corr()
                    df_corr
                
                # choise=st.checkbox("**Want to see the dataset** ")
                # if choise:
                with st.expander("**Data**"):
                    st.write("**Removed Rows Data**",df0.shape,df0)
                    st.write("**Rest Data**",df.shape,df)

                if st.checkbox("**Want to Download**"):
                    filename=st.text_input("**Write File Name**")
                    if filename:
                        filename=filename+".csv"

                        st.download_button(
                            label=f"**Download Data as a CSV File**",
                            data=df.to_csv(),
                            file_name=filename,
                            mime='csv',
                        )
        else:

            with st.expander("**Correlation Matrix**"):
                df_corr=df.corr()
                df_corr
            with st.expander("**Data**"):
                df0=df.tail(1)
                df=df.head(-int(1))
                st.write("**Removed Rows Data**",df0.shape,df0)
                st.write("**Rest Data**",df.shape,df)

            if st.checkbox("**Want to Download**"):
                    filename=st.text_input("**Write File Name**")
                    if filename:
                        filename=filename+".csv"

                        st.download_button(
                            label=f"**Download Data as a CSV File**",
                            data=df.to_csv(),
                            file_name=filename,
                            mime='csv',
                        )



            


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
                
                
                
                

            
        "### »»» Standarize Variables"

        wish=st.checkbox("**Want to Standarize the Variables** ")
        if wish:
            col=st.multiselect("**Select Variables** ",col,default=col)
            
            choice=st.selectbox("Choose Standarization Method",['None','Standard Scaler','Normalizer','Robust Scaler','Quantile Transformer','Max Abs Scaler','MinMax Scaler','Box-Cox Transformer','Yeo-Johnson','Reciprocal Transformer','Sin Transformation','Cos transformation','Tan Transformation','Log Transformation'])


            if choice=='None':
                with st.expander("Without Standarized Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Log Transformation':
                x[col]=np.log(x[col])
                x_test[col]=np.log(x_test[col])

                with st.expander("Log Transformation Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='Standard Scaler':
                std_scaler=StandardScaler()
                x[col]=std_scaler.fit_transform(x)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Standard Scalar Data"):
                    
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Normalizer':
                std_scaler=Normalizer()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Normalized Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                


            if choice=='Robust Scaler':
                std_scaler=RobustScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Robust Scalar Data"):
                
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

                with st.expander("Quantile Transformer Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Max Abs Scaler':
                std_scaler=MaxAbsScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Max Abs Scaler Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='MinMax Scaler':
                std_scaler=MinMaxScaler()
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Minimax Scaler Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='Box-Cox Transformer':
                std_scaler=PowerTransformer(method='box-cox')
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Box-Cox Transformer Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)


            if choice=='Yeo-Johnson':
                std_scaler=PowerTransformer(method='yeo-johnson')
                x[col]=std_scaler.fit_transform(x.values)
                x_test[col]=std_scaler.transform(x_test)

                with st.expander("Yeo-Johnson Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)

            if choice=='Reciprocal Transformer':
                x[col]=1/x[col]
                x_test[col]=1/x_test[col]
                with st.expander("Reciprocal Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Sin Transformation':
                x[col]=np.sin(x[col])
                x_test[col]=np.sin(x_test[col])

                with st.expander("Sine Transformation Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Cos transformation':
                x[col]=np.cos(x[col])
                x_test[col]=np.cos(x_test[col])

                with st.expander("Cox Transformation Data"):
                
                    st.write(x.shape,x)

                    st.write(x.describe().T)
                # pass
            if choice=='Tan Transformation':
                x[col]=np.tan(x[col])
                x_test[col]=np.tan(x_test[col])
                with st.expander("Tan Transformation Data"):
                
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

        "### »»» VIF Values"
        choice1=st.checkbox("**Want to see the VIF values**")
        if choice1:

            #calculate VIF for each explanatory variable
            vif = pd.DataFrame()
            vif['variable'] = x.columns
            vif['VIF'] = [VIF(x.values, i) for i in range(x.shape[1])]
            
            #view VIF for each explanatory variable 
            st.write(vif.T)

            if st.checkbox("**Want to Download VIF Data**  "):
                        filename=st.text_input("**Write File Name** ")
                        if filename:
                            filename=filename+".csv"

                            st.download_button(
                                label=f"**Download Data as a CSV File** ",
                                data=vif.to_csv(),
                                file_name=filename,
                                mime='csv',
                            )

        st.subheader("»»» Model Fit")


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
                        
                        # choice=st.checkbox(" **Want to fit the model**")
                        # if choice:
                        LR=LinearRegression()
                        model = sm.OLS(y,x).fit()
                        
                        model1=LR.fit(x,y)

                        "### »»» Model Residuals"

                        wish2=st.checkbox("**Want to See the residual values**")
                        if wish2:
                            df1=pd.DataFrame()
                            df1['Actual value']=y.values
                            df1['Predicted value']=model.predict(x)
                            df1['Residuals']=model.resid
                            df1['Standard Deviation']=model.resid.std()
                            df1['Mean']=model.resid.mean()
                            df1['MAPE']=MAPE(y.values,model.predict(x))
                            df1['Total MSE']=model.mse_total

                            with st.expander("Residuals Data"):
                                st.write("**Residuals**",df1)
                            # st.write("**Residuals**",df1)

                            if st.checkbox("**Want to Download Residual Data**  "):
                                filename=st.text_input("**Write File Name**  ")
                                if filename:
                                    filename=filename+".csv"

                                    st.download_button(
                                        label=f"**Download Data as a CSV File**  ",
                                        data=df1.to_csv(),
                                        file_name=filename,
                                        mime='csv',
                                    )

                        "### »»» Model Data"        
                        wish0=st.checkbox("**Want to see the Model data**")
                        if wish0:
                            # data=pd.concat([x,x_test],axis=0)
                            x1=pd.concat([x,x_test],axis=0)
                            data=pd.merge(x1,y,how='left',left_index=True,right_index=True)
                            
                            with st.expander("Model Data"):
                                st.write("**Data**",data)
                            
                            
                            if st.checkbox("**Want to Download Model Data**  "):
                                filename=st.text_input("**Write File Name**   ")
                                if filename:
                                    filename=filename+".csv"

                                    st.download_button(
                                        label=f"**Download Data as a CSV File**   ",
                                        data=data.to_csv(),
                                        file_name=filename,
                                        mime='csv',
                                    )

                            
                                
                                
                        #     st.write("Standard Data",x.shape,x)
                        #     st.write("Train Data")
                        #     st.write("Test Data",model)
                        
                    #     df
                    #     x
                        # df.set_index(x.set_index(keys=list(col)))
                        # df2=pd.join([df,x],axis=0,join='outer')
                        # st.dataframe(df2)
                        

                    
                        
                        
                        
                        "### »»» Prediction Data"    

                        wish=st.checkbox("**Want to see the model prediction**")
                            
                        if wish:
                            # col1,col2,col3,col4=st.columns(4)
                            # with col1:
                            #     st.write("**Test Data**",x_test)

                            # with col2:
                            #     st.write("**Prediction**",model.predict(x_test))

                            # Get predictions and confidence intervals
                            predictions = model.predict(x_test)

                            # confidence interval
                            ci = model.get_prediction(x_test).conf_int()
                    

                            # # Calculate lower and upper estimates
                            lower_estimates = predictions - (ci[:, 1] - predictions)
                            upper_estimates = predictions + (ci[:, 1] - predictions)

                            # estimate dataframe

                            estimate=pd.DataFrame({"Lower Estimates":lower_estimates,"Estimates":predictions,"Upper Estimates":upper_estimates})

                        
                            prediction_data=pd.merge(x_test,estimate,left_index=True,right_index=True)
                            
                            with st.expander("**Predicted Data**"):
                                prediction_data

                                

                                # st.write("Lower Estimate",lower_estimates)
                                # st.write("Upper Estimate",upper_estimates)


                            # with st.sidebar:
                            #     st.write("**MAPE**",MAPE(y,model.predict(x)))

                            # with st.sidebar:
                            #     st.write("**Std of Residual**",model.resid.std())

                            # with st.sidebar:
                            #     st.write("**Avg of Residual**",model.resid.mean())
                            #     st.write("**Total MSE**",model.mse_total)
                                # st.write("MSE",mean_squared_error(x_test,model.predict(x_test)))


                                # st.write(model.predict(x))
                                # Mape=(model.resid)/y
                                # # Mape=Mape.mean()
                                # st.write(Mape)
                        # wish=st.checkbox("**Want to see MAPE value**")
                        # if wish:
                            
                            if st.checkbox("**Want to Download Prediction Data**   "):
                                    filename=st.text_input("**Write File Name**     ")
                                    if filename:
                                        filename=filename+".csv"

                                        st.download_button(
                                            label=f"**Download Data as a CSV File**     ",
                                            data=prediction_data.to_csv(),
                                            file_name=filename,
                                            mime='csv',
                                        )
                        

                        "### »»» Model Result"

                        choise=st.selectbox("**Want to see the model results**",["Summary","Linearity","Homoscedasticity","Residuals","Q-Q Plot","Residual Plot"])
                        if choise=="Summary":
                            st.write(model.summary())

                            with st.sidebar:
                                st.write("**MAPE**",MAPE(y,model.predict(x)))

                            with st.sidebar:
                                st.write("**Std of Residual**",model.resid.std())

                            with st.sidebar:
                                st.write("**Avg of Residual**",model.resid.mean())
                                st.write("**Total MSE**",model.mse_total)

                                # Breush Pegan
                                bp_test = het_breuschpagan(model.resid,x)
                                bp_value = bp_test[0]
                                bp_pvalue = bp_test[1]

                                # RMSE

                                # rmse1=rmse(x_test,model.predict(x_test))
                                # st.write("",rmse1)

                            

                                st.write("Breusch-Pagan test statistic: ", bp_value)
                                st.write("Breusch-Pagan test p-value: ", bp_pvalue)

                                # # Perform Breusch-Pagan test
                                # lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(model.resid,x)

                                # st.write("Breusch-Pagan Test:")
                                # st.write("LM Test Statistic:", lm)
                                # st.write("LM Test p-value:", lm_p_value)
                                # st.write("F Test Statistic:", fvalue)
                                # st.write("F Test p-value:", f_p_value)


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
                        
                        "### »»» Model Fit"
                        # choice=st.checkbox(" **Want to fit the model**")
                        # if choice:
                        model = sm.OLS(y,x).fit()
                        # pred=model.predict()
                        # prediction

                        "### »»» Model Residuals"

                        wish2=st.checkbox("**Want to See the residual values**")
                        if wish2:
                            df1=pd.DataFrame()
                            df1['Actual value']=y.values
                            df1['Predicted value']=model.predict(x)
                            df1['Residuals']=model.resid
                            df1['Standard Deviation']=model.resid.std()
                            df1['Mean']=model.resid.mean()
                            df1['MAPE']=MAPE(y.values,model.predict(x))
                            df1['Total MSE']=model.mse_total

                            with st.expander("Residuals Data"):
                                st.write("**Residuals**",df1)

                            if st.checkbox("**Want to Download Residual Data**   "):
                                filename=st.text_input("**Write File Name**   ")
                                if filename:
                                    filename=filename+".csv"

                                    st.download_button(
                                        label=f"**Download Data as a CSV File**   ",
                                        data=df1.to_csv(),
                                        file_name=filename,
                                        mime='csv',
                                    )

                        "### »»» Model Data"        
                        wish0=st.checkbox("**Want to see the Model data**")
                        if wish0:
                            data=pd.concat([x,x_test],axis=0)
                            data=data.merge(y,how='left',left_index=True,right_index=True)
                            # data.append(x_test,ignore_index=True)
                            
                            with st.expander("Model Data"):
                                st.write("**Data**",data)
                            
                            
                            if st.checkbox("**Want to Download Model Data**   "):
                                filename=st.text_input("**Write File Name**    ")
                                if filename:
                                    filename=filename+".csv"

                                    st.download_button(
                                        label=f"**Download Data as a CSV File**    ",
                                        data=data.to_csv(),
                                        file_name=filename,
                                        mime='csv',
                                    )

                        "### »»» Model Prediction"
                        wish=st.checkbox("**Want to see the model prediction**")
                            
                        if wish:
                            # col1,col2,col3,col4=st.columns(4)
                            # with col1:
                            #     st.write("**Test Data**",x_test)

                            # with col2:
                            #     st.write("**Prediction**",model.predict(x_test))

                            # Get predictions and confidence intervals
                            predictions = model.predict(x_test)

                            # confidence interval
                            ci = model.get_prediction(x_test).conf_int()
                    

                            # # Calculate lower and upper estimates
                            lower_estimates = predictions - (ci[:, 1] - predictions)
                            upper_estimates = predictions + (ci[:, 1] - predictions)

                            # estimate dataframe

                            estimate=pd.DataFrame({"Lower Estimates":lower_estimates,"Estimates":predictions,"Upper Estimates":upper_estimates})

                        
                            prediction_data=pd.merge(x_test,estimate,left_index=True,right_index=True)
                            
                            with st.expander("**Predicted Data**"):
                                prediction_data

                                

                                # st.write("Lower Estimate",lower_estimates)
                                # st.write("Upper Estimate",upper_estimates)


                            # with st.sidebar:
                            #     st.write("**MAPE**",MAPE(y,model.predict(x)))

                            # with st.sidebar:
                            #     st.write("**Std of Residual**",model.resid.std())

                            # with st.sidebar:
                            #     st.write("**Avg of Residual**",model.resid.mean())
                            #     st.write("**Total MSE**",model.mse_total)

                                # st.write(model.predict(x))
                                # Mape=(model.resid)/y
                                # # Mape=Mape.mean()
                                # st.write(Mape)
                        # wish=st.checkbox("**Want to see MAPE value**")
                        # if wish:
                            
                            if st.checkbox("**Want to Download Prediction Data**    "):
                                    filename=st.text_input("**Write File Name**      ")
                                    if filename:
                                        filename=filename+".csv"

                                        st.download_button(
                                            label=f"**Download Data as a CSV File**      ",
                                            data=prediction_data.to_csv(),
                                            file_name=filename,
                                            mime='csv',
                                        )

                        "### »»» Model Result"
                        choise=st.selectbox("**Want to see the model results**",["Summary","Linearity","Homoscedasticity","Residuals","Q-Q Plot","Residual Plot"])
                        if choise=="Summary":
                            st.write(model.summary())

                            with st.sidebar:
                                st.write("**MAPE**",MAPE(y,model.predict(x)))

                            with st.sidebar:
                                st.write("**Std of Residual**",model.resid.std())

                            with st.sidebar:
                                st.write("**Avg of Residual**",model.resid.mean())
                                st.write("**Total MSE**",model.mse_total)
                                

                                # Breush Pegan
                                # bp_test = het_breuschpagan(model.resid,x)
                                # bp_value = bp_test[0]
                                # bp_pvalue = bp_test[1]

                            

                                # st.write("Breusch-Pagan test statistic: ", bp_value)
                                # st.write("Breusch-Pagan test p-value: ", bp_pvalue)

                    


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







