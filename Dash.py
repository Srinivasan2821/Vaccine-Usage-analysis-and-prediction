import pandas as pd
import streamlit as st
pd.set_option('display.max_columns', None)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(layout= "wide")
st.title(":rainbow[_AIRBNB DATA ANALYSIS :red[: ▶]_]")


df= pd.read_csv(r"C:\Users\ELCOT\Desktop\Srini\VS Code\Projects\Vaccine\Data\Vaccine_proper.csv")

Menu=st.sidebar.selectbox(":red[_**Please Select The Menu:-**_]",("Home", "Data Exploration"))

if Menu == "Home":
    with st.sidebar:
            st.header(":red[_Skill:-_]")
            st.write(':blue[ :star: Python scripting]') 
            st.write(':blue[ :star: Data Preprocessing]')
            st.write(':blue[ :star: Visualization]')
            st.write(':blue[ :star: EDA]')
            st.write(':blue[ :star: Streamlit]')
            st.write(':blue[ :star: MongoDb]')
            st.write(':blue[ :star: PowerBI or Tableau]')

    st.header("ABOUT THIS PROJECT")

if Menu == "Data Exploration":
    tab1, tab2, tab3= st.tabs(["***Age_Bracket***","***AVAILABILITY ANALYSIS***","***LOCATION BASED***"])
    with tab1:
        st.title("**Age Bracket**")
        col1,col2= st.columns(2)

        with col1:
            
            
            age= st.selectbox("Select the age",df["age_bracket"].unique())

            df1= df[df["age_bracket"] == age]
            df1.reset_index(drop= True, inplace= True)

            df_bar= pd.DataFrame(df1.groupby("sex")[["no_of_adults","no_of_children","h1n1_vaccine"]].sum())
            df_bar.reset_index(inplace= True)

            fig_bar= px.bar(df_bar, x='sex', y= "h1n1_vaccine", title= "h1n1_vaccine",hover_data=["no_of_adults","no_of_children"],color_discrete_sequence=px.colors.sequential.Redor_r, width=600, height=500)
            st.plotly_chart(fig_bar)

        
        with col2:
     
            housing= st.selectbox("Select the housing_status",df["housing_status"].unique())

            df2= df[df["housing_status"] == housing]
            df2.reset_index(drop= True, inplace= True)

            df_pie= pd.DataFrame(df2.groupby("age_bracket")[["h1n1_vaccine","is_health_worker"]].sum())
            df_pie.reset_index(inplace= True)

            fig_pi= px.pie(df_pie, values="h1n1_vaccine", names= "age_bracket",
                            hover_data=["is_health_worker"],
                            color_discrete_sequence=px.colors.sequential.BuPu_r,
                            title="housing_status",
                            width= 600, height= 500)
            st.plotly_chart(fig_pi)

        col1,col2= st.columns(2)

        with col1:

            df_do_bar= pd.DataFrame(df1.groupby("qualification")[["no_of_adults","employment"]].sum())
            df_do_bar.reset_index(inplace= True)

            fig_do_bar = px.line(df_do_bar, x='qualification', y=['no_of_adults'], 
            title='Qualification',hover_data="employment",  
            color_discrete_sequence=px.colors.sequential.Rainbow, width=600, height=500)
            

            st.plotly_chart(fig_do_bar)

        with col2:

            df_do_bar_2= pd.DataFrame(df2.groupby("employment")[["no_of_adults","no_of_children","h1n1_vaccine"]].sum())
            df_do_bar_2.reset_index(inplace= True)

            fig_do_bar_2 = px.bar(df_do_bar_2, x='employment', y=['no_of_adults', 'no_of_children'], 
            title='Employment',hover_data="h1n1_vaccine",
            barmode='group',color_discrete_sequence=px.colors.sequential.Rainbow_r, width= 600, height= 500)
           
            st.plotly_chart(fig_do_bar_2)