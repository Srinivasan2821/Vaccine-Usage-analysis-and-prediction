# Packages

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def age_bracket_mapping(age_bracket):

    if age_bracket == '55 - 64 Years':
        age_bracket_1= int(3)
    elif age_bracket == '35 - 44 Years':
        age_bracket_1= int(1)
    elif age_bracket == '18 - 34 Years':
        age_bracket_1= int(0)
    elif age_bracket == '65+ Years':
        age_bracket_1= int(4)
    elif age_bracket == '45 - 54 Years':
        age_bracket_1= int(2)

    return age_bracket_1

def qualification_mapping(qualification):

    if qualification == '< 12 Years':
        qualification_1= int(1)
    elif qualification == '12 Years':
        qualification_1= int(0)
    elif qualification == 'College Graduate':
        qualification_1= int(2)
    elif qualification == 'Some College':
        qualification_1= int(3)

    return qualification_1

def race_mapping(race):

    if race == 'White':
        race_1= int(3)
    elif race == 'Black':
        race_1= int(0)
    elif race == 'Other or Multiple':
        race_1= int(2)
    elif race == 'Hispanic':
        race_1= int(1)

    return race_1

def sex_mapping(sex):

    if sex == 'Female':
        sex_1= int(0)
    elif sex == 'Male':
        sex_1= int(1)

    return sex_1

def income_level_mapping(income_level):

    if income_level == 'Below Poverty':
        income_level_1= int(2)
    elif income_level == '<= $75,000, Above Poverty':
        income_level_1= int(0)
    elif income_level == '> $75,000':
        income_level_1= int(1)

    return income_level_1

def marital_status_mapping(marital_status):

    if marital_status == 'Not Married':
        marital_status_1= int(1)
    elif marital_status == 'Married':
        marital_status_1= int(0)

    return marital_status_1

def housing_status_mapping(housing_status):

    if housing_status == 'Own':
        housing_status_1= int(0)
    elif housing_status == 'Rent':
        housing_status_1= int(1)

    return housing_status_1

def employment_mapping(employment):

    if employment == 'Not in Labor Force':
        employment_1= int(1)
    elif employment == 'Employed':
        employment_1= int(0)
    elif employment == 'Unemployed':
        employment_1= int(2)

    return employment_1

def census_msa_mapping(census_msa):

    if census_msa == 'Non-MSA':
        census_msa_1= int(2)
    elif census_msa == 'MSA, Not Principle  City':
        census_msa_1= int(0)
    elif census_msa == 'MSA, Principle City':
        census_msa_1= int(1)

    return census_msa_1

def predict_price(h1n1_worry,h1n1_awareness,antiviral_medication,contact_avoidance,bought_face_mask,wash_hands_frequently,
                avoid_large_gatherings,reduced_outside_home_cont,avoid_touch_face,dr_recc_h1n1_vacc,dr_recc_seasonal_vacc,
                chronic_medic_condition,cont_child_undr_6_mnths,is_health_worker,has_health_insur,is_h1n1_vacc_effective,
                is_h1n1_risky,sick_from_h1n1_vacc,is_seas_vacc_effective,is_seas_risky,sick_from_seas_vacc,age_bracket,qualification,
                race,sex,income_level,marital_status,housing_status,employment,census_msa,no_of_adults,no_of_children):
    
    age_bracket_2= age_bracket_mapping(age_bracket)
    qualification_2= qualification_mapping(qualification)
    race_2= race_mapping(race)
    sex_2= sex_mapping(sex)
    income_level_2= income_level_mapping(income_level)
    marital_status_2= marital_status_mapping(marital_status)
    housing_status_2= housing_status_mapping(housing_status)
    employment_2= employment_mapping(employment)
    census_msa_2= census_msa_mapping(census_msa)


    with open("Classification_model.pkl","rb") as f:
        regg_model= pickle.load(f)

    user_data = np.array([[h1n1_worry,h1n1_awareness,antiviral_medication,contact_avoidance,bought_face_mask,wash_hands_frequently,
                            avoid_large_gatherings,reduced_outside_home_cont,avoid_touch_face,dr_recc_h1n1_vacc,dr_recc_seasonal_vacc,
                            chronic_medic_condition,cont_child_undr_6_mnths,is_health_worker,has_health_insur,is_h1n1_vacc_effective,
                            is_h1n1_risky,sick_from_h1n1_vacc,is_seas_vacc_effective,is_seas_risky,sick_from_seas_vacc,age_bracket_2,qualification_2,
                            race_2,sex_2,income_level_2,marital_status_2,housing_status_2,employment_2,census_msa_2,no_of_adults,no_of_children]])
    y_pred_1 = regg_model.predict(user_data)
    price= np.exp(y_pred_1[0])

    return round(price)

df= pd.read_csv("Vaccine_proper.csv")

st.set_page_config(layout="wide")
st.title(":rainbow[_Vaccine Usage analysis and prediction :red[: ▶]_]")
st.write("")

with st.sidebar:
    Menu=st.sidebar.selectbox(":red[_**Please Select The Menu:-**_]",("Home", "Data Exploration", "Vaccine Prediction"))
    with st.sidebar:
            st.header(":red[_Skill:-_]")
            st.write(':blue[ :star: Pandas]') 
            st.write(':blue[ :star: Data Visualisation]')
            st.write(':blue[ :star: Machine Learning]')


if Menu == "Home":

    st.header("ABOUT THIS PROJECT")

    st.subheader(":orange[1. DATA ENGINEERING:]")

    st.write('''***1.Understand the complete dataset and Engineer it as per the needs***''')
    st.write('''***2.Do Exploratory data analysis to get deeper understanding about the  attributes.***''')
    
    st.subheader(":orange[2. DASHBOARD DEVELOPMENT:]")

    st.write('''***The Plotly or streamlit analytical Dashboard is expected to be developed that has at least 5 types  of dynamic charts and at least 3 data filtering options.***''')
    
    st.subheader(":orange[3. MODEL DEVELOPMENT:]")

    st.write('''***Training a Logistic Regression model (or any suitable model)for prediction.***''')
    st.write('''***Model tuning and evaluation to improve predictive performance.***''')
    st.write('''***Developing a predictive model that can help in targeting vaccination campaigns effectively.***''')
    st.write('''***Create a model serving api once the model is ready.***''')
    st.write('''***Develop an application using flask or plotly or streamlit that can accept features as input and provide prediction to the end user.***''')
    st.write('''***Host the application using any free hosting service provide (example: render.com, pythonanywhere,etc).***''')



elif Menu == "Data Exploration":
    tab1, tab2, tab3= st.tabs(["***Age_Analysis***","***Employment***","***h1n1 Effect***"])
    with tab1:
        st.title("**Age Analysis**")
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

    with tab2:
        st.title("**Employment Analysis**")
        col1,col2= st.columns(2)

        with col1:
            employment= st.selectbox("Select the employment",df["employment"].unique())

            df1_e= df[df["employment"] == employment]
            df1_e.reset_index(drop= True, inplace= True)

            census_msa= st.selectbox("Select the census_msa",df1_e["census_msa"].unique())

            df2_e= df1_e[df1_e["census_msa"] == census_msa]
            df2_e.reset_index(drop= True, inplace= True)

            df_e_sunb_in= px.sunburst(df2_e, path=["income_level","marital_status","housing_status"], values="h1n1_vaccine",width=600,height=500,title="h1n1_vaccine",color_discrete_sequence=px.colors.sequential.Peach_r)
            st.plotly_chart(df_e_sunb_in)
        
        with col2:
            sex= st.selectbox("Select the sex",df["sex"].unique())

            df3_e= df[df["sex"] == sex]
            df3_e.reset_index(drop= True, inplace= True) 

            age_bracket= st.selectbox("Select the age_bracket",df3_e["age_bracket"].unique())

            df4_e= df3_e[df3_e["age_bracket"] == age_bracket]
            df4_e.reset_index(drop= True, inplace= True)

            df_e_sunb_q= px.sunburst(df4_e, path=["qualification","race","employment"], values="h1n1_vaccine",width=600,height=500,title="Employment",color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(df_e_sunb_q)

        col1,col2= st.columns(2)

        with col1:

            fig_do_bar_3 = px.bar(df1_e, x='employment', y=['is_h1n1_vacc_effective', 'is_h1n1_risky', 'sick_from_h1n1_vacc'], 
                title='H1N1',hover_data="h1n1_vaccine",
                barmode='group',color_discrete_sequence=px.colors.sequential.Rainbow_r, width= 600, height= 500) 
            
            st.plotly_chart(fig_do_bar_3)

        with col2:

            fig_do_bar_4 = px.bar(df1_e, x='employment', y=['is_seas_vacc_effective', 'is_seas_risky', 'sick_from_seas_vacc'], 
                title='Seas',hover_data="h1n1_vaccine",
                barmode='group',color_discrete_sequence=px.colors.sequential.Redor_r, width= 600, height= 500) 
            
            st.plotly_chart(fig_do_bar_4)  

    with tab3:
        st.title("**H1N1 Effect**")
        col1,col2= st.columns(2)

        with col1:

            df_bar3= pd.DataFrame(df3_e.groupby("age_bracket")[["h1n1_worry","h1n1_awareness","antiviral_medication",'contact_avoidance','bought_face_mask','wash_hands_frequently','avoid_large_gatherings','reduced_outside_home_cont']].sum())
            df_bar3.reset_index(inplace= True)

            fig_do_bar4 = px.line(df_bar3, x='age_bracket', y=["h1n1_worry","h1n1_awareness","antiviral_medication",'contact_avoidance','bought_face_mask','wash_hands_frequently','avoid_large_gatherings','reduced_outside_home_cont'], 
                title='Age',color_discrete_sequence=px.colors.sequential.Rainbow, width=600, height=500)

            st.plotly_chart(fig_do_bar4)

        with col2:

            df_bar4= pd.DataFrame(df3_e.groupby("age_bracket")[["avoid_touch_face","dr_recc_h1n1_vacc","dr_recc_seasonal_vacc",'chronic_medic_condition','cont_child_undr_6_mnths','is_health_worker','has_health_insur','h1n1_vaccine']].sum())
            df_bar4.reset_index(inplace= True)

            fig_do_bar5 = px.line(df_bar4, x='age_bracket', y=["avoid_touch_face","dr_recc_h1n1_vacc","dr_recc_seasonal_vacc",'chronic_medic_condition','cont_child_undr_6_mnths','is_health_worker','has_health_insur','h1n1_vaccine'], 
                title='Age',color_discrete_sequence=px.colors.sequential.Redor_r, width=600, height=500)

            st.plotly_chart(fig_do_bar5)

        

        df_bar5= pd.DataFrame(df.groupby("age_bracket")[["no_of_adults","no_of_children","h1n1_vaccine"]].sum())
        df_bar5.reset_index(inplace= True)

        fig_do_bar6 = px.scatter(df_bar5, x='age_bracket', y=["no_of_adults","no_of_children","h1n1_vaccine"], 
            title='Age',color_discrete_sequence=px.colors.sequential.BuPu_r, width=600, height=500)

        st.plotly_chart(fig_do_bar6)
        

elif Menu == "Vaccine Prediction":

    col1,col2,col3,col4= st.columns(4)
    with col1:
        h1n1_worry= st.selectbox("Select the h1n1_worry",[0,1,2,3])
        h1n1_awareness= st.selectbox("Select the h1n1_awareness",[0,1,2])
        antiviral_medication= st.selectbox("Select the antiviral_medication",[0,1])
        contact_avoidance= st.selectbox("Select the contact_avoidance",[0,1])
        bought_face_mask= st.selectbox("Select the bought_face_mask",[0,1])
        wash_hands_frequently= st.selectbox("Select the wash_hands_frequently",[0,1])
        avoid_large_gatherings= st.selectbox("Select the avoid_large_gatherings",[0,1])
        reduced_outside_home_cont= st.selectbox("Select the reduced_outside_home_cont",[0,1])
        
    with col2:
        avoid_touch_face= st.selectbox("Select the avoid_touch_face",[0,1])
        dr_recc_h1n1_vacc= st.selectbox("Select the dr_recc_h1n1_vacc",[0,1])
        dr_recc_seasonal_vacc= st.selectbox("Select the dr_recc_seasonal_vacc",[0,1])
        chronic_medic_condition= st.selectbox("Select the chronic_medic_condition",[0,1])
        cont_child_undr_6_mnths= st.selectbox("Select the cont_child_undr_6_mnths",[0,1])
        is_health_worker= st.selectbox("Select the is_health_worker",[0,1])
        has_health_insur= st.selectbox("Select the has_health_insur",[0,1])
        is_h1n1_vacc_effective= st.selectbox("Select the is_h1n1_vacc_effective",[1,2,3,4,5])

    with col3:
        is_h1n1_risky= st.selectbox("Select the is_h1n1_risky",[1,2,3,4,5])
        sick_from_h1n1_vacc= st.selectbox("Select the sick_from_h1n1_vacc",[1,2,3,4,5])
        is_seas_vacc_effective= st.selectbox("Select the is_seas_vacc_effective",[1,2,3,4,5])
        is_seas_risky= st.selectbox("Select the is_seas_risky",[1,2,3,4,5])
        sick_from_seas_vacc= st.selectbox("Select the sick_from_seas_vacc",[1,2,3,4,5])
        age_bracket= st.selectbox("Select the age_bracket",['55 - 64 Years', '35 - 44 Years', '18 - 34 Years', '65+ Years','45 - 54 Years'])
        qualification= st.selectbox("Select the qualification",['< 12 Years', '12 Years', 'College Graduate', 'Some College'])
        race= st.selectbox("Select the race",['White', 'Black', 'Other or Multiple', 'Hispanic'])

    with col4:
        sex= st.selectbox("Select the sex",['Female', 'Male'])
        income_level= st.selectbox("Select the income_level",['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'])
        marital_status= st.selectbox("Select the marital_status",['Not Married', 'Married'])
        housing_status= st.selectbox("Select the housing_status",['Own', 'Rent'])
        employment= st.selectbox("Select the employment",['Not in Labor Force', 'Employed', 'Unemployed'])
        census_msa= st.selectbox("Select the census_msa",['Non-MSA', 'MSA, Not Principle  City', 'MSA, Principle City'])
        no_of_adults= st.selectbox("Select the no_of_adults",[0,1,2,3])
        no_of_children= st.selectbox("Select the no_of_children",[0,1,2,3])


    button= st.button("Predict the Vaccine Usage", use_container_width= True)

    if button:

            
        pre_price= predict_price(h1n1_worry,h1n1_awareness,antiviral_medication,contact_avoidance,bought_face_mask,wash_hands_frequently,
                avoid_large_gatherings,reduced_outside_home_cont,avoid_touch_face,dr_recc_h1n1_vacc,dr_recc_seasonal_vacc,
                chronic_medic_condition,cont_child_undr_6_mnths,is_health_worker,has_health_insur,is_h1n1_vacc_effective,
                is_h1n1_risky,sick_from_h1n1_vacc,is_seas_vacc_effective,is_seas_risky,sick_from_seas_vacc,age_bracket,qualification,
                race,sex,income_level,marital_status,housing_status,employment,census_msa,no_of_adults,no_of_children)

        if pre_price == 1:
            st.write("## :green[**The H1N1 vaccine : Yes**]")
        else:
            st.write("## :red[**The H1N1 vaccine : No**]")

