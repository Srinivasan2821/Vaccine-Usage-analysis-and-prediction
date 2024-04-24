# Packages

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle

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


    with open(r"C:\Users\ELCOT\Desktop\Srini\VS Code\Projects\Vaccine\Data\Classification_model.pkl","rb") as f:
        regg_model= pickle.load(f)

    user_data = np.array([[h1n1_worry,h1n1_awareness,antiviral_medication,contact_avoidance,bought_face_mask,wash_hands_frequently,
                            avoid_large_gatherings,reduced_outside_home_cont,avoid_touch_face,dr_recc_h1n1_vacc,dr_recc_seasonal_vacc,
                            chronic_medic_condition,cont_child_undr_6_mnths,is_health_worker,has_health_insur,is_h1n1_vacc_effective,
                            is_h1n1_risky,sick_from_h1n1_vacc,is_seas_vacc_effective,is_seas_risky,sick_from_seas_vacc,age_bracket_2,qualification_2,
                            race_2,sex_2,income_level_2,marital_status_2,housing_status_2,employment_2,census_msa_2,no_of_adults,no_of_children]])
    y_pred_1 = regg_model.predict(user_data)
    price= np.exp(y_pred_1[0])

    return round(price)

st.set_page_config(layout="wide")
st.title(":rainbow[_SINGAPORE RESALE FLAT PRICES PREDICTING :red[: ▶]_]")
st.write("")

with st.sidebar:
    Menu=st.sidebar.selectbox(":red[_**Please Select The Menu:-**_]",("Home", "Vaccine Prediction"))
    with st.sidebar:
            st.header(":red[_Skill:-_]")
            st.write(':blue[ :star: Python scripting]') 
            st.write(':blue[ :star: Data Wrangling]')
            st.write(':blue[ :star: EDA]')
            st.write(':blue[ :star:  Model Building]')
            st.write(':blue[ :star:  Model Deployment]')

if Menu == "Home":

    st.header("ABOUT THIS PROJECT")

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

        st.write("## :green[**The Predicted is :**]",pre_price)

