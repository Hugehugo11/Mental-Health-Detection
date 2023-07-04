# import important packages
import streamlit as st
import joblib
import pandas as pd
from os.path import dirname, join, realpath
import joblib

# add banner image
st.header(" Detection of Mental Health")
st.image("image/image 2.PNG")
st.subheader(
    """
A simple app that predicts if an individual suffers depression.
"""
)

# form to collect user information
our_form = st.form(key="busara_form")

femaleres = our_form.selectbox("What is the Client's Gender",("Male","Female")),
age = our_form.number_input("Enter the age of the client", min_value=1, max_value=100),
married = our_form.selectbox("Is the Client married?",("Yes","No")),
edu = our_form.number_input("How many years did the Client spent on education?", min_value=0,max_value=30),
hh_totalmembers= our_form.number_input("How many members in the Client's house",min_value=1,max_value=100),
asset_durable= our_form.number_input("How much value for durable assets in usd",min_value=0,max_value=1000),
asset_phone= our_form.number_input("How much value of your phone in usd",min_value=0,max_value=1000),
cons_alcohol = our_form.number_input("How much do you spend on alcohol?(USD)", min_value=0,max_value=1000),
ent_business = our_form.selectbox("Non agriculture business as primary income",("Yes","No")),
ent_employees = our_form.number_input("How many Employees do you have?", min_value=0,max_value=100),  
ent_nonag_flowcost = our_form.number_input("Non agriculture business flow in usd", min_value=0,max_value=1000), 
ent_total_cost = our_form.number_input("total expenses monthly", min_value=0,max_value=1000), 
fs_adwholed_often = our_form.number_input("What is the number of whole days without food last month for an adult?", min_value=0,max_value=30)
fs_chskipm_often = our_form.number_input("How many meals skipped by children?", min_value=0,max_value=3), 
fs_chwholed_often= our_form.number_input("How many days children go without eating ?", min_value=0,max_value=30),
fs_enoughtom = our_form.selectbox("Is there enough food in the house for tomorrow?",("Yes","No")),
med_portion_sickinjured= our_form.number_input("What is the proportion of household sick/injured in a month?", min_value=0,max_value=1),
med_healthconsult= our_form.number_input("What is the proportion of illnesses where a doctor was consulted in a month?", min_value=0,max_value=1),
ed_expenses_perkid= our_form.number_input("What is the education expenses of the Clients kid?", min_value=0,max_value=1000),
nondurable_investment= our_form.number_input("What is the cost of nondurable investment?", min_value=0,max_value=1000),
given_mpesa = our_form.selectbox("Have you ever given mpesa",("Yes","No")),
amount_saved_mpesa= our_form.number_input("What is the amount saved mpesa?", min_value=0,max_value=1000)

submit = our_form.form_submit_button(label="make prediction")


# load the model and scaler

with open(
    join(dirname(realpath(__file__)), "hist.pkl"),
    "rb",
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "minmax-scaler.pkl"), "rb"
) as f:
    scaler = joblib.load(f)
    

def femaleres_transform(value):
	if value=='Female':
		return 1
	else: 
		return 0
		
def married_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0

def fs_enoughtom_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
	
def given_mpesa_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
	
def ent_business_transform(value):
	if value=='Yes':
		return 1
	else: 
		return 0
    

@st.cache_resource
# function to clean and tranform the input
def preprocessing_data(data, _scaler):
      

    data = scaler.transform(data.values.reshape(-1,1))
    feat_col = ['femaleres', 'age', 'married', 'edu','hh_totalmembers','asset_durable', 'asset_phone', 'cons_alcohol', 'ent_business', 'ent_employees','ent_nonag_flowcost', 'ent_total_cost', 'fs_adwholed_often',
       'fs_chskipm_often', 'fs_chwholed_often', 'fs_enoughtom', 'med_portion_sickinjured','med_healthconsult',' ed_expenses_perkid','nondurable_investment', 'given_mpesa', 'amount_saved_mpesa']

    return pd.DataFrame(data.reshape(-1,22),columns = feat_col)


    # Convert the following numerical labels from integer to float
    #float_array = data[[ "age", "children", "hhsize", "edu", "hh_totalmembers", "asset_livestock", "ent_farmexpenses", "fs_adwholed_often", "med_sickdays_hhave"]].values.astype(
     #   float
    #)

     # scale our data into range of 0 and 1
    #data = _scaler.fit_transform(data)
    #print(data)
    #return data
if submit:

    # collect inputs
    input = {
        "femaleres": femaleres_transform(femaleres),
        "age": age,
        "married": married_transform(married),
        "edu": edu,
	    "hh_totalmembers": hh_totalmembers,
	    "asset_durable":asset_durable,
        "asset_phone":asset_phone,
	    "cons_alcohol":cons_alcohol,
	    "ent_business": ent_business_transform(ent_business),
        "ent_employees": ent_employees, 
        "ent_nonag_flowcost": ent_nonag_flowcost,
        "ent_total_cost": ent_total_cost,
        "fs_adwholed_often": fs_adwholed_often,
        "fs_chskipm_often":fs_chskipm_often, 
        "fs_chwholed_often":fs_chwholed_often,
        "fs_enoughtom": fs_enoughtom_transform(fs_enoughtom), 
        "med_portion_sickinjured": med_portion_sickinjured,
	    "med_healthconsult": med_healthconsult,
	    "ed_expenses_perkid": ed_expenses_perkid,
	    "nondurable_investment": nondurable_investment,
	    "given_mpesa": given_mpesa_transform(given_mpesa),
	    "amount_saved_mpesa": amount_saved_mpesa,
        
    	#"asset_livestock": asset_livestock,
	    #"ent_farmexpenses": ent_farmexpenses,
        #"med_sickdays_hhave": med_sickdays_hhave,
	    
    }
    # create a dataframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data, _scaler=scaler)
    print(transformed_data)
    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the RFC task
    st.header("Results")
    if output == 1:
        st.write(
            "Theh Client is likely to be depressed with probability of {} 😔".format(
                probability
            )
        )
    elif output == 0:
        st.write(
            "The Client is likely not to be depressed with probability of {} 😊".format(
                probability
            )
        )
url = "https://github.com/@HugeHugo10"
st.write("Developed with @ by [Huge Hugo](%s)" % url)
