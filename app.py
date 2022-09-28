
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import time

import pandas as pd
import numpy as np
# # read Dataset if you use
# df = pd.read_csv("./")

import pickle
enc = pickle.load(open("final_model_credit_score_enc", "rb"))
le = pickle.load(open("final_model_credit_score_le", "rb"))
model = pickle.load(open("final_model_credit_score_pred", "rb"))


st.markdown("<h2 style='text-align:center; color:floralWhite;'>Credit Score Classification</h2>", unsafe_allow_html=True)
# st.title("Car Price Prediction")
# st.header('This is a header')
# st.subheader('Car Price Prediction')
# st.text('This is some text.')
# st.write('Hello, *World!* :sunglasses:')
# st.success('This is a success message!')
# st.info('This is a purely informational message')
# st.error('This is an error')

col1, col2, col3 = st.columns([1,8,1]) 
#Image, df
try:
    # Some Code
#     #read local image
#     img1 = Image.open("images.jpeg")

    #image url
    url = "https://storage.googleapis.com/kaggle-datasets-images/2289007/3846912/ad5e128929f5ac26133b67a6110de7c0/dataset-cover.jpg?t=2022-06-22-14-33-45"
    
    with col2:
        st.image(url, caption="Kaggle: Credit score classification")
        
        st.markdown('[Kaggle: Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)')
        st.markdown('[Kaggle: Credit Score Classification Data Cleaning Project](https://www.kaggle.com/code/clkmuhammed/credit-score-classification-data-cleaning-project/notebook)')
        st.markdown('[Kaggle: Credit Score Classification LogReg-RF-XGB & Deploy](https://www.kaggle.com/code/clkmuhammed/credit-score-classification-logreg-rf-xgb-deploy/notebook)')
#         if st.checkbox('Show dataframe'):
#             st.write(df)    

except:
    # Executed if error in the
    # try block   
    components.html('''
    <script>
        alert("Image Not Loading!");
    </script>
    ''')
    st.text("Image Not Loading!")
    
else:
    # execute if no exception 
    pass
        
finally:
    # Some code .....(always executed)   
    pass


# Creating side bar 
st.sidebar.header("User input parameter")

def user_input_data():
    Credit_Mix = st.sidebar.selectbox('Credit_Mix:', ['Standard', 'Bad', 'Good'])
    Interest_Rate = st.sidebar.slider('Interest_Rate', 1, 34, 14, 1)
    Outstanding_Debt = st.sidebar.slider('Outstanding_Debt', 0, 5000, 1426, 1)
    Delay_from_due_date = st.sidebar.slider('Delay_from_due_date', 0, 62, 21, 1)
    Total_EMI_per_month = st.sidebar.slider('Total_EMI_per_month', 0, 2000, 107, 1)
    Changed_Credit_Limit = st.sidebar.slider('Changed_Credit_Limit', 0, 30, 10, 1)
    Monthly_Inhand_Salary = st.sidebar.slider('Monthly_Inhand_Salary', 303, 15000, 4197, 1)
    Annual_Income = st.sidebar.slider('Annual_Income', 7000, 180000, 50505, 1)
    
    html_temp = """
    <div style="background-color:tomato;padding:1.5px">
    <h1 style="color:white;text-align:center;">Single Customer </h1>
    </div><br>"""
    st.sidebar.markdown(html_temp,unsafe_allow_html=True)
    
    data = { 
        'Credit_Mix': Credit_Mix,
        'Interest_Rate': Interest_Rate,
        'Outstanding_Debt': Outstanding_Debt,
        'Delay_from_due_date': Delay_from_due_date,
        'Total_EMI_per_month': Total_EMI_per_month,
        'Changed_Credit_Limit': Changed_Credit_Limit,
        'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Annual_Income': Annual_Income,
    }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data


#show input
col1, col2 = st.columns([4, 6])

df = user_input_data() 
with col1:
    if st.checkbox('Show User Inputs:', value=True):
        st.write(df.astype(str).T.rename(columns={0:'input_data'}))

with col2:
    for i in range(2): 
        st.markdown('#')
    if st.button('Make Prediction'):   
        sound = st.empty()
        # assign for music sound
        video_html = """
            <iframe width="0" height="0" 
            src="https://www.youtube-nocookie.com/embed/t3217H8JppI?rel=0&amp;autoplay=1&mute=0&start=2860&amp;end=2866&controls=0&showinfo=0" 
            allow="autoplay;"></iframe>
            """
        sound.markdown(video_html, unsafe_allow_html=True)
       
        cat = ['Credit_Mix']
        df[cat] = enc.transform(df[cat]) 
        prediction = model.predict(df)
        prediction = le.inverse_transform(prediction)[0]

        time.sleep(2.1)  # wait for 2 seconds to finish the playing of the audio
        sound.empty()  # optionally delete the element afterwards   
        
        st.success(f'Credit score probability is:&emsp;{prediction}')
