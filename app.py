from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# Load Model
model = load_model('deployment_24062021')


def run():

    from PIL import Image
    image_badji = Image.open('badji.jpg')

    st.sidebar.info('This app is created using PyCaret and Streamlit')
    st.sidebar.info('Abdoulaye Badji')
    st.sidebar.success('https://www.youtube.com/channel/UCZ_Sor5J6415zV8k3hNlk6Q')
    st.sidebar.image(image_badji)

    st.title('Insurance Application')

    # Capture
    age = st.number_input('Age', min_value = 1, max_value = 100, value = 21)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    if st.checkbox('Smoker'):
        smoker = 'yes'
    else:
        smoker = 'no'

    region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

    output = ""

    # Input Dict
    input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
    input_df = pd.DataFrame([input_dict])

    # Predict
    if st.button('Predict'):
        output = predict_model(model, data = input_df)
        output = '$' + str(output['Label'][0])

    # Display
    st.success('The Insurance amount is {}'.format(output))



if __name__ == '__main__':
    run()
