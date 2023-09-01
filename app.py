import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from PIL import Image


pickle_filename = "model_components (1).pkl"

# Load the model components from the pickle file
with open(pickle_filename, "rb") as pickle_file:
    model_components = pickle.load(pickle_file)


model = model_components["best_rf_regressor"]
lime_explainer = LimeTabularExplainer(
    model_components["X_train"],
    mode = 'regression',
    feature_names = model_components["feature_names"])

def predict_with_lime_explanation(input_features):
    prediction = model.predict([input_features])[0]
    explanation = lime_explainer.explain_instance(
      input_features,
      model.predict,
      num_features = len(model_components["feature_names"]))
    
    explanation_text = 'LIME Explanation:\n'
    for feature, score in explanation.as_list():
      explanation_text += f"f{feature}: {score}\n"

    low_threshold = 100
    high_threshold = 300

  # Function to classify incidence rates
    if prediction < low_threshold:
        incidence_category = 'Low'
    elif prediction <= high_threshold:
        incidence_category = 'Medium'
    else:
        incidence_category = 'High'

  #Return
    return {
      'Prediction': prediction,
      'Incidence_Category': incidence_category,
      'Explanation': explanation_text
  }



def main():
    st.title('Malaria Incidence Rate Prediction')
    st.image(Image.open('malaria.jpg'))
    reported_malaria_cases = st.number_input("What is the reported malaria cases: ")
    death_dueTo_Malaria = st.number_input("What is the death due to malaria : ")
    malaria_cases = st.number_input("Enter malaria cases : ")
    mortality_rate = st.number_input("Enter mortality rate")
    total_population = st.number_input("Enter total population : ")

    st.text("Incidence rate per 1000 population classification threshold")

    df = pd.DataFrame(
        {
           'Incidence Rate' : ['<100', '<300', '>300'],
           'Classification' : ['Low', 'Medium', 'High'] 
        }
    )
    st.table(df)

    st.empty()
    button = st.button("Predict")

    if button:
        with st.spinner("Predicting.."):
            result = predict_with_lime_explanation(np.array([reported_malaria_cases,death_dueTo_Malaria,malaria_cases ,mortality_rate, total_population]))
            st.write('Incidence Category:', result['Incidence_Category'])

            if result['Incidence_Category'] == 'Low':
                st.success('Less than 10% of the population is at risk of malaria given the current population and malaria cases')

            elif result['Incidence_Category'] == 'Medium':
                st.info('Between 10 to 30% of the population is at risk of malaria given the current population and malaria cases')
            else:
                st.warning('More than 30% of the population is at risk of malaria given the current population and malaria cases')

if __name__== '__main__':
    main()       