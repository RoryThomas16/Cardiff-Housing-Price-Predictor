# imports
import numpy as np
import streamlit as st 
import streamlit.components.v1 as components
import pickle
import os
import pandas as pd

def list_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Filter to show only files (exclude directories)
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]

        if files:
            print("Files in directory:", directory_path)
            return files
        else:
            print("No files found in the directory.")
            return None
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
    
model = {}
current_directory = os.getcwd()
model_directory = os.path.join(current_directory, 'models')
for idx, file in enumerate(list_files_in_directory(model_directory)):
    #model.append(pickle.load(os.path.join(model_directory, file)))
    with open(os.path.join(model_directory,file), 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        if 'propTypeEncoder' in file:
            propTypeEncoder = content
        elif 'scaler' in file:
            scaler = content
        elif 'poly' in file:
            poly = content
        else:
            name = file.split('.')[0].split('_')[-1]
            model[name] = content
        
# Streamlit UI
st.title('Roath (Cardiff) House Price Predictor - by Rory Thomas')

st.header("Data Set used in Training")
HtmlFile = open("img\\mapped_dataset.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, width=700, height=500, scrolling=False)
    
st.header("Data Input")
# Input fields for the model (you can adjust these based on the features your model requires)
input_feature_1 = st.number_input('Year', min_value=1970, max_value=2050, value=2025)
input_feature_2 = st.number_input('Bedrooms', min_value=1, max_value=6, value=1)
input_feature_3 = st.selectbox('Property Type', ('TERRACED', 'SEMI_DETACHED', 'DETACHED','FLAT'))
input_feature_4 = st.number_input('Bathrooms', min_value=1, max_value=6, value=1)

# Collect all the input features into a numpy array
input_data = np.array([input_feature_1, input_feature_2, input_feature_3, input_feature_4])

# Prep the input data for the model
input_data[2] = propTypeEncoder.transform(input_data[2].reshape(-1,1))[0]
input_data = scaler.transform(input_data.reshape(1,-1))

# Button to trigger prediction
_, button_col, _ = st.columns(3) # Centre a button
if input_feature_1 or input_feature_2 or input_feature_3 or input_feature_4:
#if button_col.button('Make Prediction'):
    # Make a prediction using the model
    predictions = {}
    for name, model in model.items():
        if name == 'ridge':
            temp_input_data = poly.transform(input_data.copy())
            prediction = model.predict(temp_input_data)
        else:
            prediction = model.predict(input_data)
        predictions[name] = prediction[0]
    
    # Display the highest and lowest predictions
    highest_model = max(predictions, key=predictions.get)
    mean_prediction = np.mean(list(predictions.values()))
    lowest_model = min(predictions, key=predictions.get)
    
    st.write(f"The highest predicted price is from the model '{highest_model}': £{predictions[highest_model]:.2f}")
    st.write(f"The lowest predicted price is from the model '{lowest_model}': £{predictions[lowest_model]:.2f}")
    st.write(f"The mean predicted price is: £{mean_prediction:.2f}")

    # Convert the dictionary to a pandas DataFrame for plotting
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Predicted Price'])
    # Plotting the bar chart
    st.bar_chart(predictions_df.set_index('Model')['Predicted Price'])