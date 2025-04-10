# imports
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pickle
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 
plotly_map_year = "mapped_dataset_with_slider.html"
plotly_map_region = "mapped_dataset_with_region.html"
plotly_dataset = "rightmove_housing_data_20250410_235533.csv"
# 

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
model_directory = os.path.join(os.getcwd(), 'models')  
img_directory = os.path.join(os.getcwd(), 'img')
data_directory = os.path.join(os.getcwd(), 'data')
for idx, file in enumerate(list_files_in_directory(model_directory)):
    with open(os.path.join(model_directory, file), 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        if 'propTypeEncoder' in file:
            propTypeEncoder = content
        elif 'regionEncoder' in file:
            regionEncoder = content
        elif 'scaler' in file:
            scaler = content
        elif 'poly' in file:
            poly = content
        else:
            name = file.split('.')[0].split('_')[-1]
            model[name] = content

# Streamlit UI
st.title('Cardiff House Price Predictor - by Rory Thomas')

# Tabs for HTML visualization and input/prediction
tab1, tab2 = st.tabs(["Prediction", "Training Data Visualization", ])

# Load your dataset (replace with the actual dataset used for training)
dataset_path = os.path.join(data_directory, plotly_dataset)  # Adjust the path as needed
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)

    # Extract 'year_sold' from 'date_sold' and store it as an integer
    df['year_sold'] = pd.to_datetime(df['date_sold']).dt.year
    df['display_price'] = df['display_price'].apply(lambda x: float(x.replace('£', '').replace(',', '')) if isinstance(x, str) else x)
            
with tab2:
    st.header("Data Set used in Training - Sliced by Year")
    HtmlFile = open(os.path.join(img_directory, plotly_map), 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=700, height=500, scrolling=False)
    st.header("Data Set used in Training - Sliced by Region")
    HtmlFile = open(os.path.join(img_directory, plotly_map), 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=700, height=500, scrolling=False)

with tab1:
    # Input fields for the model (you can adjust these based on the features your model requires)
    input_feature_1 = st.number_input('Year', min_value=1970, max_value=2050, value=2025)
    input_feature_2 = st.number_input('Bedrooms', min_value=1, max_value=6, value=1)
    input_feature_3 = st.selectbox('Property Type', ('TERRACED', 'SEMI_DETACHED', 'DETACHED', 'FLAT'))
    input_feature_4 = st.number_input('Bathrooms', min_value=1, max_value=6, value=1)
    input_feature_5 = st.selectbox('Region', df['region'].unique())  # New input for region

    # Collect all the input features into a numpy array
    input_data = np.array([input_feature_1, input_feature_2, input_feature_3, input_feature_4, input_feature_5])

    # Prep the input data for the model
    input_data[2] = propTypeEncoder.transform(input_data[2].reshape(-1, 1))[0]
    input_data[4] = regionEncoder.transform(input_data[4].reshape(-1, 1))[0]
    input_data = scaler.transform(input_data.reshape(1, -1))

    # Button to trigger prediction
    _, button_col, _ = st.columns(3)  # Centre a button
    if input_feature_1 or input_feature_2 or input_feature_3 or input_feature_4 or input_feature_5:
        # Make a prediction using the model
        predictions = {}
        for name, model in model.items():
            if name == 'ridge':
                temp_input_data = poly.transform(input_data.copy())
                prediction = model.predict(temp_input_data)
            else:
                prediction = model.predict(input_data)
            predictions[name] = prediction[0]

        # Display the highest, lowest, and mean predicted prices in a table
        if predictions:
            prediction_summary = {
                "Statistic": ["Highest Predicted Price", "Lowest Predicted Price", "Mean Predicted Price"],
                "Value (£)": [
                    f"{predictions[max(predictions, key=predictions.get)]:,.2f}",
                    f"{predictions[min(predictions, key=predictions.get)]:,.2f}",
                    f"{np.mean(list(predictions.values())):,.2f}"
                ]
            }
            st.table(pd.DataFrame(prediction_summary))

            # Highlight points that meet the criteria
            df['Highlight'] = (
                (df['bedrooms'] == input_feature_2) &
                (df['propertyType'] == input_feature_3) &
                (df['bathrooms'] == input_feature_4) &
                (df['region'] == input_feature_5)  # Include region in the criteria
            )

            # Create the Plotly figure
            fig = px.scatter(
                df,
                x='year_sold',  # Use the extracted year for the x-axis
                y='display_price',
                color='Highlight',
                color_discrete_map={True: 'red', False: 'blue'},
                title="House Price vs Year",
                labels={'Highlight': 'Matches Criteria'},
                hover_data=['bedrooms', 'propertyType', 'bathrooms', 'region'],  # Include region in hover data
                trendline="ols"  # Add a linear regression trendline
            )

            # Add annotations for mean, low, and high predicted prices at the specified year
            annotation_offset = 50  # Offset to prevent overlapping annotations
            fig.add_annotation(
                x=input_feature_1,
                y=np.mean(list(predictions.values())),
                text=f"Mean: £{np.mean(list(predictions.values())):,.2f}",
                showarrow=True,
                arrowhead=2,
                ay=0,
                ax=+annotation_offset*2,
                bgcolor="yellow",
                font=dict(color="black")
            )
            fig.add_annotation(
                x=input_feature_1,
                y=predictions[max(predictions, key=predictions.get)],
                text=f"High: £{predictions[max(predictions, key=predictions.get)]:,.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-annotation_offset * 2,  # Adjust offset to avoid overlap
                bgcolor="green",
                font=dict(color="white")
            )
            fig.add_annotation(
                x=input_feature_1,
                y=predictions[min(predictions, key=predictions.get)],
                text=f"Low: £{predictions[min(predictions, key=predictions.get)]:,.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=annotation_offset,  # Adjust offset to avoid overlap
                bgcolor="red",
                font=dict(color="white")
            )

            # Add the figure to Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dataset not found. Please ensure the dataset is available at the specified path.")
        