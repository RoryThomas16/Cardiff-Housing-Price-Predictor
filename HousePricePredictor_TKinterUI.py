import numpy as np
import tkinter as tk
from tkinter import messagebox
import pickle
import os

# Create a DataFrame with columns matching the scaler's feature names (if you have them)
column_names = ['year_sold', 'bedrooms', 'propertyType', 'bathrooms']


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
        return None


# Initialize the model and encoders
models = {}
current_directory = os.getcwd()
model_directory = os.path.join(current_directory, 'models')

for idx, file in enumerate(list_files_in_directory(model_directory)):
    with open(os.path.join(model_directory, file), 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        if 'propTypeEncoder' in file:
            propTypeEncoder = content
        elif 'scaler' in file:
            scaler = content
        elif 'poly' in file:
            poly = content
        else:
            name = file.split('.')[0].split('_')[-1]
            models[name] = content

# Set up the Tkinter GUI
root = tk.Tk()
root.title("Machine Learning Model Prediction")

# Define the function to make predictions
def make_prediction():
    try:
        # Get input values from the user
        year_sold = int(entry_year.get())
        bedrooms = int(entry_bedrooms.get())
        property_type = combo_property_type.get()
        bathrooms = int(entry_bathrooms.get())
        # Prepare the input data for the model
        input_data = np.array([year_sold, bedrooms, property_type, bathrooms])
        # Transform the property type
        input_data[2] = propTypeEncoder.transform(input_data[2].reshape(-1,1))[0]
        # Apply the scaler on the DataFrame
        input_data = np.array(input_data).reshape(1,-1)
        input_data_scaled = scaler.transform(input_data)
        # Make predictions for each model
        predictions = []
        for name, model in models.items():
            if name == 'ridge':
                temp_input_data_scaled = poly.transform(input_data_scaled.copy())
                prediction = model.predict(temp_input_data_scaled)
            else:
                prediction = model.predict(input_data_scaled)
            predictions.append(f"{name}: Prediction = £{prediction[0]:.2f}")
        # Display the predictions in the text box
        result_text.delete(1.0, tk.END)  # Clear previous results
        for prediction in predictions:
            result_text.insert(tk.END, prediction + "\n")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create input fields for the features
label_year = tk.Label(root, text="Year Sold")
label_year.grid(row=0, column=0, padx=10, pady=10)

entry_year = tk.Entry(root)
entry_year.grid(row=0, column=1, padx=10, pady=10)
entry_year.insert(0, "2025")  # default value

label_bedrooms = tk.Label(root, text="Bedrooms")
label_bedrooms.grid(row=1, column=0, padx=10, pady=10)

entry_bedrooms = tk.Entry(root)
entry_bedrooms.grid(row=1, column=1, padx=10, pady=10)
entry_bedrooms.insert(0, "1")  # default value

label_property_type = tk.Label(root, text="Property Type")
label_property_type.grid(row=2, column=0, padx=10, pady=10)

combo_property_type = tk.StringVar()
property_types = ['TERRACED', 'SEMI_DETACHED', 'DETACHED', 'FLAT']
combo_property_type.set(property_types[0])  # default value

property_type_menu = tk.OptionMenu(root, combo_property_type, *property_types)
property_type_menu.grid(row=2, column=1, padx=10, pady=10)

label_bathrooms = tk.Label(root, text="Bathrooms")
label_bathrooms.grid(row=3, column=0, padx=10, pady=10)

entry_bathrooms = tk.Entry(root)
entry_bathrooms.grid(row=3, column=1, padx=10, pady=10)
entry_bathrooms.insert(0, "1")  # default value

# Create button to trigger prediction
button_predict = tk.Button(root, text="Make Prediction", command=make_prediction)
button_predict.grid(row=4, column=0, columnspan=2, pady=20)

# Text box to display the predictions
result_text = tk.Text(root, height=10, width=50)
result_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
