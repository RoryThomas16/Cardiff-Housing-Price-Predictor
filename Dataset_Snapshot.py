import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime

num_pages = 10 # Number of pages to look at - change as needed
urls = []
df = pd.DataFrame()

for idx in range(1,num_pages+1):  # Create list of urls corresponding pages we are going to scrape.
    url = "https://www.rightmove.co.uk/house-prices/roath.html?pageNumber=" + str(idx)
    urls.append(url)
    
for url in urls:
    r = requests.get(url)
    # Parsing the HTML
    soup = BeautifulSoup(r.text, 'html.parser')
    content = json.loads(soup.find_all("script")[-4].text.split(" = ")[1].split(";")[0])  # Convert content of page to a dictionary
    content.pop('isAuthenticated', None)
    content.pop('metadata', None)
    content = content['searchResult']['properties'] # Access only the property data

    data = []
    for property in content:
        address = property.get('address')
        propertyType = property.get('propertyType')
        bedrooms = property.get('bedrooms')
        bathrooms = property.get('bathrooms')
        location = property.get('location')
        latitude = location['lat']
        longitude = location['lng']
        transactions = property.get('transactions')
        for transaction in transactions:
            display_price = transaction['displayPrice']
            date_sold = transaction['dateSold']
            tenure = transaction['tenure']
            new_build = transaction['newBuild']
            data.append({
                'address': address,
                'propertyType': propertyType,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'latitude': latitude,
                'longitude': longitude,
                'display_price': display_price,
                'date_sold': date_sold
                
            })
    new_df = pd.DataFrame(data)
    df = pd.concat([df,new_df])

# Some basic clean up
print(f"Number of Entries 'as read from RightMove: {len(df)}")
df.dropna(axis=0, inplace=True)# drop empty rows with missing data, it's not worth trying to preserve these.
print(f"Number of Entries 'as read from RightMove: {len(df)}")
df.reset_index(inplace=True)
df.drop(columns='index', inplace=True)

# Generate a timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Write the DataFrame to a CSV file with the timestamp in the file name
output_file = f"rightmove_housing_data_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f"DataFrame written to {output_file}")