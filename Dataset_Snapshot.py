import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime

class RightMoveScraper:
    def __init__(self, num_pages=10):
        """Initialize the scraper with the number of pages to scrape."""
        self.num_pages = num_pages
        self.urls = []
        self.df = pd.DataFrame()

    def generate_urls(self):
        """Generate the list of URLs to scrape."""
        for idx in range(1, self.num_pages + 1):
            url = f"https://www.rightmove.co.uk/house-prices/roath.html?pageNumber={idx}"
            self.urls.append(url)

    def scrape_data(self):
        """Scrape data from the generated URLs and store it in a DataFrame."""
        for url in self.urls:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            content = json.loads(soup.find_all("script")[-4].text.split(" = ")[1].split(";")[0])
            content.pop('isAuthenticated', None)
            content.pop('metadata', None)
            content = content['searchResult']['properties']

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
            self.df = pd.concat([self.df, new_df])

    def clean_data(self):
        """Perform basic cleanup on the DataFrame."""
        print(f"Number of Entries 'as read from RightMove: {len(self.df)}")
        self.df.dropna(axis=0, inplace=True)
        print(f"Number of Entries after dropping missing data: {len(self.df)}")
        self.df.reset_index(inplace=True, drop=True)

    def save_to_csv(self):
        """Save the DataFrame to a CSV file with a timestamp in the file name."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"rightmove_housing_data_{timestamp}.csv"
        self.df.to_csv(output_file, index=False)
        print(f"DataFrame written to {output_file}")

    def run(self):
        """Run the entire scraping process."""
        self.generate_urls()
        self.scrape_data()
        self.clean_data()
        self.save_to_csv()

if __name__ == "__main__":
    # Create an instance of the scraper
    scraper = RightMoveScraper(num_pages=5)  # Adjust the number of pages as needed

    # Run the scraper
    scraper.run()