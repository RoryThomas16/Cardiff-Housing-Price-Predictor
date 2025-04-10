# Machine Learning Practice | Cardiff Housing Price Predictor

<!-- Credit to: https://github.com/othneildrew/Best-README-Template/blob/main/README.md for the Template <3 -->
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="application">Application</a>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Application
You can access the application here from the following link:

https://rorythomas16-cardiff-hou-housepricepredictor-streamlitui-l5hul5.streamlit.app/

The application has been developed in Streamlit as it is suited for handling data science projects in a great web-based UI. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## About The Project

The project is practice for me, this is for me to practice more my regressors but also doesn't hurt to keep trying and practicing the same core concepts. I also want to try my hand at some neural networks and this seems a friendly project to broach this on.

- The Web Scraping is basically sifting through historical house price data and putting this into a useable format.
- The Machine Learning is basically taking the house price data (along with # of Bedrooms, # of Bathrooms, Property Type (e.g., terraced, semi-detached), and Date) and predicts what the house price will be in 'N' years.
- A Streamlit UI is provided to allow the user to input the data and get a prediction, with an offline iteration using TKinter also available.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
The code is predominantly built in Python [= 3.11.11]; primarily Jupyter Notebooks.

**Dataset_Snapshot.py** has been written with the following libraries:
* requests
* BeautifulSoup
* json

The **Notebook** has been run with the following libraries:
* Matplotlib
* numpy
* pandas
* plotly
* kaleido
* sklearn
* imblearn
* pyTorch
* Torchviz
* pickle

The **Streamlit UI** has been run with the following libraries:
* os
* streamilt
* numpy
* pickle
* pandas

**(DEPRCIATED)** - The TKinter UI has not been maintained and may not work on modern releases.
The **TKinter UI** has been run with the following libraries:
* os
* tkinter
* numpy
* pickle

<p align="right">(<a href="#readme-top">back to top</a>)</p>

