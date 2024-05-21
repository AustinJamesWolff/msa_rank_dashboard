<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<!-- <div align="center">
  <a href="https://github.com/AustinJamesWolff/housing_supply_and_demand">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">MSA Ranking Dashboard</h3>

  <p align="center">
    Identify which markets have the most ideal demographic trends for investors.
    <br />
    <a href="https://msa-ranking.streamlit.app/"
    target="_blank"
    rel="noopener noreferrer">
    Access the dashboard!
    </a>
  </p>



<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
-->


<!-- ABOUT THE PROJECT -->
## About The Project

This repo is meant to help a real estate investor determine the best market to invest into multifamily properties, and therefore prioritizes datasets such as job growth, price growth, and rent growth.

### Built With

* Python
* Pandas
* GeoPandas
* Scikit-Learn
* Streamlit
* Bureau of Labor Statistics (BLS) API
* Zillow Research

### The Datasets Used

* Number of Jobs (BLS)
  * MSA level
  * Using the BLS API
  * NOTE: When using this repo, please make a local `.env` file and put your BLS_KEY in there in the form of `BLS_KEY=yourkey`
  * Before loading the app, please run `python download_bls_data.py` in the terminal to download the most recent month's data
* Price (Zillow Home Value Index)
  * MSA level
  * Using the Data Type "ZHVI All Homes (SFR, Condo/Co-op), Time Series, Smoothed, Seasonally Adjusted ($)"
  * Using the Geography "Metro & U.S."
  * A paid account with an MLS is required to access this data via API. Because this is a non-commercial project, a manual download is required [from here](https://www.zillow.com/research/data/).
* Rent (Zillow Observed Rent Index)
  * MSA level
  * Using the Data Type "ZORI (Smoothed, Seasonally Adjusted): All Homes Plus Multifamily Time Series ($)"
  * Using the Geography "Metro & U.S."
  * A paid account with an MLS is required to access this data via API. Because this is a non-commercial project, a manual download is required [from here](https://www.zillow.com/research/data/).


### Visualizations Created

* Dashboard ranking MSAs based on your criteria.
* Job, Price, and Rent Growth Graphs.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/AustinJamesWolff/housing_supply_and_demand.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- USAGE EXAMPLES
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- ROADMAP
## Roadmap

- [] Feature 1
- [] Feature 2
- [] Feature 3
    - [] Nested Feature

See the [open issues](https://github.com/AustinJamesWolff/housing_supply_and_demand/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>
-->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Austin Wolff - austinwolff1997@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[contributors-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[forks-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/network/members
[stars-shield]: https://img.shields.io/github/stars/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[stars-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/stargazers
[issues-shield]: https://img.shields.io/github/issues/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[issues-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/issues
[license-shield]: https://img.shields.io/github/license/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[license-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
