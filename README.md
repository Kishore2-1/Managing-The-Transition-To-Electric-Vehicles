# Managing the Transition to Electric Vehicles

This project focuses on optimizing the transition from Internal Combustion Engine (ICE) vehicles to Electric Vehicles (EVs) by determining the optimal placement of EV charging stations. The case study targets Southampton, UK, combining operational research techniques with machine learning methods for enhanced decision-making. The repository contains the report, scripts, and data files used for the analysis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [File Structure](#file-structure)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview
The aim of this project is to find the optimal locations for EV charging stations in Southampton. It uses data collected from 24 petrol stations and applies clustering algorithms and facility location optimization models to propose strategic locations for EV infrastructure. The model balances coverage, cost efficiency, and accessibility while ensuring a gradual transition from ICE vehicles to EVs.

### Key Features
- **Clustering Techniques:** K-Means and Hierarchical Clustering are used to group petrol stations.
- **Facility Location Optimization:** A mathematical model is developed to optimize the placement of EV charging stations to maximize coverage and minimize costs.
- **Sensitivity Analysis:** Evaluates the impact of different distance thresholds for strategic decision-making.
  
## Technologies Used
- **Python 3.x**
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - Folium (for interactive maps)
  - Geopy (for geodesic distance calculations)
  - PuLP (for optimization)
  
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Kishore2-1/Managing-The-Transition-To-Electric-Vehicles.git
   ```
2. Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt

   ```

3. Ensure Python 3.x is installed on your machine.

## File Structure
```
├── Report.pdf               # Full case study report
├── Script.py                # Python script containing the code for data analysis, clustering, and optimization
├── input_files/             # Directory containing any input datasets used (e.g., petrol station locations, distances)
└── README.md                # This file
```
4. Install the required Python libraries:
   ```bash
   pip install pandas
   pip install seaborn
   pip install matplotlib
   pip install windrose
   pip install glob
   pip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn
   pip install --upgrade tensorflow
   pip install --upgrade scikit-learn
   pip install scikeras
   ```
5. Run the Jupyter notebook or Python script to perform the analysis and generate the models.

## Usage
1. **Run the Python Script:**
   The main script `Script.py` contains all the necessary code to perform the analysis. It includes distance calculations, clustering, optimization, and sensitivity analysis.
   Run the script:
   ```bash
   python Script.py
   ```

2. **Visualization:**
   The script generates visual outputs, including cluster maps and sensitivity analysis plots. Interactive maps displaying the locations of petrol stations and clusters can be viewed using the Folium library.

3. **Customization:**
   You can adjust the distance threshold for sensitivity analysis in the script to explore different scenarios and configurations.

## Results
The analysis shows that maintaining 9 strategically located petrol stations within a 1.5 km distance covers all 15 districts of Southampton. This strikes a balance between minimizing the number of stations and maximizing coverage, allowing for a smooth transition to EV infrastructure.

## Contributing
If you’d like to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation of the changes.

---

This comprehensive analysis and its findings have been documented and made available on GitHub, providing a valuable resource for researchers, policymakers, and practitioners in the field of urban mobility and bike-sharing systems. The repository includes detailed code, data visualizations, and results, promoting transparency and facilitating further research and collaboration.

Feel free to explore the project on GitHub [here](https://github.com/Kishore2-1/Managing-The-Transition-To-Electric-Vehicles).


