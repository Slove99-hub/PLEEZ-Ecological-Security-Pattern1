# PLEEZ-Ecological-Security-Pattern1
Data and code for ecological security pattern construction in the Poyang Lake Eco-Economic Zone.

 Ecological Security Pattern Construction in the Poyang Lake Eco-Economic Zone

 Overview

This repository contains the datasets and Python scripts supporting the construction and analysis of ecological security patterns (ESPs) in the Poyang Lake Eco-Economic Zone (PLEEZ), China, for 2000, 2010, 2020, and 2023.

The study integrates ecosystem service assessment, Zonation, morphological spatial pattern analysis (MSPA), XGBoost, and circuit theory to identify ecological sources, characterize ecological resistance, and analyze ecological connectivity.

 Repository Contents

- Code/: Python scripts for XGBoost sample extraction, model training, and resistance surface prediction.
- Samples/: XGBoost training datasets.
- xgb/: Ecological resistance surfaces for the four study years.
- Sources/: Ecological source layers for the four study years.

 Data Sources

The original geospatial datasets were obtained from publicly accessible databases and documented data providers. Detailed information on data sources and spatial resolutions is provided in Table 1 of the associated manuscript.

 XGBoost Model

Seven environmental predictors were used: elevation, slope, distance to roads, distance to rivers, NDVI, population density, and nighttime light.

For each study year, 20,000 samples were selected from the low- and high-resistance classes, with 10,000 samples per class. The data were divided into training (70%) and testing (30%) subsets using stratified random sampling.

The model configuration and preprocessing settings are documented in Supplementary Materials, Section S6.2 and Table S9.

 Code Usage

The XGBoost modeling workflow consists of three steps:

1. `Sample_Extraction.py` — Extract training samples from raster datasets.
2. `XGBoost_Training.py` — Train and evaluate the XGBoost model.
3. `Resistance_Prediction.py` — Predict spatial ecological resistance.

 Data and Code Availability

The repository provides research datasets and Python scripts to support transparency and reproducibility. The original datasets remain subject to the access and reuse conditions of their respective providers.
