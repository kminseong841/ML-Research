# ML-Research

This GitHub project aims to implement a modified version of the paper “SPATIOTEMPORAL DEEP LEARNING MODEL FOR CITYWIDE AIR POLLUTION INTERPOLATION AND PREDICTION.”

Pytorch_Ozone_predict_CNN.ipynb -> Current Main file

Folder
- Module : used in Main file
  - Model
  - Visualization
  - Pre_proecess
 
Description

1) Data Extraction
Input data: Meteorological Data obtained from the Korea Meteorological Administration’s Data Portal
Label data: Seoul’s hourly average air quality information acquired from the Seoul Open Data Plaza

2) Data Preprocessing
Missing value handling: (x, y) adjacency interpolation, Time adjacency interpolation
Standardization: Compare performance when using Quantile Transform, MinMax scaling, BoxCox Transform, and others
Mapping: Implement preprocessing functions to convert a DataFrame into image data (DataFrame → img Data Mapping)

3) Model Architecture
Conv3D: Predict the image at the next time step using three sequential time-step images
ConvLSTM: Predict the image at the next time step using three time sequences along with forget, input, and output gates
BitMasking: Prevent artificial reduction of loss in regions outside of measurement stations during backpropagation

4) Result & Visualization
Test MAE: 0.0009049615166800958 (Conv3D), 0.0010369176743552089 (ConvLSTM)
Achieved a model with the above prediction performance
Comparisons
Compare Ground Truth and Predictions on the Seoul Ozone Map at a specific time point
Compare Ground Truth and Predictions for a specific measurement station over time (t, Ozone_Concentration(t))
