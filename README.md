# Paper title and data availability
This repository contains codes for the paper entitled "Machine learning approaches to predict the photocatalytic performance of bismuth ferrite-based materials in the removal of malachite green" authored by Zeeshan Haider Jaffari, Ather Abbas, Sze-Mun Lam, Sanghun Park, Kangmin Chon, Sun-Sik Kim and Kyung Hwa Cho, also known as the WEIL group at UNIST, Korea.
File name "Raw data" contains the data set of 1200 data points used in this study to build this model. Besides, The published article can be found on https://www.sciencedirect.com/science/article/pii/S0304389422018258

# Introduction
In the Sustainable Development Goals (SDGs) agenda for 2030, one of the 16 goals is devoted to sanitation and water. This goal has been subdivided into eight targets, one of which is dedicated to effective wastewater management. The SDG report states that it requires the following four vital actions: decline in contamination at the source, contaminant treatment, recycling of treated wastewater, and recovery of by-products. The issue of water pollution in the surface and groundwater has recently attracted increasing attention. This study compare and predict the photodegradation efficiency of malachite green dye from wastewater using various regression trees and neural networks algorithms with a large detaset (1200 data point) and ten comprehensive input variables. The accuracy of the ML models were substentially improved using hyperparameter optimization. Interdependence among various input paramters and thier influences on malachite green degration was also investigated. Finally, numrous post-processing techniques, such as permutatio importance analysis and partial dependence plot (PDP) were utilized to analyze the behaviour of the developed ML model for the malachite green dye removal efficiency. 
# Files in this Repository 
File named "ann" was used to build the artifical neural networks. In order to obatain the code, please open the file. 
file named "pdp" discription: partial denpendence analysis (PDP) was used to investigate the depencence of output variable (removal efficiency) on each of the input variables. 
Deep learning models: in this study, two deep learning models named ANN and LightLSTM were used. the codes, weights and other relevent files were available in their respective .py files. 
process parameters optimizations were investigated using heatmap technique. the required code and related materials are available in the heatmap.py file.
various post processing tools such as permutation importance, PDP, resgression plots were also studied. the files named for permutation importance, PDP and resgression plots are perm_imp.py, pdp.py and reg_plots.py, respectively. All the required details including environments, weights are included in the codes files. 
Two categorical input features, the catalyst type and anions, were converted to numerical values using the one-hot encoding method. 
Each of the numerical input features was rescaled between 0 to 1 before feeding it to neural network. This step was carried out so that the variables which are measured at different scales contribute equally to model training.
Besides, the feature importance and contribution factors to any certain data point can also be checked using SHAP analysis. the code is avaiable in the file named "with_SHAP.py"

# Findings
The ten input variables selected were the catalyst type, reaction time, light intensity, initial concentration, catalyst loading, solution pH, humic acid concentration, anions, surface area, and pore volume of various photocatalysts. The MG dye degradation efficiency was selected as the output variable. An evaluation of the performance metrics suggested that the CatBoost model, with the highest test coefficient of determination (0.99) and lowest mean absolute error (0.64) and root-mean-square error (1.34), outperformed all other models. The CatBoost model showed that the photocatalytic reaction conditions were more important than the material properties. The modelling results suggested that the optimized process conditions were a light intensity of 105 W, catalyst loading of 1.5 g/L, initial MG dye concentration of 5 mg/L and solution pH of 7. 
# Correspondance 
If you feel any dificulties in executing these codes, please contact us through email on jaffarizh@hotmail.com or ather_abbas786@yahoo.com. Thank you
