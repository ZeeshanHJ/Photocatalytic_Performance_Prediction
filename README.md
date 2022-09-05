# NM_BiFeO3_Modelling_paper
Machine learning approaches to predict the photocatalytic performance of bismuth ferrite-based materials in the removal of malachite green
File name "Raw data" contains the data set of 1200 data points used in this study to build this model.
# Project Introduction
This study compare and predict the photodegradation efficiency of malachite green dye using various regression trees and neural networks algorithms with a large detaset (1200 data point) and ten comprehensive input variables. The accuracy of the ML models were substentially improved using hyperparameter optimization. Interdependence among various input paramters and thier influences on malachite green degration was also investigated. Finally, numrous post-processing techniques, such as permutatio importance analysis and partial dependence plot (PDP) were utilized to analyze the behaviour of the developed ML model for the malachite green dye removal efficiency. 
# Model building and their discription 
file named "ann" was used to build the artifical neural networks 

# Results
The ten input variables selected were the catalyst type, reaction time, light intensity, initial concentration, catalyst loading, solution pH, humic acid concentration, anions, surface area, and pore volume of various photocatalysts. The MG dye degradation efficiency was selected as the output variable. An evaluation of the performance metrics suggested that the CatBoost model, with the highest test coefficient of determination (0.99) and lowest mean absolute error (0.64) and root-mean-square error (1.34), outperformed all other models. The CatBoost model showed that the photocatalytic reaction conditions were more important than the material properties. The modelling results suggested that the optimized process conditions were a light intensity of 105 W, catalyst loading of 1.5 g/L, initial MG dye concentration of 5 mg/L and solution pH of 7. 
