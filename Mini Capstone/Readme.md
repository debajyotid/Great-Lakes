### Determining the preparedness of the German Government and Medical Authorities in handling COVID19 Crisis
---
![Image](https://taj-strategie.fr/content/uploads/2020/03/germany-coronavirus.png)

Purpose:
---
In this paper, we try and analyse the trends in the spread of COVID-19 in the German population. 
We look into the statistical significance of the lockdown enforced by the German authorities in curbing the spread of the virus across the German population.
We also try to develop a forecasting model to predict the number of infected patients & fatalities arising out of the infection. 

Materials & Methods:
---
As part of this study, we obtained the dataset from 2 sources, namely Kaggle & Institute of Health Metrics & Evaluation (IHME). 
The datasets had data like number of cases, age-group of patient, number of available beds, number of tests done on a given day across the 2 datasets & the datasets were merged based different criterion like State, County, Date of Infection,etc. 
The dataset had data from 24-Jan-2020 up till 3-June-2020, and the same was obtained from across the 16 German states & their individual counties. 

Results:
---
Statistical methods like t-Test & ANOVA resulted in extremely low p-values, based on which we can say with 99% confidence level that the different measures undertaken by German authorities, especially the imposition of a nationwide lockdown, had a statistically significant impact on controlling the spread of the SARS-CoV-2 virus in Germany. 
Visually we saw that the SARS-CoV-2 virus spread rose significantly in the pre-lockdown period (which we have assumed to be till 07-April-2020) and then started slowly tapering off. 
This also helped avoid severe stress on the medical institutions & we once again saw visually that the hospital admission rate was always below the rate at which new hospital beds were being added daily. 
This was also vindicated in our time-series forecasting model using Prophet, an open-source forecasting library from Facebook. 

Conclusion:
---
Based on the above, we could statistically prove that the timely measures and steps taken by German authorities, like imposition of the lockdown, helped in both controlling the spread of COVID-19 across Germany, as well as kept the rate of fatalities due to the same at a relatively low rate.
