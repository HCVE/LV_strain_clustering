# Feature-based clustering of left ventricular strain curve for risk stratification in general population. 

# Abstract
Identifying patients in subclinical stages of cardiovascular (CV) diseases is crucial for their close monitoring and for adopting cost-effective management strategies. In current practice, only the peak left ventricular (LV) strain is used to predict the incidence of adverse events, neglecting in this way the temporal information hidden in all phases of cardiac cycle. Therefore, in this study we employed unsupervised machine learning methods on time-series-derived features from LV strain to identify distinct clinical phenogroups associated with the risk of developing adverse events in the general population. 

We prospectively studied 1185 community-dwelling individuals (mean age, 53.2 years; 51.3% women), in whom we acquired clinical and echocardiographic data including LV strain traces at baseline and collected adverse events (CV: n=116; cardiac: n=87) on average 8.7 years later. A Gaussian Mixture Model (GMM) was applied on features derived from LV time series strain curves, including slope during systole and early diastole, peak strain, slope during late diastole and the duration and height of diastasis. To assess the clinical importance of the derived clusters we compared the clinical characteristics, CV and cardiac adverse outcome. 

Based on BIC score we identified that the optimal number of clusters was four. In the first two clusters we observed differences in heart rate and age distributions, but they revealed similar low risk profiles. Cluster 4 had the worst CV risk factors combination, and higher prevalence of LV remodeling and diastolic dysfunction (i.e. lowest e’ velocity and highest E/e’) compared to other clusters. Kaplan–Meier showed an increased cumulative incidence risk for all CV and cardiac events compared to the average population. Clusters 3 and 4 had the highest CV (cluster 3: HR 1.28; P=0.038, cluster 4: HR 1.20;P=0.034) and cardiac (cluster 3: HR 1.57; P=0.024, cluster 4: HR 1.43;P=0.010) risk. On the other hand, the risk of adverse events did not reach the significant level in subjects with an abnormal LV peak strain (<17%) (HR 1.12; P=0.66). Finally, using SHAP values and Random Forest we observed that from the selected features, those that incorporate temporal information such as the slope during systole and diastole were considered more important. 

Employing a Gaussian Mixture Model on features derived from time series LV strain curves we identified clinically meaningful clusters which could provide important prognostic information over the peak LV strain. 

  ![alt text](https://github.com/HCVE/LV_strain_clustering/blob/main/Images/LV%20Computational%20Pipeline.png?raw=true) 

# Model Configuration
n_components = 4 (number of clusters)  
n_init = 30  
random_state = 0  
covariance_type = "diag"  

# Training and Validation on FLEMENGHO Cohort
SI_FLEMENGHO_TRAINING.ipynb

# External Validation on EPOGH Cohort
SI_EPOGH_TRAINING.ipynb

### To install the required packages run "pip install -r requirements.txt". 

Some of the major libraries are:   
scikit-learn==1.2.2  
scipy==1.10.1  
matplotlib==3.7.1  
pandas==1.5.3 
