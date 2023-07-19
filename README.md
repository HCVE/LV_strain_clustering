# Feature-based clustering of left ventricular strain curve for risk stratification in general population. 

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
scikit-learn==1.3.0 
scipy==1.11.1  
matplotlib==3.7.2 
pandas==2.0.3
