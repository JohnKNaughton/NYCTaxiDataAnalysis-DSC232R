# NYC Taxi Data Set
Repo for UCSD DSC232R Group Project using NYC Taxi dataset
w/ Angela Nadine Junda and John

- Link to dataset: [The NYC Yellow Taxi dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Link to Jupyter notebook: [notebook](Taxi-Milestone3.ipynb)

# Group Project Part 3: Pre-Processing

# SDSC Expanse Setup
```
Time Limit: 02:00:00
Num Cores: 8
Memory Required Per Node: 64 GB
GPUs: 0
```
## Screenshot of Spark UI showing multiple executors active during data loading
See Jupyter Notebook.

# Fitting Analysis
- Where does your model fit in the fitting graph (underfitting vs. overfitting)?
  Our random forest model shows good generalization with Train AUC: 0.9682 and Test AUC: 0.9583. Since the performance remains high on the unseen test set, the model has successfully generalized without memorizing noise (overfitting). 
  
- Build at least one model with different hyperparameters and compare results
  We compared the baseline model against a Random Forest with tuned parameters of numTrees = 50 and maxDepth = 10, which resulted in an AUC: 0.9658, only a very slight improvement.

- Which model performs best and why?
  Hyperparameter tuning slightly improved AUC by better balancing model complexity and generalization, allowing the Random Forest to capture meaningful feature interactions while reducing overfitting. The Random Forest with a depth of 10 performed best because it was deep enough to capture interactions between payment types and trip distances, outperforming the shallower Random Forest. 
  
- What are the next models you are thinking of for Milestone 4 and why?
  We are considering evaluating Gradient‑Boosted Trees, which often outperform Random Forests on structured tabular data. XGBoost will likely be useful as we try to push the AUC even closer to 1. 

# Conclusion from First Model
- What is the conclusion of your 1st model?
  An AUC of 0.965 means our model can correctly rank a tipped trip above a non‑tipped trip about 96.5% of the time. That indicates very strong discriminative ability. The model separates the two classes very well, and the features selected contain strong predictive signal. This first model provides a high-performing baseline that proves that Random Forest can model the relationships effectively, and shows that the dataset itself contains some clear signals for classification. 

- What can be done to possibly improve it?
  Feature Refinement: Engineer additional interaction features (e.g., fare per mile, average speed). Implementing ratio based features, like fare/mile or tips as a percentage of fare, would normalize the data and help the model distinguish between standard rates and peak based rates. 
  Location-Based Feature Framework: We could incorporate Location-Based Features by mapping the PULocationID and DOLocationID to their respective NYC boroughs or neighborhoods. This may not be possible given the scope of this project.  
  Hyperparameter Cross-Validation: We could use CrossValidator in Spark to systematically test combinations of features to find the most robust tree configuration that prevents overfitting on outliers. 
  Temporal Generalization: Train on earlier years and test on later years to improve robustness to time‑dependent patterns.
  
- How did distributed computing help with this task?
  Distributed computing was essential for both data processing and model training due to the scale of the dataset. The dataset contained over 140 million records, making in‑memory processing infeasible. Spark enabled distributed ingestion, filtering, and feature engineering across multiple compute nodes. Random Forest training was parallelized across executors, allowing trees to be built simultaneously. Distributed computation reduced training time and allowed experimentation with multiple hyperparameter settings.

