# NYC Taxi Data Set
Repo for UCSD DSC232R Group Project using NYC Taxi dataset
w/ Angela Nadine Junda and John

- Link to dataset: [The NYC Yellow Taxi dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Link to Jupyter notebook: [notebook](Taxi-Milestone2.ipynb)

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
  Our random forest model shows good generalization with Train AUC: 0.9682 and Test AUC: 0.9583.
  
- Build at least one model with different hyperparameters and compare results
  Random Forest with tuned parameters resulted in an AUC: 0.9658, only very slight improvement.

- Which model performs best and why?
  Hyperparameter tuning slightly improved AUC by better balancing model complexity and generalization, allowing the Random Forest to capture meaningful feature interactions while reducing overfitting.
  
- What are the next models you are thinking of for Milestone 4 and why?
  We are considering evaluating Gradient‑Boosted Trees, which often outperform Random Forests on structured tabular data.

# Conclusion from First Model
- What is the conclusion of your 1st model?
  An AUC of 0.965 means your model can correctly rank a tipped trip above a non‑tipped trip about 96.5% of the time. That indicates very strong discriminative ability. The model separates the two classes very well. The features selected contain strong predictive signal. Random Forest is modeling the relationships effectively.

- What can be done to possibly improve it?
  Feature Refinement: Engineer additional interaction features (e.g., fare per mile, average speed).
  Temporal Generalization: Train on earlier years and test on later years to improve robustness to time‑dependent patterns.
  
- How did distributed computing help with this task?
  Distributed computing was essential for both data processing and model training due to the scale of the dataset. The dataset contained over 140 million records, making in‑memory processing infeasible. Spark enabled distributed ingestion, filtering, and feature engineering across multiple compute nodes. Random Forest training was parallelized across executors, allowing trees to be built simultaneously. Distributed computation reduced training time and allowed experimentation with multiple hyperparameter settings.

