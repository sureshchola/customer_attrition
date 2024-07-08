

Attrition Model Documentation


Summary

To investigate the underlying causes of customer attrition for its diabetes pump device and to build a predictive model around the consumption of device consumables within a 6-month dtime span since a customer’s last purchase. To meet this goal, multiple sources of business data were extracted and transformed into features for a Bayesian model which proved successful in identifying causal factors for attrition as well as having a high performance on identifying likely attritioners. With the conclusion of the POC, Medtronic wanted to expand upon this work by productionizing the attrition model on AWS and creating an MLOPs framework to productionize future models created by its internal data science team. This entailed a thirteen week effort to gather new requirements, build out the cloud infrastructure, and test an end to end pipeline to operationalize this vision. Among the new requirements included utilizing Sagemaker for training and serving models, enabling the reading and writing of data to and from Snowflake, adapting the model to the XGBoost framework, and incorporating CICD tools such as Gitlab and Artifactory to automate deployment as much as possible. 

