# What factors influence grocery sales in Ecuador?

## Data-centric, Model-centric and Shap Value

In this article, I explore the factors influencing grocery sales in Ecuador by building a Forecasting model with both data-centric and model-centric approch. Then, I will use SHAP values to understand the model output.

In the rest of the article, I will:

- Load and process the original data
- Detail the model development steps
- Analyze the insights drawn from the SHAP values


### Data

The dataset is publicly available and can be downloaded from the following link: [stores-sales-time-series-forcasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

#### Read and merge the original data

![orignal_df](./pictures/original_df.png)

This dataframe provides the number of sales per day of a product family in a given store : `date`, `sales`, `store_nbr`, `family`

We also have these following features :

- `onpromotion` : the number of product in promotion in each store every day
- `dcoilwtico` : oil price per day
- `typedays`, `locale`, `locale_name`, `description`: informations about the day (Is it holidays ? Does a particular event happened this day?)
- `transfered` : A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government
- `city`, `state`, `typestores`, `cluster` : informations about the stores

#### Process the data

We apply a series of operations to our original dataframe :

- Drop possible duplicates
- Interpolate the `dcoilwtico` column due to missing values
- Fix the column `typedays` with the informations provided in the `transfered` column
- Group data by date in order to simplify the modeling phase. As a result, we have the number of sales for each day.
- Remove useless columns which cannot be used as features for the model, ex: `description`, `typestores` (as all stores are now aggregated)
- Normalize data using MinMaxScaler. It ensures that the values are within a fixed range and contributes equally to the analysis.
- Input missing dates
- Use One Hot Encoding to the `typedays` column in order to have usable features for the modeling phase

As a result, we got the following dataframe :

![processed_df](./pictures/processed_df.png)

### Sales Forecasting Modeling

#### XGBoost Regressor

XGBoost regressor, in a nutshell, is like a powerful decision tree with multiple layers - imagine stacked trees! It combines the strengths of many weak learners (smaller trees) to create a stronger, more accurate prediction model for continuous values. 

Here's how it works:

1. **Starts with a simple tree :** It builds a basic decision tree based on data, dividing it into groups based on best feature splits.
2. **Adds another tree :** It analyzes the errors made by the first tree and adds another tree focused on improving those predictions.
3. **Keeps learning :** It repeats this process iteratively, adding new trees each time, focusing on correcting previous mistakes and becoming more accurate with each addition.

![xgboost](./pictures/xgboost.png)


- xgboost
- split train / test
- entrainement du modèle (cross val, metric d'eval, etc)
- validation du meilleur modèle avec le jeu de test

#### Modele de base XGBOOST

![xgboost_default](pictures/xgboost_default_wo_dates_features.png)

#### Amélioration grace à la data explo

création de features, détection d'anomalies, etc.

#### Fine-tuning 

Opti bayesienne

### Explicabilité du modèle (SHAP)

Conclusion qu'on peut tirer des différentes viz + amélioration du modèle grâce aux apprentissages