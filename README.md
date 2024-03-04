# stores-sales-forecasting-poc



## Getting started


Install dependencies :

```sh
$ poetry install
```

Download dataset :

```sh
$ kaggle competitions download -c store-sales-time-series-forecasting
$ mkdir data; mv store-sales-time-series-forecasting.zip data
$ unzip store-sales-time-series-forecasting.zip
```

Dataset relevant columns (X) :

- onpromotion : # of products in promo in the store ~ (scaled)
- dcoilwtico : gas price ~ (scaled) -> Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.
- cluster : # of similar stores in the area ~ (scaled)
- Holiday, Normal, Event, Additional : type of day ~ (one hot encode)
- Automotive, seafood, beauty, etc : type of products sold ~ (one hot encode)
- A, B, C, D, E : type of store (supermarket, etc) ~ (one hot encode)

Target (y) :
    - sales : gives the total sales for a product family at a particular store at a given date.

TODO :
    - Explicabilité des modèles de time series :
        - Shap value
        - Lime
        - Partial distribution
