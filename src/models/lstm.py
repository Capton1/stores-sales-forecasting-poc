import pickle
from typing import Tuple

import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential

from .training_helpers import get_timeseries_generator
from keras.preprocessing.sequence import TimeseriesGenerator
from datetime import date

MODEL_PATH = 'models'

def lstm(train=False, load_model=False, generator: TimeseriesGenerator = None, model_name=f'lstm {date.today().strftime('%Y-%m-%d-%H-%M-%S')}'):
    """
    LSTM model for training or loading a pre-trained model.

    Args:
        train (bool, optional): Flag indicating whether to train the model. Defaults to False.
        load_model (bool, optional): Flag indicating whether to load a pre-trained model. Defaults to False.
        generator (TimeseriesGenerator, optional): The generator used for training the model. Required if train=True. Defaults to None.
        model_name (str, optional): The name of the model. Defaults to 'lstm {current_date_time}'.

    Raises:
        ValueError: Raised when neither train nor load_model is True.
        ValueError: Raised when train is True but generator is not provided.

    Returns:
        The trained or loaded LSTM model.
    """
    if not train and not load_model:
        raise ValueError('You must either train or load a model')
    
    if train and not load_model and generator is None:
        raise ValueError('You must provide a generator if you want to train a model')

    if load_model:
        pickled_model = pickle.load(open(f'{MODEL_PATH}/{model_name}.h5', 'rb')) 
    else:
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(generator.length, generator.data.shape[1]), return_sequences=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=["accuracy"])

        model.fit(generator, epochs=5, batch_size=512)

        pickle.dump(model, open(f'{MODEL_PATH}/{model_name}.h5', 'wb'))
        pickled_model = model

    return pickled_model
