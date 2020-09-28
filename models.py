import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

def make_diamond_miner_model(pov_shape: tuple, state_shape: tuple) -> Model:
    assert isinstance(pov_shape, tuple)
    assert isinstance(state_shape, tuple)

    pov_input = Input(name="pov", shape=pov_shape)
    x = Lambda(lambda x: x * (1/255))(pov_input)
    x = Conv2D(filters=1, kernel_size=(3, 3))(x)
    x = MaxPool2D()(x)
    x = Conv2D(filters=1, kernel_size=(3, 3))(x)
    x = MaxPool2D()(x)
    pov_parsing = Flatten()(x)

    state_input = Input(name="state", shape=state_shape)
    state_parsing = Dense(64, kernel_initializer='random_normal')(state_input)

    x = Concatenate()([pov_parsing, state_parsing])
    x = Dense(512, kernel_initializer='random_normal')(x)
    # x = Dense(128, kernel_initializer='random_normal')(x)
    # x = Dense(128, kernel_initializer='random_normal')(x)
    output = Dense(64, activation='linear', kernel_initializer='random_normal')(x)
    return Model(inputs=[pov_input, state_input], outputs=output)
