import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 1e-3

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

    x = Concatenate()([pov_parsing, state_input])
    x = Dense(128, activation='relu')(x)
    output = Dense(64, activation='relu')(x)
    model = Model(inputs=[pov_input, state_input], outputs=output)
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model



