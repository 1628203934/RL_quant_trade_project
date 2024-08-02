from model_libraries import models, layers, optimizers


def create_dqn_model(input_shape, action_space):
    model = models.Sequential()
    model.add(layers.Dense(24, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model
