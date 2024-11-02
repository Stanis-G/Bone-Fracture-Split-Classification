from PIL import Image
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Rescaling, Input, Dropout, BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import SGD


def read_img(filepath, as_array=False, black_white=False):
    """Read image, turn it to black and white scale if needed. The result can be returned as img or array"""
    img = Image.open(filepath)
    if black_white:
        img = img.convert('L')
    if as_array:
        return np.asarray(img)
    return img


def set_model(lr, n, p, pad_shape, seed=0):

    model = Sequential([
        Input(shape=(pad_shape[0], pad_shape[1], 1)),
        Conv2D(
            20,
            (3,3),
            padding='valid',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform(seed=seed),
        ),
        MaxPool2D(
            (2,2),
            strides=1,
            padding='valid',
        ),
        Conv2D(
            20,
            (3,3),
            padding='same',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform(seed=seed),
        ),
        MaxPool2D(
            (2,2),
            strides=1,
            padding='valid',
        ),
        Conv2D(
            20,
            (3,3),
            padding='same',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform(seed=seed),
        ),
        MaxPool2D(
            (3,3),
            strides=1,
            padding='valid',
        ),
        BatchNormalization(),
        Flatten(),
        Dense(n, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=seed)),
        Dropout(p, seed=seed),
        Dense(12, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=seed)),
    ])

    model.compile(
        optimizer=SGD(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model 


def fit_model(model, train_data, valid_data, n_epochs):

    his = model.fit(
        train_data,
        epochs=n_epochs,
        validation_data=valid_data,
    )

    return his.history


def evaluate_model(model, data):
    """Evaluate model on every dataset from list"""
    return [model.evaluate(dataset) for dataset in data]


def plot_history(
        history,
        params,
        fig_loss=go.Figure(layout={'title': 'Loss'}),
        fig_accuracy=go.Figure(layout={'title': 'Accuracy'}),
        show_train=True,
        show_valid=True,
):
    """Add learning curve for passed history
    
    If fig_loss is not specified, it will be created, otherwise new trace will be added. Same for fig_accuracy
    """

    params = str(params)

    if show_train:

        fig_loss.add_trace(
            go.Scatter(
                x=np.arange(len(history['loss'])),
                y=history['loss'],
                name=f'train, {params}',
            )
        )

        fig_accuracy.add_trace(
            go.Scatter(
                x=np.arange(len(history['accuracy'])),
                y=history['accuracy'],
                name=f'train, {params}',
            )
        )

    if show_valid:
        
        fig_loss.add_trace(
            go.Scatter(
                x=np.arange(len(history['val_loss'])),
                y=history['val_loss'],
                name=f'valid, {params}',
            )
        )

        fig_accuracy.add_trace(
            go.Scatter(
                x=np.arange(len(history['val_accuracy'])),
                y=history['val_accuracy'],
                name=f'valid, {params}',
            )
        )

    return fig_loss, fig_accuracy


def create_data(data_dir_name, batch_size, pad_shape, seed=0):
    """Create train, valid and test datasets and rescale them to range (-0.5; 0.5)"""

    validation_split = 0.15 # must be the same for train and valid data

    train_data = image_dataset_from_directory(
        directory=f'{data_dir_name}/train',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=pad_shape,
        color_mode='grayscale',
        validation_split=validation_split,
        seed=seed,
        subset='training',
        shuffle=True,
    )
    valid_data = image_dataset_from_directory(
        directory=f'{data_dir_name}/train',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=pad_shape,
        color_mode='grayscale',
        validation_split=validation_split,
        seed=seed,
        subset='validation',
        shuffle=True,
    )
    test_data = image_dataset_from_directory(
        directory=f'{data_dir_name}/test',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=pad_shape,
        color_mode='grayscale',
        shuffle=True,
    )

    # Rescale datasets to put pixel values in range (0, 1)
    train_data = train_data.map(lambda x, y: (Rescaling(1/255, offset=0)(x), y))
    valid_data = valid_data.map(lambda x, y: (Rescaling(1/255, offset=0)(x), y))
    test_data = test_data.map(lambda x, y: (Rescaling(1/255, offset=0)(x), y))

    return train_data, valid_data, test_data
