#!/usr/bin/env python3

# Limit RAM usage
from resource import RLIMIT_AS, getrlimit, setrlimit

soft, hard = getrlimit(RLIMIT_AS)
setrlimit(RLIMIT_AS, (24000000000, hard))

# Data Preprocessing/Loading
from pathlib import Path

from numpy import array, load, savez_compressed, zeros
from pandas import read_csv

t_x_train = []
t_y_train = []
t_x_test = []
t_y_test = []
x_x_train = []
x_y_train = []
x_x_test = []
x_y_test = []

def load_data(path: str, classification: int):
    for i in range(1, 7):
        # T
        file = path.format(i=i)
        data = read_csv(file, header=None).to_numpy()

        # X
        data_b = read_csv(file.replace("T", "X"), header=None).to_numpy()

        # Y
        y = zeros((10, 4))
        y[:-1, 0] = 1
        y[-1, classification] = 1

        if i == 6:
            t_x_test.append(data)
            t_y_test.append(y)
            x_x_test.append(data_b)
            x_y_test.append(y)
        else:
            t_x_train.append(data)
            t_y_train.append(y)
            x_x_train.append(data_b)
            x_y_train.append(y)

if not Path("data.npz").exists():
    load_data("data/T/A/T_a{i}_new.csv", 0)
    load_data("data/T/L/T_L{i}_new.csv", 1)
    load_data("data/T/LS/T_LS{i}_new.csv", 2)
    load_data("data/T/S/T_S{i}_new.csv", 3)

    t_x_train = array(t_x_train)
    t_y_train = array(t_y_train)
    t_x_test = array(t_x_test)
    t_y_test = array(t_y_test)
    x_x_train = array(x_x_train)
    x_y_train = array(x_y_train)
    x_x_test = array(x_x_test)
    x_y_test = array(x_y_test)

    # # Data Exporting
    savez_compressed("data.npz", t_x_train=t_x_train, t_y_train=t_y_train, t_x_test=t_x_test, t_y_test=t_y_test, x_x_train=x_x_train, x_y_train=x_y_train, x_x_test=x_x_test, x_y_test=x_y_test)
else:
    data = load("data.npz")
    t_x_train = data["t_x_train"]
    t_y_train = data["t_y_train"]
    t_x_test = data["t_x_test"]
    t_y_test = data["t_y_test"]
    x_x_train = data["x_x_train"]
    x_y_train = data["x_y_train"]
    x_x_test = data["x_x_test"]
    x_y_test = data["x_y_test"]

print(t_x_train.shape)
print(t_y_train.shape)
print(t_x_test.shape)
print(t_y_test.shape)
print(x_x_train.shape)
print(x_y_train.shape)
print(x_x_test.shape)
print(x_y_test.shape)

# T Model
from keras import Model
from keras.layers import Conv1D, Dense, Flatten, Input
from keras.optimizers.adamw import AdamW
from keras.utils import plot_model
from matplotlib.pyplot import clf, legend, plot, savefig, show, title, xlabel, ylabel
from numpy import swapaxes
from sklearn.metrics import ConfusionMatrixDisplay

# T_input_layer = Input((500001, 10))
# T_intermediate = Conv1D(10, 1000, 250, activation="selu")(T_input_layer)
# T_intermediate = Flatten()(T_intermediate)
# T_intermediate = Dense(40, "selu")(T_intermediate)
# T_output = []
# for i in range(10):
#     T_output.append(Dense(4, "softmax")(T_intermediate))
# T_model = Model(inputs=T_input_layer, outputs=T_output)
# T_model.compile(optimizer=AdamW(0.0001, amsgrad=True), loss="categorical_crossentropy", metrics="accuracy")
# T_model.summary()
# plot_model(T_model, "images/T_model.png", True)

# T_history = T_model.fit(t_x_train, [i for i in swapaxes(t_y_train, 0, 1)], epochs=50, batch_size=4, validation_data=(t_x_test, [i for i in swapaxes(t_y_test, 0, 1)]))
# T_history = T_history.history

# plot(T_history["loss"])
# plot(T_history["val_loss"])
# title("Torque Model Loss")
# xlabel("Epoch")
# ylabel("Loss")
# legend(["loss", "validation loss"])
# savefig("images/T_model_loss.png")
# show()
# clf()

# plot(T_history["dense_10_accuracy"])
# plot(T_history["val_dense_10_accuracy"])
# title("Torque Model Rod 10 Accuracy")
# xlabel("Epoch")
# ylabel("Rod 10 Accuracy")
# legend(["accurary", "validation accuracy"])
# savefig("images/T_model_accuracy.png")
# show()
# clf()

# ConfusionMatrixDisplay.from_predictions(swapaxes(t_y_test, 0, 1).argmax(axis=-1).flatten(), array(T_model.predict(t_x_test)).argmax(axis=-1).flatten(), display_labels=["Healthy", "Ramp", "Step", "Short Circuit"])
# savefig("images/T_model_confustion_matrix.png")
# show()
# clf()

# T_model.save("T_model")

# X Model
X_input_layer = Input((500001, 10))
X_intermediate = Conv1D(10, 1000, 250, activation="selu")(X_input_layer)
X_intermediate = Flatten()(X_intermediate)
X_intermediate = Dense(40, "selu")(X_intermediate)
X_output = []
for i in range(10):
    X_output.append(Dense(4, "softmax")(X_intermediate))
X_model = Model(inputs=X_input_layer, outputs=X_output)
X_model.compile(optimizer=AdamW(0.0001, amsgrad=True), loss="categorical_crossentropy", metrics="accuracy")
X_model.summary()
plot_model(X_model, "images/X_model.png", True)

T_history = X_model.fit(x_x_train, [i for i in swapaxes(x_y_train, 0, 1)], epochs=50, batch_size=4, validation_data=(x_x_test, [i for i in swapaxes(x_y_test, 0, 1)]))
T_history = T_history.history

plot(T_history["loss"])
plot(T_history["val_loss"])
title("Position Model Loss")
xlabel("Epoch")
ylabel("Loss")
legend(["loss", "validation loss"])
savefig("images/X_model_loss.png")
show()
clf()

plot(T_history["dense_10_accuracy"])
plot(T_history["val_dense_10_accuracy"])
title("Position Model Rod 10 Accuracy")
xlabel("Epoch")
ylabel("Rod 10 Accuracy")
legend(["accurary", "validation accuracy"])
savefig("images/X_model_accuracy.png")
show()
clf()

ConfusionMatrixDisplay.from_predictions(swapaxes(x_y_test, 0, 1).argmax(axis=-1).flatten(), array(X_model.predict(x_x_test)).argmax(axis=-1).flatten(), display_labels=["Healthy", "Ramp", "Step", "Short Circuit"])
savefig("images/X_model_confustion_matrix.png")
show()
clf()

X_model.save("X_model")
