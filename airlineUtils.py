import math

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def display_results(dataset, targets, predictions, val_start_idx, test_start_idx):
    # Desnormalizar los valores
    targets_inverse = dataset.inverse_normalize(targets)
    predictions_inverse = dataset.inverse_normalize(predictions)

    # Separar train/val/test
    train_predictions = np.empty_like(predictions)
    train_predictions[:] = np.nan
    train_predictions[:val_start_idx] = predictions_inverse[:val_start_idx]
    if test_start_idx - val_start_idx > 0:
        val_predictions = np.empty_like(predictions)
        val_predictions[:] = np.nan
        val_predictions[val_start_idx:test_start_idx] = predictions_inverse[val_start_idx:test_start_idx]
    test_predictions = np.empty_like(predictions)
    test_predictions[:] = np.nan
    test_predictions[test_start_idx:] = predictions_inverse[test_start_idx:]

    # Imprimir RMSE
    print("Test RMSE: ", math.sqrt(mean_squared_error(targets_inverse[test_start_idx:], predictions_inverse[test_start_idx:])))

    # Graficar targets y predicciones
    fig = plt.figure(figsize=(12, 8))
    plt.plot(targets_inverse, color="blue", label="Original")
    plt.plot(train_predictions, color="green", label="Predicho en entrenamiento")
    if test_start_idx - val_start_idx > 0:
        plt.plot(val_predictions, color="purple", label="Predicho en validación")
    plt.plot(test_predictions, color="red", label="Predicho en test")
    plt.xlabel('Tiempo [Meses]')
    plt.ylabel('Cantidad de pasajeros en la aerolinea')
    plt.title('Pasajeros en aeorlinea (Enero 1949 - Diciembre 1960)')
    plt.legend(loc="upper left")
    plt.savefig("plot.png")
