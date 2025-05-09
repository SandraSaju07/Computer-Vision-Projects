# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot train and validation accuracy
def plot_training_history(history):
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()

def plot_heatmap(matrix):
    # Plot heatmap
    sns.heatmap(matrix, annot = True, fmt = 'd')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Predictions - HeatMap')
    plt.show()