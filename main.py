from model import Model
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# run the model to see the results


if __name__ == '__main__':

    multiprocessing.freeze_support()

    model = Model()
        
    n_epochs = 7

    total_loss, best_val_acc = model.train_model(num_epochs=n_epochs, learning_rate=0.3, momentum=0.9)

    print(f"Best Val Accuracy: {best_val_acc:.2f}%")


    # Plot the loss evolution, where the x-axis is the epoch and the y-axis is the loss
    plt.plot(range(1, n_epochs + 1), total_loss)
    plt.xticks(range(1, n_epochs + 1))  # Set the x-axis ticks to 1, 2, 3
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.show()

    test_accuracy = model.test_model()
    print(f"Test Accuracy: {test_accuracy:.2f}%")