
from model import Model
import multiprocessing
import matplotlib.pyplot as plt

# run the model to see the results

multiprocessing.freeze_support()

model = Model()
    
n_epochs = 15

total_loss = model.train_network(num_epochs=n_epochs, learning_rate=0.85, momentum=0.9)


# Plot the loss evolution, where the x-axis is the epoch and the y-axis is the loss
plt.plot(range(1, n_epochs + 1), total_loss)
plt.xticks(range(1, n_epochs + 1))  # Set the x-axis ticks to 1, 2, 3
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Evolution')
plt.show()

test_accuracy = model.test_network()
print(f"Test Accuracy: {test_accuracy:.2f}%")

model.save_model("./models")