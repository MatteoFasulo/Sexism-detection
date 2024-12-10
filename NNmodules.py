import torch
from detector import SexismDetector
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import copy
import matplotlib.pyplot as plt
import gc
import os

class BaselineModel(torch.nn.Module):
    def __init__(self, embedding_model, detector, hidden_dim, output_dim: int = 1, lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding.from_pretrained(
            embeddings=torch.from_numpy(embedding_model.vectors),
            freeze=True,
            padding_idx=embedding_model.get_index(detector.PAD_TOKEN)
        )
        self.lstm = torch.nn.LSTM(input_size=embedding_model.vector_size, hidden_size=hidden_dim, bidirectional=True, num_layers=lstm_layers, batch_first=True)
        self.fc = torch.nn.Linear(in_features=2 * hidden_dim, out_features=output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        out = self.word_embeddings(inputs)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = torch.mean(out, dim=1)  # Average pooling over the seq len
        out = self.fc(out)
        return out

# fai che ha inherit dall'altra così quando devi prendere i dataloader basta che calli self e hai già tutto (imo)
# ha senso

class LSTM_network():
    def __init__(self, name, seed: int, detector : SexismDetector, device, embedding_model, embedding_model_aug, hidden_dim, output_dim, lstm_layers, dropout):
        self.name = name

        self.seed = seed
        self.__set_seed(seed)

        self.device = device
        self.network = BaselineModel(embedding_model, detector, hidden_dim, output_dim, lstm_layers, dropout)
        self.network.to(device)
        self.detector = detector
        self.__embedding_model_aug = embedding_model_aug
        self.history = None

        self.trained = False

    def __set_seed(self, seed):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def plot_history(self):
        if self.history:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            baseline_model_name = self.history['model'] == self.name
            self.history[baseline_model_name].plot(x='epoch', y=['train_loss', 'val_loss'], title='Loss', ax=axes[0])
            self.history[baseline_model_name].plot(x='epoch', y=['train_acc', 'val_acc'], title='Accuracy', ax=axes[1])
            self.history[baseline_model_name].plot(x='epoch', y='learning_rate', title='Learning Rate', ax=axes[2], logy=True)

            plt.tight_layout()
            plt.show()
        else:
            raise Exception("You need to train the model first!")

    def train(self, train, val, epochs, batch_size, lr, verbose=True):
        # Preparation
        train_dloader = self.detector.get_dataloader(data=train,
                                                     embedding_model=self.__embedding_model_aug,
                                                     type="train",
                                                     batch_size=batch_size,
                                                     shuffle=True)

        val_dloader = self.detector.get_dataloader(data=val,
                                                     embedding_model=self.__embedding_model_aug,
                                                     type="val",
                                                     batch_size=batch_size,
                                                     shuffle=True)
                                            

        history = pd.DataFrame(columns=['model', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'learning_rate'])
        
        #Queste cose le ho messe qua se no c'era l'init che era con 10000 parametri
        #Tra cui il test,train e val set, che secondo me stanno meglio quando chiami il train

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train['label']), y=train['label'])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(device), 
            ignore_index=self.__embedding_model_aug.get_index(self.detector.PAD_TOKEN)
        )

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        # Training loop

        # Initialize parameters of Early stopping
        best_val_loss = np.inf
        best_model = None
        best_model_epoch = 0
        best_f1_score = 0


        # Train the model
        for epoch in range(epochs):
            # Initialize the training variables for each epoch
            train_correct = 0
            train_total = 0

            # Set the model to training mode
            self.network.train()
            # Iterate over the training data in batches
            for batch in train_dloader:
                # Get the inputs and labels
                sentences, labels = batch

                # Move the inputs and labels to the device (GPU)
                sentences = sentences.to(device)
                labels = labels.to(device)  # Labels should be integers for CrossEntropyLoss

                # Zero grad the optimizer
                optimizer.zero_grad()

                # Forward pass
                output = self.network(sentences)

                # Compute the loss
                loss = loss_function(output, labels)

                # Backward pass
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Compute the accuracy
                predicted = torch.argmax(output, dim=1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            self.network.eval()
            final_val_loss = 0
            val_correct = 0
            val_total = 0

            predicted_arr = []
            labels_arr = []

            with torch.no_grad():
                for batch in val_dloader:
                    sentence, labels = batch
                    sentence = sentence.to(device)
                    labels = labels.to(device)

                    output = self.network(sentence)
                    val_loss = loss_function(output, labels)

                    final_val_loss += val_loss.item()

                    # Apply sigmoid to logits before thresholding
                    predicted = torch.argmax(output, dim=1)

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Store predictions and true labels
                    predicted_arr.append(predicted)
                    labels_arr.append(labels)

            # Concatenate tensors
            predicted_arr = torch.cat(predicted_arr, dim=0).cpu().numpy()
            labels_arr = torch.cat(labels_arr, dim=0).cpu().numpy()

            # Compute the macro F1 score
            f1 = f1_score(y_true=labels_arr, y_pred=predicted_arr, average='macro')

            # Update the learning rate
            scheduler.step(final_val_loss)

            # Early stopping
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model = copy.deepcopy(self.network.state_dict())
                best_model_epoch = epoch
                best_f1_score = f1

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}\tLoss: {loss.item():.4f}\tAcc: {train_correct / train_total:.4f}\tVal Loss: {final_val_loss:.4f}\tVal Acc: {val_correct / val_total:.4f}\tmacro_f1: {f1:.4f}\tlr: {scheduler.get_last_lr()[0]:.7f}")

            # Save the history
            history.loc[len(history)] = [self.name, epoch + 1, loss.item(), final_val_loss, train_correct / train_total, val_correct / val_total, scheduler.get_last_lr()[0]]

        # Restore the best model
        self.network.load_state_dict(best_model)
        print(f"Best model found at epoch {best_model_epoch + 1} with validation loss: {best_val_loss:.4f} and f1 sore: {best_f1_score:.4f}")

        # Save the best model
        torch.save(self.network.state_dict(), detector.MODEL_FOLDER / f'{self.name}.pth')

        #Save the history
        self.history = history
        self.trained = True

        #Return the history
        return history

    def test(self, test, batch_size):
        if not self.trained:
            print("WARNING: network is not trained yet!")

        test_dloader = self.detector.get_dataloader(data=test,
                                                     embedding_model=self.__embedding_model_aug,
                                                     type="test",
                                                     batch_size=batch_size,
                                                     shuffle=True)
        #TODO: create test code
        pass
        



# Usage
detector = SexismDetector()
embedding_model = None
embedding_model_aug = None
device = "cuda"
hidden_dim=128
output_dim=1
lstm_layers=1
drouput=0.4

seed1 = 1337
#Creating the newtork
net1 = LSTM_network(seed1, detector, device, embedding_model, embedding_model_aug, hidden_dim, output_dim, lstm_layers, drouput)


#Training the network
train_data = None
val_data = None
epochs = 20
batch_size = 128
hist = net1.train(train=train_data, val=val_data, epochs=epochs, batch_size=batch_size, lr=1e-3, verbose=True)

#You can print the history
net1.plot_history()

# Clear the cache
with torch.no_grad():
    torch.cuda.empty_cache()

gc.collect()


test_data = None
#TODO: implement
net1.test(test_data)