import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, utils
from datasets import load_dataset
import warnings
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class TransformerTrainer:
    def __init__(self, model_name="bert-base-uncased", num_labels=2, lr=5e-5, batch_size=8):
        # Initialize the model, tokenizer, optimizer, and loss function
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.loss_history = []  # Add a list to store loss values

    def tokenize_data(self, examples):
        # Tokenize the inputs (for text classification)
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def train_one_epoch(self, dataloader):
        # Set the model to train mode
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            inputs = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, train_dataset, epochs=3):
        # Create DataLoader for the training dataset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in trange(epochs, desc="Epochs"):
            avg_loss = self.train_one_epoch(train_loader)
            self.loss_history.append(avg_loss)  # Store the average loss for each epoch

    def train_ensemble(self, train_dataset, epochs=3):
        # Create DataLoader for the training dataset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize an accumulator for the weights and biases
        weight_accumulator = None

        for epoch in trange(epochs, desc="Learners"):
            # Initialize a new model for each epoch
            model = BertForSequenceClassification.from_pretrained(self.model.config._name_or_path, num_labels=self.model.config.num_labels)
            model.to(self.device)
            optimizer = AdamW(model.parameters(), lr=self.optimizer.defaults['lr'])

            # Train the model for one epoch
            loss = self.train_one_epoch(train_loader)

            avg_loss = loss / len(train_loader)

            # Accumulate the weights and biases using a running average
            state_dict = model.state_dict()
            if weight_accumulator is None:
                weight_accumulator = {key: value.clone() for key, value in state_dict.items()}
            else:
                for key in weight_accumulator:
                    weight_accumulator[key] = (weight_accumulator[key] * epoch + state_dict[key]) / (epoch + 1)


def plot_loss(normal_trainer, ensemble_trainer, output_path="loss.png"):
    plt.plot(normal_trainer.loss_history, label="Normal Training")
    plt.plot(ensemble_trainer.loss_history, label="Ensemble Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Quitting.")
        exit(0)

    # Load the IMDb dataset from Hugging Face
    dataset = load_dataset("imdb", split="train[:1%]")  # Use a small sample for testing (2% of train set)

    # Initialize the trainer
    normal_trainer = TransformerTrainer()
    ensemble_trainer = TransformerTrainer()

    # Tokenize the dataset
    dataset = dataset.map(normal_trainer.tokenize_data, batched=True)

    # Rename the label column to match the expected 'labels' name in Hugging Face models
    dataset = dataset.rename_column("label", "labels")

    # Set the dataset format to PyTorch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Train the model
    normal_trainer.train(dataset, epochs=6)
    ensemble_trainer.train_ensemble(dataset, epochs=6)

    # Plot the loss history
    plot_loss(normal_trainer, ensemble_trainer)
