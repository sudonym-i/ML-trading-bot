import json
from model import GRUPredictor
from data_handler import make_training_dataset
from train import train_gru_predictor
from trade import simulate_trading

# Load config values from config.json, contains all user specifications
with open("config.json", "r") as fil:
    config = json.load(fil)

tickers = config["tickers"]
start = config["start"]
end = config["end"]
seq_length = config["seq_length"]
interval = config["interval"]
epochs = config["epochs"]
batch_size = config["batch_size"]
lr = config["lr"]


print(f"Tickers: {tickers}, Start: {start}, End: {end}, Interval: {interval}")


def main():

    dataset, input_dim = make_training_dataset(tickers, start, end, seq_length, interval, normalize=True)
    model = GRUPredictor(input_dim=input_dim)
    train_gru_predictor(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr)

    testDataset, input_dim = make_training_dataset(tickers, start, end, seq_length, interval, normalize=False)
    simulate_trading(model, testDataset, testDataset.tensors[1].numpy())

main()