import json
from model import GRUPredictor
from data_handler import make_dataset
from train import train_gru_predictor
from trade import simulate_trading


def train_model():

    with open("train_nn.json", "r") as fil:
        train_config = json.load(fil)

    train_tickers = train_config["tickers"]
    train_start = train_config["start"]
    train_end = train_config["end"]
    train_seq_length = train_config["seq_length"]
    train_interval = train_config["interval"]
    train_epochs = train_config["epochs"]
    train_batch_size = train_config["batch_size"]
    train_lr = train_config["lr"]

    print(" -------------- TRAINING --------------")
    
    print(f"Tickers: {train_tickers}, Start: {train_start}, End: {train_end}, Interval: {train_interval}")

    dataset, input_dim = make_dataset(train_tickers, train_start, train_end, train_seq_length, train_interval, normalize=True)
    model = GRUPredictor(input_dim=input_dim)
    train_gru_predictor(model, dataset, epochs=train_epochs, batch_size=train_batch_size, lr=train_lr)

    return model




def test_model(model):

    with open("test_nn.json", "r") as fil:
        test_config = json.load(fil)

    test_tickers = test_config["tickers"]
    test_start = test_config["start"]
    test_end = test_config["end"]
    test_seq_length = test_config["seq_length"]
    test_interval = test_config["interval"]
    test_epochs = test_config["epochs"]
    test_batch_size = test_config["batch_size"]
    test_lr = test_config["lr"]

    print(" -------------- TESTING --------------")

    print(f"Tickers: {test_tickers}, Start: {test_start}, End: {test_end}, Interval: {test_interval}")

    testDataset, input_dim = make_dataset(test_tickers, test_start, test_end, test_seq_length, test_interval, normalize=False)
    simulate_trading(model, testDataset, testDataset.tensors[1].numpy())

    return