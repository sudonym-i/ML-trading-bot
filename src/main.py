from nn import train_model, test_model

def main():

    model = train_model()
    test_model(model)
    
main()