import tsr_model

def main():

    model = tsr_model.train_model()
    tsr_model.test_model(model)
    
main()