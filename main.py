from src.data_analysis import run_analysis
from src.training.train import train_model
from src.training.evaluate import evaluate_model


def main():

    # Step 1: Dataset analysis
    run_analysis()

    # Step 2: Train model
    model, X_test, y_test = train_model()

    # Step 3: Evaluate model
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":

    main()