import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # index into this dict to get what must be done for each value
    conversion_map = ["int", "float", "int", "float", "int", 
                      "float", "float", "float", "float", "float",
                      "month", "int", "int", "int", "int", 
                      "visitor", "tbool"]
    
    def convert(funcval, val):
        if funcval == "int":
            return int(val)
        if funcval == "float":
            return float(val)
        if funcval == "month":
            return {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
                    "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}[val]
        if funcval == "visitor":
            return 1 if val == "Returning_Visitor" else 0
        if funcval == "tbool":
            return 1 if val == "TRUE" else 0
        # shouldn't be any other kinds
        raise AssertionError
    with open(filename) as f:
        reader = csv.reader(f)
        # skip over header row
        next(reader)
        # initialize collection variables
        evidence = []
        labels = []
        counter = 0
        for row in reader:
            # build up rowlist from all except last column
            rowlist = []
            for i in range(len(conversion_map)):
                rowlist.append(convert(conversion_map[i], row[i]))
            evidence.append(rowlist)
            # last column is label
            labels.append(convert("tbool", row[-1:][0]))
    return (np.array(evidence), np.array(labels))


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    totalPos = (labels == 1).sum()
    correctPos = (predictions[labels == 1] == 1).sum()
    totalNeg = (labels == 0).sum()
    correctNeg = (predictions[labels == 0] == 0).sum()
    sensitivity = float(correctPos)/totalPos
    specificity = float(correctNeg)/totalNeg
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
