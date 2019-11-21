'''
# created 20/11/2019 15:33
# by Q.Ducasse
'''

import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
from loader          import *
from naive_bayes     import *
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =======================================================
# ------------------- DECISION TREES --------------------
# =======================================================

def decision_trees_train_test(df,class_feature,ratio=0.2,heatmap=False):
    '''
    Runs a decision model over the dataset with the class feature.

    '''
    print("================================================================")
    print("Running Decision Trees on dataframe '{0}'\nwith class feature '{1}'".format(df.name,class_feature))
    # Separate train and test data
    data_train, data_test, target_train, target_test = separate_train_test(df,class_feature,ratio=ratio)
    # Create a Decition Tree classifier with the gini criterion
    dct_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
    # Train the algorithm on training data and predict using the testing data
    dct_gini.fit(data_train, target_train)

    # Create a Decition Tree classifier with the entropy criterion
    dct_entr = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
    # Train the algorithm on training data and predict using the testing data
    dct_entr.fit(data_train, target_train)

    # Predictions
    trees = [dct_gini, dct_entr]
    for tree in trees:
        # Perform the prediction on the test_data
        pred = tree.predict(data_test)
        # Print te accuracy
        print("Decision Tree ("+ tree.criterion +") accuracy: ", accuracy_score(target_test, pred, normalize = True))
        # Print the confusion matrix of the model
        print_cmat(target_test,pred,heatmap=heatmap)




if __name__ == "__main__":
        # Loading and Preprocessing
    signs, signs_rd = load_base_dataset(path_x_train,path_y_train)
    sm_signs = select_instances(signs,'label')
    sm_signs_rd = randomise(sm_signs)

    ## UNCOMMENT IF YOU HAVE TO GENERATE THE FILES WITH THE BEST ATTRIBUTES
    # store_best_attributes()

    signs_ba2, signs_ba2_rd = dataset_best_n_attributes(2,signs)
    signs_ba5, signs_ba5_rd = dataset_best_n_attributes(5,signs)
    signs_ba10, signs_ba10_rd = dataset_best_n_attributes(10,signs)
    sm_signs_ba2, sm_signs_ba2_rd = dataset_best_n_attributes(2,sm_signs)
    sm_signs_ba5, sm_signs_ba5_rd = dataset_best_n_attributes(5,sm_signs)
    sm_signs_ba10, sm_signs_ba10_rd = dataset_best_n_attributes(10,sm_signs)

    # Store the datasets
    # signs_ba2.to_csv('./data/x_train_gr_smpl_2ba.csv')
    # signs_ba5.to_csv('./data/x_train_gr_smpl_5ba.csv')
    # signs_ba10.to_csv('./data/x_train_gr_smpl_10ba.csv')

    df_to_test = [
        # signs,
        signs_rd,
        # signs_ba2_rd,
        # signs_ba5_rd,
        # signs_ba10_rd,
        # sm_signs,
        # sm_signs_rd,
        # sm_signs_ba2_rd,
        # sm_signs_ba5_rd,
        # sm_signs_ba10_rd
    ]
    # Run Bayes over the new sets
    for df in df_to_test:
        decision_trees_train_test(df,'label',heatmap=False)
