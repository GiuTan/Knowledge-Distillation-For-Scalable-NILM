import numpy as np
from sklearn.metrics import  precision_recall_curve
from matplotlib import pyplot as plt


def standardize_data(agg, mean, std):
    agg= agg -  mean
    agg /= std
    return agg
def app_binarization_strong(output,thres, classes):
    new_output = []
    for k in range(len(output)):
             for i in range(classes):
                #curr = output[k]
                if output[k][i] >= thres[i]:
                    output[k][i] = 1
                else:
                    output[k][i] = 0
                #matrix[l] = curr[l]
             new_output.append(output[k])

    new_output = np.array(new_output)
    # return new_output
    return new_output

def thres_analysis(Y_test,new_output,classes):

    precision = dict()
    recall = dict()
    thres_list_strong = []

    for i in range(classes):

        precision[i], recall[i], thresh = precision_recall_curve(Y_test[:, i], new_output[:, i])

        plt.title('Pres-Recall-THRES curve')
        plt.plot(precision[i], recall[i])
        plt.show()
        plt.close()

        f1 = (2 * precision[i] * recall[i] )/ (precision[i] + recall[i])
        opt_thres_f1 = np.argmax(f1)
        optimal_threshold_f1 = thresh[opt_thres_f1]
        print("Threshold for F1-SCORE value is:", optimal_threshold_f1)
        if (optimal_threshold_f1 >= 0.95 and i!=0): #
             optimal_threshold_f1 = 0.8


        thres_list_strong.append(optimal_threshold_f1)

    return thres_list_strong
