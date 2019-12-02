import matplotlib.pyplot as plt
import numpy as np



l_val = [x/10 for x in range(1,11)]
m_val = [x/10 for x in range(1,10)]
e_val = [x for x in range(100,1100,100)]
t_val = [x for x in range(10,110,10)]

l_10c_acc = []
l_10c_tpr = []
l_10c_fpr = []
l_10c_pre = []
l_10c_rec = []
l_10c_fme = []
l_10c_auc = []

dic_10c = {
    'acc':l_10c_acc,
    'tpr':l_10c_tpr,
    'fpr':l_10c_fpr,
    'pre':l_10c_pre,
    'rec':l_10c_rec,
    'fme':l_10c_fme,
    'auc':l_10c_auc
}

base_path = 'results/MultilayerPerceptron/10_cross/'

def fill_args_up(folder_name,file_name,list_dict):
    with open(base_path+folder_name+'/'+file_name,"r") as file:
        line=file.readline()
        while line:
            if line[:30] == 'Correctly Classified Instances':
                split = line.split(' ')
                split = [elt for elt in split if elt!='']
                list_dict['acc'].append(float(split[-2])/100)
            if line[:13] == 'Weighted Avg.':
                split = line.split(' ')
                split = [elt for elt in split if elt!='']
                list_dict['tpr'].append(float(split[2]))
                list_dict['fpr'].append(float(split[3]))
                list_dict['pre'].append(float(split[4]))
                list_dict['rec'].append(float(split[5]))
                list_dict['fme'].append(float(split[6]))
                list_dict['auc'].append(float(split[8]))
            line=file.readline()


if __name__ == "__main__":
    for elt in e_val:
        fill_args_up('4-Epochs','E'+str(elt)+'.txt',dic_10c)
    print(dic_10c)
    plt.figure()
    plt.xlabel('Number of epochs')
    plt.ylabel('Precision')
    plt.plot(e_val,dic_10c['tpr'])
    plt.show()
