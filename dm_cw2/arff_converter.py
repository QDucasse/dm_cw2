'''
# created 02/11/2019 18:56
# by Q.Ducasse
'''

import random
from random          import shuffle
from loader          import path_x_train, path_y_train, path_boolean_mask
from best_attributes import ba0,ba1,ba2,ba3,ba4,ba5,ba6,ba7,ba8,ba9

path_base_arff = './data/'


def read_rows(path = path_x_train,index_list = [i for i in range(2304)]):
    '''
    Store the rows of a file in a list.
    Returns
    =======
    rows: string list
        List of the rows read from the file.
    '''
    rows = []
    with open(path_x_train,'r') as reader:
        for i,line in enumerate(reader.readlines()):
            pixels = line.split(",")
            selected_pixels = [pixels[i] for i in index_list]
            rows.append(','.join(selected_pixels))
    return rows

def read_add_labels(path,rows):
    '''
    Read lines from a given path and add them as a prefix to
    the given rows list.
    Parameters
    ==========
    path: string
        Path to the labels or prefixes to add.
    rows: string list
        List of the rows previously extracted that need to be prefixed

    Returns
    =======
    Rows prefixed by the elements from the file <path>
    '''
    rows_with_label = []
    with open(path,'r') as reader:
        for i,line in enumerate(reader.readlines()):
            if (i!=0):
                rows_with_label.append(rows[i].strip() + ', label' + line)
    return rows_with_label

def read_add_labels_numeric(path,rows):
    '''
    Read lines from a given path and add them as a prefix to
    the given rows list.
    Parameters
    ==========
    path: string
        Path to the labels or prefixes to add.
    rows: string list
        List of the rows previously extracted that need to be prefixed

    Returns
    =======
    Rows prefixed by the elements from the file <path>
    '''
    rows_with_label = []
    with open(path,'r') as reader:
        for i,line in enumerate(reader.readlines()):
            if (i!=0):
                rows_with_label.append(rows[i].strip() + ',' + line)
    return rows_with_label

def select_instances(lab_dataset,rd=True,numeric=False):
    '''
    Select the same number of instances for each label (first n ones).
    '''
    instance_holder = [[],[],[],[],[],[],[],[],[],[]]
    for instance in lab_dataset:
        label = int(instance[-2])
        instance_holder[label].append(instance)
    min_instances = min(map(len,instance_holder))

    new_dataset = []
    # [instance for instance_set[:min_instances] in instance_holder for instance in instance_set]
    for instance_set in instance_holder:
        if rd:
            sm_instance_set = random.sample(instance_set, min_instances)
        else:
            sm_instance_set = instance_set[:min_instances]
        for instance in sm_instance_set:
            new_dataset.append(instance)
    return new_dataset

def apply_boolean_mask(lab_dataset,nb):
    '''
    Apply a boolean mask to a labelled dataset.
    Parameters
    ==========
    lab_dataset: string
        Labelled dataset.
    nb: int
        Label that will become 0 while others are 1.

    Returns
    =======
    Dataset labelled with a boolean mask for <nb>.
    '''
    boolean_mask_dataset = []
    for instance in lab_dataset:
        label = int(instance[-2])
        if label==nb:
            boolean_mask_dataset.append(instance[:-2]+'0\n')
        else:
            boolean_mask_dataset.append(instance[:-2]+'1\n')
    return boolean_mask_dataset

def create_header(index_list = [i for i in range(2304)]):
    '''
    Create a Weka header for our dataset.
    Parameters
    ==========
    index_list: int list
        List of the pixels that are used. By default, we use
        all of them and index_list = [0,1,2,...,2303]

    Returns
    =======
    header: string
    String of the
    '''
    header = "@RELATION signs\n\n"
    attributes = [("@ATTRIBUTE 'pixel" + str(i) + "' NUMERIC\n") for i in index_list]
    attributes += ["@ATTRIBUTE 'label' {label0,label1,label2,label3,label4,label5,label6,label7,label8,label9}\n"]
    attributes_string = "".join(attributes)
    return header + attributes_string + '\n\n'

def create_header_numeric(index_list = [i for i in range(2304)]):
    '''
    Create a Weka header for our dataset.
    Parameters
    ==========
    index_list: int list
        List of the pixels that are used. By default, we use
        all of them and index_list = [0,1,2,...,2303]

    Returns
    =======
    header: string
    String of the
    '''
    header = "@RELATION signs\n\n"
    attributes = [("@ATTRIBUTE 'pixel" + str(i) + "' NUMERIC\n") for i in index_list]
    attributes += ["@ATTRIBUTE 'label' NUMERIC\n"]
    attributes_string = "".join(attributes)
    return header + attributes_string + '\n\n'

def add_data(header,data_list):
    '''
    Combines a header and a data list.
    Parameters
    ==========
    header: string
        Weka header as produced by create_header().
    data_list: list string
        Data instances.

    Returns
    =======
    Concatenation of header, @DATA and the joined instance list
    '''
    shuffle(data_list)
    return header + "@DATA " + ''.join(data_list)

def best_n(n):
    best_attributes = ba0[:n] + ba1[:n] + ba2[:n] + ba3[:n] + ba4[:n] \
                     + ba5[:n] + ba6[:n] + ba7[:n] + ba8[:n] + ba9[:n]
    return best_attributes

def write_dataset(dataset_name,dataset):
    '''
    Write an arff file.
    Parameters
    ==========
    dataset_name: string
        Name of the dataset.
    dataset: string list
        Content of the file as generated by add_data().
    '''
    f = open(path_base_arff+dataset_name+".arff","w")
    f.write(dataset)
    f.close()

def create_dataset(path_content,path_labels,dataset_name, index_list = [i for i in range(2304)]):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels(path_labels,images)
    header = create_header(index_list)
    dataset = add_data(header,data)
    write_dataset(dataset_name,dataset)

def create_base_dataset():
    '''
    Base dataset with all instances and all attributes.
    '''
    create_dataset(path_x_train,path_y_train,"base_dataset")

def create_reduced_dataset(path_content,path_labels,dataset_name,
                            index_list = [i for i in range(2304)], random_selection=True):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels(path_labels,images)
    reduced_data = select_instances(data,rd=random_selection)
    header = create_header(index_list)
    dataset = add_data(header,reduced_data)
    write_dataset(dataset_name,dataset)

def create_reduced_dataset_numeric(path_content,path_labels,dataset_name,
                            index_list = [i for i in range(2304)], random_selection=True):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels_numeric(path_labels,images)
    reduced_data = select_instances(data,rd=random_selection)
    header = create_header_numeric(index_list)
    dataset = add_data(header,reduced_data)
    write_dataset(dataset_name,dataset)

def create_reduced_test_train_dataset(path_content,path_labels,dataset_name, ratio=0.7,
                                      index_list = [i for i in range(2304)], random_selection=True):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels(path_labels,images)
    reduced_data = select_instances(data,rd=random_selection)
    shuffle(reduced_data)
    index_split = int(ratio*len(reduced_data))
    reduced_train = reduced_data[:index_split]
    reduced_test = reduced_data[index_split:]
    header = create_header(index_list)
    dataset_train = add_data(header,reduced_train)
    dataset_test = add_data(header,reduced_test)
    write_dataset(dataset_name+'_train_'+str(int(ratio*100)),dataset_train)
    write_dataset(dataset_name+'_test_'+str(int(ratio*100)),dataset_test)

def create_reduced_test_train_dataset_numeric(path_content,path_labels,dataset_name, ratio=0.7,
                                              index_list = [i for i in range(2304)], random_selection=True):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels_numeric(path_labels,images)
    reduced_data = select_instances(data,rd=random_selection)
    shuffle(reduced_data)
    index_split = int(ratio*len(reduced_data))
    reduced_train = reduced_data[:index_split]
    reduced_test = reduced_data[index_split:]
    header = create_header_numeric(index_list)
    dataset_train = add_data(header,reduced_train)
    dataset_test = add_data(header,reduced_test)
    write_dataset(dataset_name+'_train_'+str(int(ratio*100))+'_num',dataset_train)
    write_dataset(dataset_name+'_test_'+str(int(ratio*100))+'_num',dataset_test)

def create_reduced_dataset_bm(path_content,path_labels,dataset_name, nb,
                              index_list = [i for i in range(2304)], random_selection=True):
    '''
    Creates a Weka file with the content from the first path prefixed by the
    content from the second, selecting only the <index_list> attributes
    and finally storing it under the <dataset_name>
    Parameters
    ==========
    path_content: string
        Path to the content of the instances.
    path_labels: string
        Path to the labels (prefix) of the instances.
    dataset_name: string
        Name of the final arff file.
    index_list: int list
        List of the wanted attributes.
    '''
    images = read_rows(path_content,index_list)
    data = read_add_labels(path_labels,images)
    reduced_data = select_instances(data,rd=random_selection)
    bm_reduced_data = apply_boolean_mask(reduced_data,nb)
    header = create_header(index_list)
    dataset = add_data(header,bm_reduced_data)
    write_dataset(dataset_name,dataset)

if __name__ == "__main__":
    # create_base_dataset()
    # create_reduced_dataset(path_x_train,path_y_train,'sm_rd_dataset')
    # create_reduced_dataset(path_x_train,path_y_train,'sm_dataset',random_selection = False)
    # create_reduced_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba2',index_list=list(set(best_n(2))))
    # create_reduced_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba5',index_list=list(set(best_n(5))))
    # create_reduced_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba10',index_list=list(set(best_n(10))))

    # # Boolean masked datasets
    # for i in range(10):
    #     create_reduced_dataset_bm(path_x_train,path_y_train,'/bm/sm_dataset'+str(i), random_selection=True, nb=i)
    #     create_reduced_dataset_bm(path_x_train,path_y_train,'/bm/sm_dataset_rd'+str(i), random_selection=False, nb=i)
    for ratio in [0.3, 0.7, 0.9]:
        # create_reduced_test_train_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba2',ratio=ratio,index_list=list(set(best_n(2))))
        # create_reduced_test_train_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba5',ratio=ratio,index_list=list(set(best_n(5))))
        create_reduced_test_train_dataset(path_x_train,path_y_train,'sm_rd_dataset_ba10',ratio=ratio,index_list=list(set(best_n(10))))
        # create_reduced_test_train_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba2',ratio=ratio,index_list=list(set(best_n(2))))
        # create_reduced_test_train_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba5',ratio=ratio,index_list=list(set(best_n(5))))
        # create_reduced_test_train_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba10',ratio=ratio,index_list=list(set(best_n(10))))
    # create_reduced_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba2_num',index_list=list(set(best_n(2))))
    # create_reduced_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba5_num',index_list=list(set(best_n(5))))
    # create_reduced_dataset_numeric(path_x_train,path_y_train,'sm_rd_dataset_ba10_num',index_list=list(set(best_n(10))))
