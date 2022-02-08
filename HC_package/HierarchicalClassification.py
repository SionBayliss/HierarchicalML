#!/usr/bin/env python
# coding: utf-8

# ###  NOTES
# - needs filter on node size and appropraite feedback when appling classifiers
# - needs logging to file/variable
# - run and store stats per classifier for easy debugging?
# - is dict the best for storage - can it be saved to file (perhaps updated each loop to ensure no data is lost on error)? 
# - subfunction to run classifier and evaluate at same time
# - not very pythony at the moment. 
# - remove test data and examples
# - find good example data for testing

# In[1]:


# dependencies 
import networkx as nx
import numpy as np
import pandas as pd
import copy 
import re

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score

from matplotlib import pyplot as plt

# load resamplers 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# generate example data 
def generate_example_data():

    # generate random seed
    np.random.seed(1)
    
    # example graph
    #example_graph = nx.DiGraph()
    #example_graph.add_edges_from([("root", "a"), ("root", "b"), ("root", "c"), ("a", "a1"), 
    #                      ("a", "a2"), ("b", "b1"), ("b", "b2"), ("a1", "a1a"), ("a1", "a1b"), ("c", "c1"), ("c", "c2") ])

    # example dataframe 
    example_labels = np.array(["a", "a2", "b", "b1", "b", "b2", "a1", "a1a", 
                              "a1", "a1b", "a1a", "a1b", "a1a", "a1b","a1a", 
                              "a1b", "c1", "c2"])
    test_indices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
    bin = np.random.randint(0,2,size=(len(example_labels),3))
    example_dataframe = pd.DataFrame(bin, columns = ['col1','col2', 'col3'], index = test_indices)
    
    # example metadata
    example_meta = [ ["a", "a1", "a1a"],
                ["a", "a1", "a1a"],
                ["a", "a1", "a1b"],
                #["a", "a1", "a1c"],
                ["a", "a2", ""],
                ["b", "b1", ""],
                #["b", "a", ""], ##
                #["b", "a", ""], ##
                ["b", "b2", ""],
                ["c", "c1", ""],
                ["c", "c2", ""]
              ]
    example_metadata = pd.DataFrame(example_meta, columns = ['a','b','c'])
    
    return(example_dataframe, example_labels, example_metadata) # example_graph


# In[3]:


def metadata_to_DAG(df, columns = "", visualise = False):
  
    # check input is dataframe
    if not isinstance(df, pd.DataFrame):
        return(None)
    else:

        # check it contains input columns - subset data on columns 
        if len(columns) > 0:
            if not all([item in df.columns for item in columns]):
                print(" - ERROR: Filter columns not in input dataframe, using full dataframe.")
            else:
                df = df[columns]
                
        # convert nans to empty strings
        df = df.fillna("")

        # create output graph
        out_graph = nx.DiGraph()

        # parse all rows 
        for r in range(df.shape[0]):
            for c in range(df.shape[1]-1):
                val1 = df.iloc[r,c]
                val2 = df.iloc[r,c+1]

                # add root
                if c == 0:
                    out_graph.add_edges_from([('root', val1)])

                # store edge
                if (val1 != "") and (val2 != ""):
                    out_graph.add_edges_from([(val1, val2)])

        # [optional] plot DAG
        if visualise is True:
            plt.tight_layout()
            nx.draw_networkx(out_graph, arrows=True)
            
        ### TODO: Additional checks on type of input graph maybe needed
        # check is DAG and return - otherwise return None
        if not (nx.is_directed_acyclic_graph(out_graph)):
            print(" - ERROR: input metadata does not generate a DAG")
            
        return(out_graph)
        


# In[4]:


# generate label variable 
def create_labels(df, columns = ""):
    '''generate labels variable from input dataframe. 
    Order of columns is assumed to be that of --columns variable. 
    If no columns variable is passed all columns will be used 
    in provided order'''
  
    # check input is dataframe
    if not isinstance(df, pd.DataFrame):
        return(None)
    else:

        # check it contains input columns - subset data on columns 
        if len(columns) > 0:
            if not all([item in df.columns for item in columns]):
                print(" - ERROR: Filter columns not in input dataframe, using full dataframe.")
            else:
                df = df[columns]
        else:
            columns = df.colnames()
        
        # convert nans to empty strings
        df = df.fillna("")
        
        # for each row select rightmost column as labels variable
        output_labels = []
        for i in df.index: 
            #print(i)
            
            # process column in reverse order - stop when non-empty string found
            found = 0
            for j in columns[::-1]:
                val = df[j].loc[i]
                if not val == "":
                    output_labels.append(val)
                    found = 1
                    break
            
            if found == 0:
                print(" - ERROR: one row (i) did not contain any label data.")
                return(None)
        
        # return output labels
        return(output_labels)
        
        
#labs = create_labels(meta, columns = ['Region', 'Subregion', 'Country'])


# In[5]:


# identify nodes in DAG which contain < threshold number of samples and return indices
def identify_nodes_to_remove(graph, labels, threshold=0):

    # check labels is array 
    if not type(labels) == 'numpy.ndarray':
        labels = np.array(labels)
    
    # check threshold is a number
    if type(threshold) != int and type(threshold) != float:
        raise ValueError("Invalid threshold value. Expected numeric.")
        
    # set storage variables 
    class_store = []
    index_store = []

    # generate nodes in topographical order
    nodes = nx.topological_sort(graph)
    
    # process nodes topographically
    for current in nx.topological_sort(graph):

        if not current == 'root':
            
            # find sublabels for current level in hierarchy
            relab = copy.copy(labels) 
            
            # relabel all descendant of current node to current node (byDecendants = False)
            if not is_tip(graph, current):
                 relab = rename_labels_by_node(graph, current, copy.copy(labels), byDecendants = False)
            
            # using list comprehension and enumerate() to get match indices
            res = [key for key, val in enumerate(relab) if val == current]

            # find size of node
            node_size = len(res)
            
            # if node_size <= threshold then add indices to those to remove
            if node_size<threshold:
                index_store = list(set(index_store)|set(res)) # find union 
                class_store.append(current)
            
    
    return(index_store,class_store)
    
#identify_nodes_to_remove(graph, labels, 100)    ####


# In[6]:


# identify nodes in DAG which contain < threshold number of samples and return indices
def relabel_samples_by_count_per_node(graph, labels, threshold=0, verbose = False, remove = False):

    # check labels is array 
    if not type(labels) == 'numpy.ndarray':
        labels = np.array(labels)
    
    # check threshold is a number
    if type(threshold) != int and type(threshold) != float:
        raise ValueError("Invalid threshold value. Expected numeric.")
        
    # set storage variables 
    class_store = []
    index_store = []
    labels_out = copy.copy(labels)
    
    # generate nodes in topographical order
    nodes = nx.topological_sort(graph)
    
    # process nodes topographically
    for current in nx.topological_sort(graph):
        
        if current == 'root':
              pass
        else:

            # find sublabels for current level in hierarchy
            relab = copy.copy(labels_out) 

            # relabel all descendant of current node to current node (byDecendants = False)
            if not is_tip(graph, current):
                 relab = rename_labels_by_node(graph, current, copy.copy(labels_out), byDecendants = False)

            # using list comprehension and enumerate() to get matching indices
            res = [key for key, val in enumerate(relab) if val == current]

            # find size of node
            node_size = len(res)
            
            if (verbose == True):
                print(" - ", current, ": ", node_size)

            # if node_size <= threshold then roll samples up a node in the hierarchy and 
            # relabel appropriately or remove if root [optional]
            if node_size < threshold:

                # get ancestor node 
                anc = find_ancestors(graph, current)
                
                # relabel samples in node and descendants to ancestor
                labels_out[res] = anc
                
                # append label to output class store 
                class_store.append(current)
                
                # if that ancestor is root identify indices and node labels to remove 
                if (anc == 'root') or (remove == True):
                    index_store = list(set(index_store)|set(res)) # find union                     
                    if (verbose == True):
                        print("\t - remove: ", current)
                    
                # feedback 
                elif node_size == 0:
                     if (verbose == True):
                        print("\t - previously renamed")
                else:
                    if (verbose == True):
                        print("\t - rename: ", current, " to ", anc)

    return(labels_out, index_store,class_store)

#(labels_out, index_store,class_store) = relabel_samples_by_count_per_node(graph, 
#                                                                          labels, 
#                                                                          10, 
#                                                                          verbose = True, 
#                                                                         remove = True) 


# In[7]:


## notes

# plot example graph
#plt.tight_layout()
#nx.draw_networkx(graph, arrows=True)

# check graph is directed acyclic
#nx.is_directed_acyclic_graph(graph)

# shortest path from root to node
#nx.shortest_path(graph, 'root', 'a1') 

#nx.dfs_successors(graph, 'a')


# In[8]:


# check root node is named 'root'
def check_root(graph):
    '''returns TRUE if basal node is named root'''
    value = 0;
    root_node = graph.has_node('root')
    if (root_node == 1):
        upstream = nx.ancestors(graph,'root')
        if (len(upstream) == 0 ):
            value = 1 
    return(value)


# In[9]:


# find ancestors 
def find_ancestors(graph, target):
    '''find immediate ancestor of target node'''
    all_nodes = nx.dfs_predecessors(graph, 'root')
    value = all_nodes.get(target)
    return(value)


# In[10]:


# find descendants 
def find_decendants(graph, target):
    '''find immediate descendants of target node'''
    all_nodes = nx.dfs_successors(graph, target, 0)
    return(all_nodes.get(target))


# In[11]:


# check if node is tip
def is_tip(graph, target):
    '''returns 1 if node is tip'''
    all_nodes = nx.dfs_successors(graph, target, 0)
    value = 0
    if (len(all_nodes) == 0 ):
        value = 1
    return(value)


# In[12]:


# find all nodes which have > descendants ## maybe redundant
def find_working_nodes(graph):
    '''returns (nodes>1 decendants, nodes<=1 decendants)'''
    nodes = nx.topological_sort(graph) # graph.nodes()
    working = []
    dropped = []
    for i in nodes:
        dec = find_decendants(graph, i)
        if  (dec) and (len(dec)>1): #
            working.append(i)
        else:
            dropped.append(i)
    return(working, dropped)        


# In[13]:


# identify path from sample to root
def root_dist(graph, target):
    '''identify nodes on path from target to root'''
    path = nx.shortest_path(graph, 'root', target)
    return(path)


# In[14]:


# gather class labels for subsetting
def get_class_labels(graph, target):
    dec = nx.dfs_successors(graph, target)
    out = []
    for i in dec:
        sub = dec[i] 
        for j in sub:
            out.append(j)
    return(out)


# In[15]:


# subset data on node
def subset_on_node(graph, target, labels, features="", excludeSelf = False):
    '''subset class labels/features on target node. Returns subsetted data and indices.'''
    
    # find sub-labels from DAG
    target_labels = get_class_labels(graph, target)
    
    # append self
    if excludeSelf == False:
        target_labels.append(target) 
    
    # get indices of labels/sub_labels
    indices = np.array(np.nonzero(np.in1d(labels, target_labels))[0])
    subset_labels = np.array(labels)[indices]
    
    if isinstance(features, pd.DataFrame):
        subset_features = features.iloc[indices]
        return(subset_labels, indices, subset_features)
    else:
        return(subset_labels, indices) 

#print(subset_on_node(graph, 'Africa', train_labels, train_features))


# In[16]:


# rename_labels_by_node, return renamed labels 
def rename_labels_by_node(graph, target, labels, byDecendants = True):
    '''output labels where descendants are renamed as the target node or by decendants'''
    sub_labels = find_decendants(graph, target)    
    target_labels = copy.copy(labels).astype('U256') # to ensure there are sufficient characters for longer reassigned labels 
    if (byDecendants == False):
        for i in sub_labels:
            target_labels[target_labels == i] = target # rename immediate node
            suc = nx.dfs_successors(graph, i)
            for j in suc: 
                for k in suc[j]:
                    target_labels[target_labels == k] = target # rename distant nodes
    else:
        for i in sub_labels:
            suc = nx.dfs_successors(graph, i)
            for j in suc: 
                for k in suc[j]:
                    target_labels[target_labels == k] = i # rename distant nodes

    return(target_labels) 

#test = rename_labels_by_node(graph, 'root', copy.copy(labels), byDecendants = True)
#print(labels)
#print(test)


# In[17]:


# apply classifier to each internal node in the DAG
# exclude feature and labels which are not downstream from the current node
def fit_hierarchical_classifier(graph, labels, features, classifier, feature_selection = "", 
                                         subsampler = "", plot = 0, verbose = False):
    '''Fit individual multi-class classifier per internal node in the hierachy. Samples
    are renamed to match the classes appropriate for the internal node. Non-branching nodes 
    and tips are ignored. Pipelines and AutoML (TPOT) classifiers can be passed. 
    Undersampling/Oversampling can also be included as a seperate step outside of a pipeline
    to avoid conflicts with TPOT and deepcopy. Returns a dictionary of classifier with keys 
    named after the appropriate node'''

    labels = copy.copy(labels)
    
    # initialise storage dictionary
    storage = {}

    # process in topographical order
    nodes = nx.topological_sort(graph) # graph.nodes()
    for current in nodes:

        # initialise classifier 
        clf_global = classifier
        
        if verbose is True:
            print(" - starting", current)

        # check if node is tip or has only one downstream node
        dec = find_decendants(graph, current)

        isTip = is_tip(graph, current)
        if ( (dec is None) or (len(dec)==1) or (isTip == 1) ):
            if verbose is True:
                    print(' - tip or non-branching node')
        else:
            if verbose is True:
                    print(" - processing: " + str(current))
           
            # subset data 
            (sub_labels, sub_indices, sub_features) = subset_on_node(graph, current, copy.copy(labels), features, excludeSelf = True)
            
            # subset sample if subsampler provided - NOTE: deepcopy does not work well with Pipelines+TPOT
            if not subsampler == "":
                (sub_features, sub_labels) = subsampler.fit_resample(copy.copy(sub_features), copy.copy(sub_labels))
            
            # rename subsampled labels for current level in hierarchy
            sub_labels = rename_labels_by_node(graph, current, sub_labels, byDecendants = True)
            
            # feature selection [optional]
            FS_model = ""
            if not feature_selection == "":
                
                FS_model = copy.deepcopy(feature_selection)
                sub_features = FS_model.fit_transform(copy.copy(sub_features), copy.copy(sub_labels))
                
                # plot feature selection [optional]
                if plot == 1:
                    FS_model.plot()    
                
            # initialise classifier
            clf = copy.copy(clf_global)

            # fit to sub data
            clf.fit(sub_features, sub_labels)
            
            # check if classifier is TPOT classifier - extract pipeline
            if str(type(classifier)).find('tpot.tpot.TPOTClassifier') >=0:
                clf = clf.fitted_pipeline_
                            
            # get classes in model 
            classes = ""
            if str(type(clf)).find('imblearn.pipeline.Pipeline') >=0:
                    last = models[len(clf)-1] ### potential error here
                    if str(type(last)).find('tpot.tpot.TPOTClassifier') >=0:
                        classes = str(last.fitted_pipeline_.classes_)
                    else:
                        classes = str(clf.classes_)        
            else:
                 classes = str(clf.classes_)

            if verbose is True:
                    print("\t - classes in model:" + classes)
            
           # return(clf)
            
            # make dictionary of relevant data - ensure they are copies and not refs (these would update with each loop)
            store = {
                'indices' : copy.copy(sub_indices),
                'classifier' : copy.deepcopy(clf),
            }
            
            # store subsampler and resamler models if present
            if not FS_model == "":
                store.update ( {'FS_model' : copy.deepcopy(FS_model) })
            if not subsampler == "":
                store.update ( {'resampler_model' : copy.deepcopy(subsampler) })

            # save it to the storage dict
            storage.update( { copy.copy(current) : copy.deepcopy(store) } )
            
    return(storage)

# fit classifier
#classifier = RandomForestClassifier()
#sampler = RandomUnderSampler(random_state = seed)
#featsel = chi2_FS(n = 1000)
#models = fit_hierarchical_classifier(graph, train_labels, train_features, classifier, featsel, sampler, plot = 1)

# fit classifier
#classifier = RandomForestClassifier()
#sampler = RandomUnderSampler(random_state = seed)
#sampler = HierarchicalSampler(resampler = RandomUnderSampler(), random_state = seed, graph = graph)
#models = fit_hierarchical_classifier(graph, train_labels, train_features, classifier, 
#                                     subsampler = sampler, verbose = True, hier_resample = True)


# In[18]:


def generate_hierarchical_predicted_proba(graph, features, model_store):
    '''Generate predicted probability per sample per node in hierarchy.
    Output a table where each row is a sample and each column is a node in 
    the hierarchy.'''

    # generate nodes in topographical order
    nodes = nx.topological_sort(graph)

    # convert nodes from iter object to array
    node_num = 0 # number of nodes 
    node_names = [] # name of nodes
    for i in nodes:
        node_num = node_num+1
        node_names.append(i)
    
    # create empty output table (col = node, row = isolate)
    empty_cf = np.empty( shape = (features.shape[0], node_num) ) 
    empty_cf[:] = np.nan
    classifications = pd.DataFrame(empty_cf, columns=node_names, index = features.index.values)
    classifications = classifications.drop(['root'], axis=1)
    
    # set features
    current_features = features
    
    # process nodes topographically
    for current in node_names:

        print(" - processing node:", current)

        # get model from storage
        clf = None
        if current in model_store: 
            clf = model_store[current]['classifier']
          
        # check if node is tip 
        isTip = is_tip(graph, current)

        # find downstream nodes (if any)
        dec = find_decendants(graph, current)
        print("\t - descendants:" + str(dec))
        
        ### check for existing classification model ###

        # process node according to type
        if (  isTip == 1 ):
            print('\t - tip')
        elif ( (dec is None) or (len(dec)==1) ):
            
            print('\t - non-branching node')
            
            # find decendant and pass values of current nodes to downstream node
            classifications[dec[0]] = copy.copy(classifications[current])
        
        elif ( current_features is None ):
            print('\t - no samples for classification')    
        elif ( clf is None ):
            print('\t - ERROR: no classifier found for node - ', current)
        else:

            print('\t - classifying node:', current)
          
            # get classes and predict labels
            classes = clf.classes_
            pred = clf.predict_proba(current_features)
            print("\t - classes in model:" + str(classes))

            # create pd.dataframe for ease of access 
            pred_df = pd.DataFrame(pred, columns=classes, index = current_features.index.values)
                        
            # store predicted probability per sample
            for ind in current_features.index.values:
                for col in classes:
                    val = pred_df.loc[ind, col]
                    classifications.loc[ind, col] = val
            
    # return table
    return(classifications)
    
            
#(train_classification_table) = generate_hierarchical_predicted_proba(graph, features, models)
#print(train_classification_table)


# In[19]:


# run hierachical prediction with threshold cutoff

# 5 strategies
# threshold - if proba > theshold then store (allows 0 theshold)
# max -> follow max value per node
# max stop -> follow max if > previous node
# max above but with threshold 

# currently cannot provide multiple classifications for the same sample although 
# this is possible if threshold < 0.05 and thresh is used as mode 

# maybe worth just creating classification table and then generating classifications afterwards
# Althernatively allow multi-classificatins in classification array aand modify subsequent scripts 
# to allo for this 

def classify_samples_in_hierarchy_old(graph, features, model_store, mode='max', threshold=0, verbose = False):

    # check input feature index does not contain duplicates
    if not features.index.is_unique:
        df_dup = features.index.duplicated()
        df_dup_ind = features.index.values[df_dup]
        raise Exception(" - ERROR: Dupicate indices in input features: ", df_dup_ind)
    
    # check threshold is a number
    if type(threshold) != int and type(threshold) != float:
        raise ValueError("Invalid threshold value. Expected numeric.")
        
    # load presets for mode of operation
    mode_types = ['max', 'max_prev', 'thresh']
    if mode not in mode_types:
        raise ValueError("Invalid mode type. Expected one of: %s" % mode_types)
    
    # feedback
    if verbose is True:
        if mode == 'max':
            print(" - traversing the DAG using max probability per node, threshold >", threshold)
        elif mode == 'max_prev':
            print(" - traversing the DAG using max probability per node if > ancerstor node probability, threshold >", threshold)
        else:
             print(" - traversing the DAG using probability values > threshold value of", threshold)
    
    # set storage variables 
    classification_store = {}
    feature_store = {'root' : copy.copy(features)} # assign sample set to root

    # generate nodes in topographical order
    nodes = nx.topological_sort(graph)

    # convert nodes from iter object to array
    node_num = 0 # number of nodes 
    node_names = [] # name of nodes
    for i in nodes:
        node_num = node_num+1
        node_names.append(i)

    # create empty output table (col = node, row = isolate)
    empty_cf = np.empty( shape = (features.shape[0], node_num) ) 
    empty_cf[:] = np.nan
    classifications = pd.DataFrame(empty_cf, columns=node_names, index = features.index.values)
    classifications = classifications.drop(['root'], axis=1)
    
    # process nodes topographically
    for current in nx.topological_sort(graph):

        if verbose is True:
                print(" - processing node:", current)

        # collect current features - this is misleading #### should be labelled as current samples ####
        ### note - samples inlcuded changes throughout running - features SHOULDNT
        current_features = None
        if feature_store.get(current) is not None:
            current_features = feature_store[current]
            
        # get model from storage
        clf = None
        if current in model_store: 
            clf = model_store[current]['classifier']
          
        # check if node is tip 
        isTip = is_tip(graph, current)

        # find downstream nodes (if any)
        dec = find_decendants(graph, current)
        if verbose is True:
                print("\t - descendants:" + str(dec))
        
        ### check for existing classification model ###

        # process node according to type
        if (  isTip == 1 ):
            if verbose is True:
                    print('\t - tip')
        elif ( current_features is None ):
            if verbose is True:
                    print('\t - no samples for classification')
        elif ( (dec is None) or (len(dec)==1) ):
            
            if verbose is True:
                    print('\t - non-branching node')
            
            # push features to descendant
            feature_store.update( { copy.copy(dec[0]) : copy.copy(current_features) } )
            classification_store.update( { copy.copy(dec[0]) : copy.copy(current_features.index.values) } ) 
            
            # remove from this node
            classification_store.pop(current)
            feature_store.pop(current)
            
        elif ( clf is None ):
            if verbose is True:
                    print('\t - ERROR: no classifier found')
        else:

            if verbose is True:
                    print('\t - classifying node:', current)
          
            # get classes
            classes = clf.classes_
            
            # run feature selection if required 
            temp_features = copy.copy(current_features) # passed to model BUT not passed deeper into hierarchy as features included may change
            if 'FS_model' in model_store['root'].keys():
                FS_model = model_store[current]['FS_model']
                temp_features = FS_model.transform(copy.copy(current_features))
                
            # predict labels
            pred = clf.predict_proba(temp_features)
            if verbose is True:
                    print("\t - classes in model:" + str(classes))

            # create pd.dataframe for ease of access 
            pred_df = pd.DataFrame(pred, columns=classes, index = current_features.index.values)
                        
            # store predicted probability per sample
            for ind in current_features.index.values:
                for col in classes:
                    val = pred_df.loc[ind, col]
                    classifications.loc[ind, col] = val

            # find max value/class per sample
            max_vals = pred_df.max(axis=1)
            max_class = pred_df.idxmax(axis=1)

            # generate previous probabilities from upstream node, make empty matrix if current node is root
            if current == 'root':
                prev_proba = np.zeros([len(max_vals)], dtype=float)
            else:
                prev_proba = classifications.loc[current_features.index.values, current]

            # assign samples (features) to subsequet nodes based on the current mode of operation
            assigned = [] # store assigned indices
            for cls in classes:
                 
                # subset data by current class
                sub_vals = pred_df[cls]
                
                # subset method based on mode
                sub_ind = []
                if mode == 'max': # only store if it the maximum proba value
                    sub_ind = sub_vals[ (sub_vals>=threshold) & (sub_vals==max_vals) ].index.values
                elif mode == 'max_prev': # only store if it the maximum proba value is > threshold AND greater than previous node
                    sub_ind = sub_vals[ (sub_vals>=threshold) & (sub_vals==max_vals) & (sub_vals>prev_proba) ].index.values
                else: # find index of values > than threshold
                    sub_ind = sub_vals[ (pred_df[cls]>=threshold)].index.values 

                # identify number of samples to be moved to downstream node
                l_moved = len(sub_ind)
                if verbose is True:
                        print("\t\t - ", l_moved, "samples assigned to", cls)

                # store to feature_store unless empty
                if l_moved > 0:

                    # store assigned indices
                    assigned = np.append(assigned,sub_ind)

                    # create subset of features from indices
                    sub_features = current_features.loc[sub_ind]

                    # update feature store and value store variables
                    feature_store.update( { copy.copy(cls) : copy.copy(sub_features) } )
                    classification_store.update( { copy.copy(cls) : copy.copy(sub_ind) } ) 

            # store unassigned features/samples to the current node
            unassigned = []
            for ass in current_features.index.values:
                test = np.where(assigned == ass)
                if test[0].size==0:
                    unassigned.append(ass)

            # identify number of samples to be fixed to current node
            l_fixed = len(unassigned)
            if verbose is True:
                    print("\t\t - ", l_fixed, "samples fixed at", current)

            #remove data at current node [if any]
            feature_store.pop(current)
            if current in classification_store:
                classification_store.pop(current)

            # store to feature_store unless empty
            if l_fixed > 0:

                # subset features
                sub_features = current_features.loc[unassigned]        

                # update feature store and value store variables
                feature_store.update( { copy.copy(current) : copy.copy(sub_features) } )
                classification_store.update( { copy.copy(current) : copy.copy(unassigned) } )
                
    if (mode != 'thresh') or (threshold>=0.5):
        
        # generate classifications per feature
        output_classifications = pd.DataFrame(columns = ['classification']) 
        for i in classification_store:
            for j in classification_store[i]:
                temp_df = pd.DataFrame([i], columns = ['classification'], index = [j]) 
                output_classifications = output_classifications.append(temp_df)
        
        # check for multiple classifications assigned to single isolates 
        reind = 1;
        if not output_classifications.index.is_unique:
            df_dup = output_classifications.index.duplicated()
            df_dup_ind = output_classifications.index.values[df_dup]
            reind = 0
            print(" - WARNING: Samples have been assigned to multiple classes, try increasing threshold (>0.5) to prevent this behaviour. Classes: ", df_dup_ind)
        
        # reindex on input if all sample assigned a single class
        if reind == 1:
            output_classifications = output_classifications.reindex(features.index.values) # reindex to get input order of features    
        
        #return classification table - all probability values explored and final classification per sample
        return(classifications, output_classifications)
    
    else:
        
        return(classifications, None)    
            
#test_run()
#(train_classification_table, train_classifications) = classify_samples_in_hierarchy(graph, train_features, 
#                                                                       models,
#                                                                       mode = 'max', 
#                                                                       threshold = 0.5) 
#print(train_classification_table)#
#print(train_classifications)


# In[ ]:


# run hierachical prediction with threshold cutoff

def classify_samples_in_hierarchy(graph, features, model_store, mode='max', threshold=0, verbose = False):
    '''Assign a classification per sample per node in the hierachy. Allows for multiple assignment 
    strategies to applied. 
    
    Modes: 
    threshold - classify if predict_proba > threshold then store [default: zero threshold]
    max - classify sample follow max value per node
    max_prev - same as above but stop if probability at child node is lower than parent node.
    NOTE: threshold values can be applied to any of the strategies above
    
    Threshold (greater than):
    'float' - value between 0-100.
    'adaptive' - value calculated on number of child classes - 100/no_classes

    Note threshold allows multiple classifications per sample but this is currently not handled by
    the script. Errors possible if mode == threshold and threshold =< 0.05. 
    '''
    
    # check input feature index does not contain duplicates
    if not features.index.is_unique:
        df_dup = features.index.duplicated()
        df_dup_ind = features.index.values[df_dup]
        raise Exception(" - ERROR: Dupicate indices in input features: ", df_dup_ind)
    
    # check threshold is a number
    if isinstance(threshold, str):
        if threshold != "adaptive":
            raise ValueError("Invalid threshold value (string). Expected float between 0-1 or 'adaptive'.")
    elif isinstance(threshold, int) or isinstance(threshold, float):
        if threshold >1 and threshold <0:
            raise ValueError("Invalid threshold value (integer/float). Expected float between 0-1 or 'adaptive'.")
    else:
        raise ValueError("Invalid threshold value. Expected float between 0-1 or 'adaptive'.")
        
    # load presets for mode of operation
    mode_types = ['max', 'max_prev', 'thresh']
    if mode not in mode_types:
        raise ValueError("Invalid mode type. Expected one of: %s" % mode_types)
    
    # feedback
    if verbose is True:
        if mode == 'max':
            print(" - traversing the DAG using max probability per node, threshold >", threshold)
        elif mode == 'max_prev':
            print(" - traversing the DAG using max probability per node if > ancerstor node probability, threshold >", threshold)
        elif mode == "threshold":
             print(" - traversing the DAG using probability values > threshold value of", threshold)
        else:
            raise Exception(" - ERROR: mode type not recognised, expects max, max_prev or threshold")
            
    classification_store = {}
    feature_store = {'root' : copy.copy(features)} # assign sample set to root

    # generate nodes in topographical order
    nodes = nx.topological_sort(graph)

    # convert nodes from iter object to array
    node_num = 0 # number of nodes 
    node_names = ["classification"] # name of nodes
    for i in nodes:
        node_num = node_num+1
        node_names.append(i)

    # create empty output table (col = node, row = isolate) - add one col for label variable
    empty_cf = np.empty( shape = (features.shape[0], node_num+1) ) 
    empty_cf[:] = np.nan
    classifications = pd.DataFrame(empty_cf, columns=node_names, index = features.index.values)
    classifications = classifications.drop(['root'], axis=1) # drop root
    
    # process nodes topographically
    for current in nx.topological_sort(graph):

        if verbose is True:
                print(" - processing node:", current)

        # collect current features - this is misleading #### should be labelled as current samples ####
        ### note - samples inlcuded changes throughout running - features SHOULDNT
        current_features = None
        if feature_store.get(current) is not None:
            current_features = feature_store[current]
            
        # get model from storage
        clf = None
        if current in model_store: 
            clf = model_store[current]['classifier']
          
        # check if node is tip 
        isTip = is_tip(graph, current)

        # find downstream nodes (if any)
        dec = find_decendants(graph, current)
        if verbose is True:
                print("\t - descendants:" + str(dec))
        
        ### check for existing classification model ###

        # process node according to type
        if (  isTip == 1 ):
            if verbose is True:
                    print('\t - tip')
        elif ( current_features is None ):
            if verbose is True:
                    print('\t - no samples for classification')
        elif ( (dec is None) or (len(dec)==1) ):
            
            if verbose is True:
                    print('\t - non-branching node')
            
            # push features to descendant
            feature_store.update( { copy.copy(dec[0]) : copy.copy(current_features) } )
            classification_store.update( { copy.copy(dec[0]) : copy.copy(current_features.index.values) } ) 
            
            # remove from this node
            classification_store.pop(current)
            feature_store.pop(current)
            
        elif ( clf is None ):
            if verbose is True:
                    print('\t - ERROR: no classifier found')
        else:

            if verbose is True:
                    print('\t - classifying node:', current)
          
            # get classes
            classes = clf.classes_
            
            # set threshold 
            if threshold == "adaptive":
                threshold = 1/len(classes)
                if verbose:
                    print("\t - adaptive threshold set: ", threshold)
            
            # run feature selection if required 
            temp_features = copy.copy(current_features) # passed to model BUT not passed deeper into hierarchy as features included may change
            if 'FS_model' in model_store['root'].keys():
                FS_model = model_store[current]['FS_model']
                temp_features = FS_model.transform(copy.copy(current_features))
                
            # predict labels
            pred = clf.predict_proba(temp_features)
            if verbose is True:
                    print("\t - classes in model:" + str(classes))

            # create pd.dataframe for ease of access 
            pred_df = pd.DataFrame(pred, columns=classes, index = current_features.index.values)
                        
            # store predicted probability per sample
            for ind in current_features.index.values:
                for col in classes:
                    val = pred_df.loc[ind, col]
                    classifications.loc[ind, col] = val

            # find max value/class per sample
            max_vals = pred_df.max(axis=1)
            max_class = pred_df.idxmax(axis=1)

            # generate previous probabilities from upstream node, make empty matrix if current node is root
            if current == 'root':
                prev_proba = np.zeros([len(max_vals)], dtype=float)
            else:
                prev_proba = classifications.loc[current_features.index.values, current]

            # assign samples (features) to subsequet nodes based on the current mode of operation
            assigned = [] # store assigned indices
            for cls in classes:
                 
                # subset data by current class
                sub_vals = pred_df[cls]
                
                # subset method based on mode
                sub_ind = []
                if mode == 'max': # only store if it the maximum proba value
                    sub_ind = sub_vals[ (sub_vals>threshold) & (sub_vals==max_vals) ].index.values
                elif mode == 'max_prev': # only store if it the maximum proba value is > threshold AND greater than previous node
                    sub_ind = sub_vals[ (sub_vals>threshold) & (sub_vals==max_vals) & (sub_vals>prev_proba) ].index.values
                else: # find index of values > than threshold
                    sub_ind = sub_vals[ (pred_df[cls]>threshold)].index.values 
                
                # identify number of samples to be moved to downstream node
                l_moved = len(sub_ind)
                if verbose is True:
                        print("\t\t - ", l_moved, "samples assigned to", cls)

                # store to feature_store unless empty
                if l_moved > 0:

                    # store assigned indices
                    assigned = np.append(assigned,sub_ind)

                    # create subset of features from indices
                    sub_features = current_features.loc[sub_ind]

                    # update feature store and value store variables
                    feature_store.update( { copy.copy(cls) : copy.copy(sub_features) } )
                    classification_store.update( { copy.copy(cls) : copy.copy(sub_ind) } ) 

            # store unassigned features/samples to the current node
            unassigned = []
            for ass in current_features.index.values:
                test = np.where(assigned == ass)
                if test[0].size==0:
                    unassigned.append(ass)

            # identify number of samples to be fixed to current node
            l_fixed = len(unassigned)
            if verbose is True:
                    print("\t\t - ", l_fixed, "samples fixed at", current)

            #remove data at current node [if any]
            feature_store.pop(current)
            if current in classification_store:
                classification_store.pop(current)

            # store to feature_store unless empty
            if l_fixed > 0:

                # subset features
                sub_features = current_features.loc[unassigned]        

                # update feature store and value store variables
                feature_store.update( { copy.copy(current) : copy.copy(sub_features) } )
                classification_store.update( { copy.copy(current) : copy.copy(unassigned) } )
                
    if (mode != 'thresh') or (threshold>=0.5):
        
        # generate classifications per feature
        output_classifications = pd.DataFrame(columns = ['classification']) 
        for i in classification_store:
            for j in classification_store[i]:
                temp_df = pd.DataFrame([i], columns = ['classification'], index = [j]) 
                output_classifications = output_classifications.append(temp_df)
                
                # add to label column in classification_table
                classifications.loc[j,"classification"] = i
        
        # check for multiple classifications assigned to single isolates 
        reind = 1;
        if not output_classifications.index.is_unique:
            df_dup = output_classifications.index.duplicated()
            df_dup_ind = output_classifications.index.values[df_dup]
            reind = 0
            print(" - WARNING: Samples have been assigned to multiple classes, try increasing threshold (>0.5) to prevent this behaviour. Classes: ", df_dup_ind)
        
        # reindex on input if all sample assigned a single class
        if reind == 1:
            output_classifications = output_classifications.reindex(features.index.values) # reindex to get input order of features    
        
        #return classification table - all probability values explored and final classification per sample
        return(classifications, output_classifications)
    
    else:
        
        return(classifications, None)    
            
#test_run()
#(train_classification_table, train_classifications) = classify_samples_in_hierarchy(graph, train_features, 
#                                                                       models,
#                                                                       mode = 'max', 
#                                                                       threshold = 0.5) 
#print(train_classification_table)#
#print(train_classifications)


# In[20]:


def hierarchical_confusion_matrix_old (graph, classifications, labels):
    '''Create confusion matrix for heirachical classifier
    (Row = Actual, Column = Prediction).'''

    node_order = nx.lexicographical_topological_sort(graph)

    # check if classifications include root
    inc_root = 0
    if 'root' in classifications.values:
        inc_root = 1
        
    # create node list (remove_root) 
    node_list = []
    for node in node_order:
        if (node == "root") and (inc_root == 0):
            pass
        else:
            node_list.append(node)

    # number of nodes
    node_n = len(node_list)

    # make matrix of zeros n x n size in node order 
    out_mat = np.zeros([node_n, node_n], dtype=int)

    # store actual to predicted classifications
    for i,true in enumerate(labels):
 
        # find prediction
        pred = classifications.iloc[i].values[0]

        # find node_index of true/pred
        ind_true = node_list.index(true)
        ind_pred = node_list.index(pred)
        
        # increment count in relevant cell (row = actual, column = pred)
        out_mat[ind_true, ind_pred] += 1 

    # convert matrix to dataframe and label row/column indices
    df_out = pd.DataFrame(out_mat, columns = node_list, index = node_list, copy = True)
    return(df_out)

#confusion_mat = hierarchical_confusion_matrix_old (graph, classifications, train_labels)
#print(confusion_mat)


# In[21]:


def hierarchical_confusion_matrix(graph, classifications, labels, 
                                          limit_depth = 0, match_depth = False,
                                          fix_depth = 0):
    '''Create confusion matrix for heirachical classifier
    (Row = Actual, Column = Prediction).
    - limit_depth (integer) limit depth of graph from node (root = 1).
    - match_depth (integer) convert all predicted labels to match depth of current label'''
    
    # check inputs
    check = 0;
    if not ((match_depth == True) or (match_depth == False)):
        print (" - ERROR: match_depth should be True/False")
        check += 1
    if not (isinstance(limit_depth, int)):
        print (" - ERROR: limit_depth should be an integer")
        check += 1
    if not (isinstance(fix_depth, int)):
        print (" - ERROR: fix_depth should be an integer")
        check += 1
    if (fix_depth > 0) and (limit_depth > 0):
        print(" - ERROR: choose either fix_depth or limit_depth")
        check += 1        
    if check == 1:
        return
    
    # match depth of depth is fixed to one level
    if fix_depth > 0:
        match_depth = True
        limit_depth = fix_depth
    
    # get node order
    node_order = nx.lexicographical_topological_sort(graph)

    # check if classifications include root
    inc_root = 0
    if 'root' in classifications.values:
        inc_root = 1
        
    # create node list (remove_root) 
    node_list = []
    for node in node_order:
        node_depth = len(root_dist(graph, node))
        if fix_depth > 0:
            if node_depth == fix_depth:
                node_list.append(node)
        else:
            if (node == "root") and (inc_root == 0):
                pass
            elif (limit_depth == 0) or (node_depth <= limit_depth):
                node_list.append(node)
  
    # number of nodes
    node_n = len(node_list)

    # make matrix of zeros n x n size in node order 
    out_mat = np.zeros([node_n, node_n], dtype=int)

    # store actual to predicted classifications
    for i,true in enumerate(labels):
        
        # find prediction
        pred = classifications.iloc[i].values[0]
        
        # get node depths
        node_depth_true = len(root_dist(graph, true))
        node_depth_pred = len(root_dist(graph, pred))
           
        # adjust true prediction to desired node depth [optional]
        if limit_depth > 0: 
            while node_depth_true > limit_depth:
                true = find_ancestors(graph, true)
                node_depth_true = node_depth_true - 1
                
        # adjust prediction according to node depth or match depth of true value [optional]
        if (limit_depth>0) or (match_depth is True):
            limit_pred = limit_depth
            if match_depth is True:
                limit_pred = node_depth_true
            while node_depth_pred > limit_pred:
                pred = find_ancestors(graph, pred)
                node_depth_pred = node_depth_pred - 1
                
        # only store if true and predicted value are at fixed node level [optional]
        if (fix_depth == 0) or ( (fix_depth > 0) and (node_depth_true == node_depth_pred) and (node_depth_true == fix_depth) ):

            # find node_index of true/pred
            ind_true = node_list.index(true)
            ind_pred = node_list.index(pred)

            # increment count in relevant cell (row = actual, column = pred)
            out_mat[ind_true, ind_pred] += 1 

    # convert matrix to dataframe and label row/column indices
    df_out = pd.DataFrame(out_mat, columns = node_list, index = node_list, copy = True)
    return(df_out)

#confusion_mat = hierarchical_confusion_matrix (graph, train_classifications, train_labels, 
#                                              fix_depth = 4)
#print(confusion_mat)
#confusion_mat.to_csv('test_conf_updated.tsv', sep = "\t", header = True, index = True)


# In[22]:


def per_node_summary_stats_old(graph, labels, features, model_storage, verbose = False):
    '''Generate summary statistics for each class on a per node basis.
    This uses only data relevant to the nodes, renaming class levels where 
    appropriate.'''

    # generate empty output tables
    clf_reports = {}
    per_node = pd.DataFrame(columns=['node', 'model-type', 'root-dist', 'acc', 'bal-acc', 
                                     'macroP', 'macroR', 'macroF1', 'weightedP', 
                                     'weightedR', 'weightedF1', 'support'])
    per_class = pd.DataFrame(columns=['node', 'parent-node',  'root-dist', 'precision', 
                                      'recall', 'f1-score', 'support'])
    
    # set pattern for pipeline extraction
    pat = r'[\[\"](.*?)[\"\]]'

    # process pre-generated classifiers in topographical order
    nodes = nx.lexicographical_topological_sort(graph)
    for current in nodes:

       # retrieve classifier for node
        clf = None
        if current in model_storage:
            clf = model_storage[current]['classifier']

        # process if node has existing classifier
        if not clf is None:
            
            # get model type
            model_type = str(type(clf)).split(".")[-1][:-2]
            
            # if model type is a pipeline extract and format steps as strings for output
            if model_type == 'Pipeline':
                steps = str(clf.named_steps.keys())
                matches = re.findall(pat, steps)
                model_type = ','.join(map(str, matches))
                model_type = model_type.replace(', ', ':')
                model_type = model_type.replace('\'', '')
            
            if verbose == True:
                print("classes in model:" + str(clf.classes_))
            
            # subset data 
            (sub_labels, sub_indices, sub_features) = subset_on_node(graph, current, labels, features, excludeSelf=True)
            
            if verbose == True:
                print("A", sub_labels)
            
            # rename labels for current level in hierachy
            sub_labels = rename_labels_by_node(graph, current, sub_labels, byDecendants = True)
            
            if verbose == True:
                print("B", sub_labels)
            
            # predict labels 
            pred = clf.predict(sub_features)

            # generate summary stats 
            clf_report = classification_report(sub_labels, pred, output_dict=True)

            # balanced accuracy
            bal_acc = balanced_accuracy_score(sub_labels, pred)

            # get node depth from root 
            dist = len(root_dist(graph, current)) # don't adjust for root

            # extract comparable stats per node
            temp_sum =  {}   
            temp_sum.update({'node':copy.copy(current)})
            temp_sum.update({'model-type':copy.copy(model_type)})
            temp_sum.update({'root-dist':copy.copy(dist)-1})
            temp_sum.update({'bal-acc':copy.copy(bal_acc)})
            for i in clf_report:

                if i == "accuracy":
                    temp_sum.update({'acc':copy.copy(clf_report[i])})
                elif i == "macro avg":
                    temp_sum.update({'macroP':copy.copy(clf_report[i]['precision'])})
                    temp_sum.update({'macroR':copy.copy(clf_report[i]['recall'])})
                    temp_sum.update({'macroF1':copy.copy(clf_report[i]['f1-score'])})
                    temp_sum.update({'support':copy.copy(clf_report[i]['support'])})
                elif i == "weighted avg":
                    temp_sum.update({'weightedP':copy.copy(clf_report[i]['precision'])})
                    temp_sum.update({'weightedR':copy.copy(clf_report[i]['recall'])})
                    temp_sum.update({'weightedF1':copy.copy(clf_report[i]['f1-score'])})
                else:

                    # create and populate per_class dataframe 
                    temp_pc =  {}   

                    # variables 
                    temp_pc.update({'node':copy.copy(i)})
                    temp_pc.update({'parent-node':copy.copy(current)})
                    temp_pc.update({'root-dist':copy.copy(dist)})
                    temp_pc.update({'precision':copy.copy(clf_report[i]['precision']) })
                    temp_pc.update({'recall':copy.copy(clf_report[i]['recall'])})
                    temp_pc.update({'f1-score':copy.copy(clf_report[i]['f1-score'])})
                    temp_pc.update({'support':copy.copy(clf_report[i]['support'])})

                    # append to per class
                    per_class = per_class.append(temp_pc, ignore_index = True)

            # append to overall summary stats
            per_node = per_node.append(temp_sum, ignore_index= True)  

            # store individually
            temp_stats = {'classification_report': clf_report}
            clf_reports.update({current:temp_stats}, copy = True)
    
    # return outputs
    return(per_node, per_class, clf_reports)

#(test_summary_per_node, test_summary_per_class, test_clf_tables) = per_node_summary_stats(graph, test_labels, test_features, models, verbose = False)
#print(test_summary_per_node)
#print(test_summary_per_class)
#print(test_clf_tables)


# In[ ]:


def per_node_summary_stats(graph, labels, features, model_storage, verbose = False):
    '''Generate summary statistics for each class on a per node basis.
    This uses only data relevant to the nodes, renaming class levels where 
    appropriate.'''
    
    # generate empty output tables
    clf_reports = {}
    per_node = pd.DataFrame(columns=['node', 'model-type', 'root-dist', 
                                     'FN', 'FP', 'TN', 'TP','precision','recall','microF1',
                                     'acc', 'bal-acc','cohens-kappa',
                                     'macroP', 'macroR', 'macroF1', 'weightedP', 
                                     'weightedR', 'weightedF1', 'support'])
    per_class = pd.DataFrame(columns=['node', 'parent-node',  'root-dist', 'FN', 'FP', 'TN', 'TP','precision', 
                                      'recall', 'f1-score', 'support'])
    
    # set pattern for pipeline extraction
    pat = r'[\[\"](.*?)[\"\]]'

    # process pre-generated classifiers in topographical order
    nodes = nx.lexicographical_topological_sort(graph)
    for current in nodes:

       # retrieve classifier for node
        clf = None
        if current in model_storage:
            clf = model_storage[current]['classifier']

        # process if node has existing classifier
        if not clf is None:
            
            # get model type
            model_type = str(type(clf)).split(".")[-1][:-2]
            
            # if model type is a pipeline extract and format steps as strings for output
            if model_type == 'Pipeline':
                steps = str(clf.named_steps.keys())
                matches = re.findall(pat, steps)
                model_type = ','.join(map(str, matches))
                model_type = model_type.replace(', ', ':')
                model_type = model_type.replace('\'', '')
            
            model_classes = clf.classes_
            if verbose == True:
                print("classes in model:" + str(model_classes))
            
            # subset data 
            (sub_labels, sub_indices, sub_features) = subset_on_node(graph, current, labels, features, excludeSelf=True)
            
            # check any test samples have relevant metadata to include in node summary (i.e. are of same class 
            # or are a parent in hierarchy)
            if ( sub_features.shape[0] > 0 ):
                
                #if verbose == True:
                #    print("A", sub_labels)

                # rename labels for current level in hierachy
                sub_labels = rename_labels_by_node(graph, current, sub_labels, byDecendants = True)

                #if verbose == True:
                #    print("B", sub_labels)

                # apply features selection if present 
                temp_features = sub_features
                if 'FS_model' in model_storage['root'].keys():
                    FS_model = model_storage[current]['FS_model']
                    temp_features = FS_model.transform(copy.copy(sub_features))

                # predict labels 
                pred = clf.predict(temp_features)

                # generate summary stats 
                clf_report = classification_report(sub_labels, pred, output_dict=True)
                
                # balanced accuracy
                bal_acc = balanced_accuracy_score(sub_labels, pred)
                
                # get cohen kappa score
                cks = cohen_kappa_score(sub_labels, pred, labels = model_classes)
                
                # generate confusion matrix 
                cf_mat = confusion_matrix(sub_labels, pred, labels = model_classes)
                # NB. Actual in rows, pred in cols
                
                # per class
                FP = (cf_mat.sum(axis=0) - np.diag(cf_mat) ).flatten() 
                FN = cf_mat.sum(axis=1) - np.diag(cf_mat)
                TP = np.diag(cf_mat)
                TN = cf_mat.sum() - (FP + FN + TP)
                ACC = (TP+TN)/(TP+FP+FN+TN)
               
                # overall
                FP_sum = FP.sum()
                FN_sum = FN.sum()
                TP_sum = TP.sum()
                TN_sum = TN.sum()
                ACC_sum = (TP_sum+TN_sum)/(TP_sum+FP_sum+FN_sum+TN_sum)
                
                # precision/recall/microF1
                prec_sum = TP_sum / (TP_sum + FP_sum)
                rec_sum = TP_sum / (TP_sum + FN_sum)
                microF1 = (2 * prec_sum * rec_sum) / (prec_sum + rec_sum)
              
                # get node depth from root 
                dist = len(root_dist(graph, current)) # don't adjust for root

                # extract comparable stats per node
                temp_sum =  {}   
                temp_sum.update({'node':copy.copy(current)})
                temp_sum.update({'model-type':copy.copy(model_type)})
                temp_sum.update({'root-dist':copy.copy(dist)-1})
                
                temp_sum.update({'TP':copy.copy(TP_sum)})
                temp_sum.update({'TN':copy.copy(TN_sum)})
                temp_sum.update({'FP':copy.copy(FP_sum)})
                temp_sum.update({'FN':copy.copy(FN_sum)})
                
                temp_sum.update({'precision':copy.copy(prec_sum)})
                temp_sum.update({'recall':copy.copy(rec_sum)})
                temp_sum.update({'microF1':copy.copy(microF1)})
                                
                temp_sum.update({'bal-acc':copy.copy(bal_acc)})
                temp_sum.update({'cohens-kappa':copy.copy(cks)})
                                
                for i in clf_report:
               
                    if i == "accuracy":
                        temp_sum.update({'acc':copy.copy(clf_report[i])})
                    elif i == "macro avg":
                        temp_sum.update({'macroP':copy.copy(clf_report[i]['precision'])})
                        temp_sum.update({'macroR':copy.copy(clf_report[i]['recall'])})
                        temp_sum.update({'macroF1':copy.copy(clf_report[i]['f1-score'])})
                        temp_sum.update({'support':copy.copy(clf_report[i]['support'])})
                    elif i == "weighted avg":
                        temp_sum.update({'weightedP':copy.copy(clf_report[i]['precision'])})
                        temp_sum.update({'weightedR':copy.copy(clf_report[i]['recall'])})
                        temp_sum.update({'weightedF1':copy.copy(clf_report[i]['f1-score'])})
                    else:

                        # create and populate per_class dataframe 
                        temp_pc =  {}   

                        # variables 
                        temp_pc.update({'node':copy.copy(i)})
                        temp_pc.update({'parent-node':copy.copy(current)})
                        temp_pc.update({'root-dist':copy.copy(dist)})
                        
                        # add tp,tn,fp,fn
                        temp_pc.update({'TP':str(TP[model_classes==i][0])})
                        temp_pc.update({'TN':str(TN[model_classes==i][0])})
                        temp_pc.update({'FP':str(FP[model_classes==i][0])})
                        temp_pc.update({'FN':str(FN[model_classes==i][0])})
                        
                        temp_pc.update({'precision':copy.copy(clf_report[i]['precision']) })
                        temp_pc.update({'recall':copy.copy(clf_report[i]['recall'])})
                        temp_pc.update({'f1-score':copy.copy(clf_report[i]['f1-score'])})
                        temp_pc.update({'support':copy.copy(clf_report[i]['support'])})
                        

                        # append to per class
                        per_class = per_class.append(temp_pc, ignore_index = True)

                # append to overall summary stats
                per_node = per_node.append(temp_sum, ignore_index= True)  

                # store individually
                temp_stats = {'classification_report': clf_report}
                clf_reports.update({current:temp_stats}, copy = True)
            
            else:
                
                # add empty values to table
                dist = len(root_dist(graph, current)) - 1 
                empty_line = {"node":current, "model-type":model_type, "root-dist":dist, 
                              'FN':0, 'FP':0, 'TN':0, 'TP':0, 'precision':0.0, 'recall':0.0, 'microF1':0.0,
                              "acc":0.0, "bal-acc":0.0, 'cohens-kappa':0.0, "macroP":0.0, 
                              "macroR":0.0, "macroF1":0.0, "weightedP":0.0, 
                              "weightedR":0.0, "weightedF1":0.0, "support":0.0}
                per_node = per_node.append(empty_line, ignore_index = True)
                              
                for cls in clf.classes_:
                    
                    # append to per class
                    empty_line_class = {"node":cls, "parent-node":current, "root-dist":dist+1, 
                                        'FN':0, 'FP':0, 'TN':0, 'TP':0,
                                        "precision":0.0, "recall":0.0, "f1-score":0.0, "support":0.0}
                    per_class = per_class.append(empty_line_class, ignore_index = True)
                    
    # return outputs
    return(per_node, per_class, clf_reports)

#(test_summary_per_node, test_summary_per_class, test_clf_tables) = per_node_summary_stats(graph, test_labels, test_features, model_store, verbose = True)
#print(test_summary_per_node)
#print(test_summary_per_class)
#print(test_clf_tables)


# In[23]:


def summary_statistics_per_class_tips_only(graph, labels, classifications, penalty=False):
    
    '''calculate hierachical per-class summary statistics and micro-macro average. Statistics 
    are calculated as described in "A survey of hierarchical classification across different 
    application domains". This function will return hR, hP, hF1, hAccuracy (ad-hoc) per class and 
    and overall micro- and macro-average'''

    # make output dict
    class_dict = {}
    micro_int = 0;
    micro_pred = 0;
    micro_true = 0;
    
    # make per class table
    per_class = pd.DataFrame(columns=['node', 'root-dist', 'hP',  'hR', 'hF1', 'n'])

    # loop through all classes
    unique_classes = list(set(labels))# ["b1"]
    for test_class in unique_classes:

        # find test path
        test_path = root_dist(graph, test_class)
        test_path.remove('root')
        l_test = len(test_path)

        # initialise variables
        sum_pred = 0;
        sum_true = 0;
        sum_int = 0;

        n = 0;

        for i,true in enumerate(labels):

            pred = classifications.iloc[i].values[0]

            # find path to root and remove root stub (artifical node)
            true_path = root_dist(graph, true)
            true_path.remove('root')
            pred_path = root_dist(graph, pred)
            pred_path.remove('root')

            # check prediction or true class match test_class 
            if( (test_class in true_path) or (test_class in pred_path) ): 

                # increment n if test_class is in true_path
                if (test_class in true_path):
                    n = n+1;

                # find intersection between true_label and test_class paths to identify l_true
                int_true = list(set(true_path)&set(test_path))

                # find length of true/test intersection and pred paths
                l_true = len(int_true)
                l_pred = len(pred_path)

                # find intersection of true/test intersection and pred
                intersection = list(set(int_true)&set(pred_path))
                l_int = len(intersection)

                # adjust length of predicted path if it is > l_true path [optional - don't penalise specific predictions]
                if (penalty == False):
                    if l_pred > l_test:
                        l_pred = l_test

                # aggregate result 
                sum_pred = sum_pred + l_pred;
                sum_true = sum_true + l_true;
                sum_int = sum_int + l_int;

        # calculate hierarchical precision/recall/f1 for test_class
        hP = 0;
        if (sum_pred > 0):
            hP = sum_int/sum_pred

        hR = 0;
        if (sum_true > 0):
            hR = sum_int/sum_true 

        hF1 = 0;
        if ((hP+hR) > 0):
            hF1 = (2*hP*hR)/(hP+hR)
            
        # increment micro stats aggregate values
        micro_int += sum_int
        micro_true += sum_true
        micro_pred += sum_pred
        
        # get node depth from root 
        dist = len(root_dist(graph, test_class)) # don't adjust for root

        # store results per class
        class_dict.update( { copy.copy(test_class): 
                                       { 'hP' : copy.copy(hP),
                                         'hR' : copy.copy(hR),
                                         'hF1' : copy.copy(hF1),
                                          'n' : copy.copy(n) } } ) 
        # sore in table 
        temp_pc =  {}   

        # variables 
        temp_pc.update({'node':copy.copy(test_class)})
        temp_pc.update({'root-dist':copy.copy(l_test)})
        temp_pc.update({'hP':copy.copy(hP)})
        temp_pc.update({'hR':copy.copy(hR)})
        temp_pc.update({'hF1':copy.copy(hF1)})
        temp_pc.update({'n':copy.copy(n)})
        
        # append to per class
        per_class = per_class.append(temp_pc, ignore_index = True)

    # calculate micro average
    microP = 0
    if (micro_pred > 0):
        microP = micro_int/micro_pred

    microR = 0
    if (micro_pred > 0):
        microR = micro_int/micro_true 

    microF1 = 0
    if ((microP+microR) > 0):
        microF1 = (2*microP*microR)/(microP+microR)
    
    # Get values for macro averages
    macroP_array = [];
    macroR_array = [];
    for k, v in class_dict.items():
        for k1, v1 in v.items():
            if k1 == "hP":
                macroP_array.append(v1)
            if k1 == "hR":
                macroR_array.append(v1)
         
    # calculate macro average
    macroP = np.average(macroP_array)
    macroR = np.average(macroR_array)
    macroF1 = 0
    if ((macroP+macroR) > 0):
        macroF1 = (2*macroP*macroR)/(macroP+macroR)
     
    # create summary dict
    summary_dict = { "macro": {"hR": macroR, 
                              "hP": macroP, 
                              "F1": macroF1}, 
                     "micro": {"hR": microR, 
                              "hP": microP, 
                              "F1": microF1}, 
                     "per_class": class_dict }
    
    #return summary
    return(summary_dict, per_class)

#(summary, summary_table) = summary_statistics_per_class(graph, test_labels, test_classifications, penalty=False)
#print(summary)
#print(summary_table)


# In[24]:


def summary_statistics_per_class_old(graph, labels, classifications, penalty=False, macro_inc_all=False):
    
    '''calculate hierachical per-class summary statistics and micro-macro average. Statistics 
    are calculated as described in "A survey of hierarchical classification across different 
    application domains". This function will return hR, hP, hF1, hAccuracy (ad-hoc) per class and 
    and overall micro- and macro-average'''

    # make output dict
    class_dict = {}
    micro_int = 0;
    micro_pred = 0;
    micro_true = 0;
    
    # make per class table
    per_class = pd.DataFrame(columns=['node', 'root-dist', 'hP',  'hR', 'hF1', 'n'])

    # create node list (remove_root) 
    unique_classes = []
    for node in  nx.lexicographical_topological_sort(graph):
        if (node != "root"):
            unique_classes.append(node)

    for test_class in unique_classes:

        # find test path
        test_path = root_dist(graph, test_class)
        test_path.remove('root')
        l_test = len(test_path)

        # initialise variables
        sum_pred = 0;
        sum_true = 0;
        sum_int = 0;

        n = 0;

        for i,true in enumerate(labels):

            pred = classifications.iloc[i].values[0]

            # find path to root and remove root stub (artifical node)
            true_path = root_dist(graph, true)
            true_path.remove('root')
            pred_path = root_dist(graph, pred)
            pred_path.remove('root')

            # check prediction or true class match test_class 
            if( (test_class in true_path) or (test_class in pred_path) ): 

                # increment n if test_class is in true_path
                if (test_class in true_path):
                    n = n+1;

                # find intersection between true_label and test_class paths to identify l_true
                int_true = list(set(true_path)&set(test_path))

                # find length of true/test intersection and pred paths
                l_true = len(int_true)
                l_pred = len(pred_path)

                # find intersection of true/test intersection and pred
                intersection = list(set(int_true)&set(pred_path))
                l_int = len(intersection)

                # adjust length of predicted path if it is > l_true path [optional - don't penalise specific predictions]
                if (penalty == False):
                    if l_pred > l_test:
                        l_pred = l_test

                # aggregate result 
                sum_pred = sum_pred + l_pred;
                sum_true = sum_true + l_true;
                sum_int = sum_int + l_int;

        # calculate hierarchical precision/recall/f1 for test_class
        hP = 0;
        if (sum_pred > 0):
            hP = sum_int/sum_pred

        hR = 0;
        if (sum_true > 0):
            hR = sum_int/sum_true 

        hF1 = 0;
        if ((hP+hR) > 0):
            hF1 = (2*hP*hR)/(hP+hR)
            
        # increment micro stats aggregate values [only for leaf nodes/tips] 
        if is_tip(graph, test_class):
            micro_int += sum_int
            micro_true += sum_true
            micro_pred += sum_pred
        
        # get node depth from root 
        dist = len(root_dist(graph, test_class)) # don't adjust for root

        # store results per class
        class_dict.update( { copy.copy(test_class): 
                                       { 'hP' : copy.copy(hP),
                                         'hR' : copy.copy(hR),
                                         'hF1' : copy.copy(hF1),
                                          'n' : copy.copy(n) } } ) 
        # sore in table 
        temp_pc =  {}   

        # variables 
        temp_pc.update({'node':copy.copy(test_class)})
        temp_pc.update({'root-dist':copy.copy(l_test)})
        temp_pc.update({'hP':copy.copy(hP)})
        temp_pc.update({'hR':copy.copy(hR)})
        temp_pc.update({'hF1':copy.copy(hF1)})
        temp_pc.update({'n':copy.copy(n)})
        
        # append to per class
        per_class = per_class.append(temp_pc, ignore_index = True)

    # calculate micro average
    microP = 0
    if (micro_pred > 0):
        microP = micro_int/micro_pred

    microR = 0
    if (micro_pred > 0):
        microR = micro_int/micro_true 

    microF1 = 0
    if ((microP+microR) > 0):
        microF1 = (2*microP*microR)/(microP+microR)
    
    # Get values for macro averages
    macroP_array = [];
    macroR_array = [];
    for k, v in class_dict.items():
       
        # include class in macro stats if there are >0 samples in test set
        # OR outclass samples have been classified. 
        if ((v['n']>0) and (v['hF1']>0)) or (macro_inc_all is True):
            macroP_array.append(v['hP'])
            macroR_array.append(v['hR'])
            
        #for k1, v1 in v.items():
        #    if k1 == "hP":
        #        
        #    if k1 == "hR":
        #        macroR_array.append(v1)
        #        print(k1, v1)
         
    # calculate macro average
    macroP = np.average(macroP_array)
    macroR = np.average(macroR_array)
    macroF1 = 0
    if ((macroP+macroR) > 0):
        macroF1 = (2*macroP*macroR)/(macroP+macroR)
     
    # create summary dict
    summary_dict = { "macro": {"hR": macroR, 
                              "hP": macroP, 
                              "F1": macroF1}, 
                     "micro": {"hR": microR, 
                              "hP": microP, 
                              "F1": microF1}, 
                     "per_class": class_dict }
    
    #return summary
    return(summary_dict, per_class)

#(summary, summary_table) = summary_statistics_per_class_ext(graph, test_labels, test_classifications, penalty=False)
#print(summary)
#print(summary_table)


# In[ ]:


def summary_statistics_per_class(graph, labels, classifications, penalty=False, macro_inc_all=False):
    
    '''calculate hierachical per-class summary statistics and micro-macro average. Statistics 
    are calculated as described in "A survey of hierarchical classification across different 
    application domains". This function will return hR, hP, hF1, hAccuracy (ad-hoc) per class and 
    and overall micro- and macro-average'''

    # make output dict
    class_dict = {}
    micro_int = 0;
    micro_pred = 0;
    micro_true = 0;
    
    # make per class table
    per_class = pd.DataFrame(columns=['node', 'root-dist', 'hP',  'hR', 'hF1', 'n', 'nP', 'nT'])

    # create node list (remove_root) 
    unique_classes = []
    for node in  nx.lexicographical_topological_sort(graph):
        if (node != "root"):
            unique_classes.append(node) # lower case

    for test_class in unique_classes:

        # find test path
        test_path = root_dist(graph, test_class)
        test_path.remove('root')
        l_test = len(test_path)

        # initialise variables
        sum_pred = 0;
        sum_true = 0;
        sum_int = 0;

        n = 0;
        nT = 0;
        nP = 0;
        
        for i,true in enumerate(labels):
            
            pred = classifications.iloc[i].values[0]
               
            # find path to root and remove root stub (artifical node)
            true_path = root_dist(graph, true)
            true_path.remove('root')
            pred_path = root_dist(graph, pred)
            pred_path.remove('root')
        
            # check prediction or true class match test_class 
            if( (test_class in true_path) or (test_class in pred_path) ): 
                
                # store true/real classes
                if(test_class in true_path):
                    nT = nT + 1
                        
                # store predicted classes
                if (test_class in pred_path):
                    nP = nP + 1
                    
                # increment n if test_class is in true_path
                if (test_class in true_path):
                    n = n+1;

                # find intersection between true_label and test_class paths to identify l_true
                int_true = list(set(true_path)&set(test_path))

                # find length of true/test intersection and pred paths
                l_true = len(int_true)
                l_pred = len(pred_path)

                # find intersection of true/test intersection and pred
                intersection = list(set(int_true)&set(pred_path))
                l_int = len(intersection)

                # adjust length of predicted path if it is > l_true path [optional - don't penalise specific predictions]
                if (penalty == False):
                    if l_pred > l_test:
                        l_pred = l_test

                # aggregate result 
                sum_pred = sum_pred + l_pred;
                sum_true = sum_true + l_true;
                sum_int = sum_int + l_int;
        
        
        #### not could exclude classes based on nT rather than n
        # n captures the hierachy rather than the original label 
        # i.e the class may not be present BUT it shares a class further up 
        # the hierachy which is! ###
        # print( sum_pred, sum_true, sum_int , n, nP, nT )

        # calculate hierarchical precision/recall/f1 for test_class
        hP = 0;
        if (sum_pred > 0):
            hP = sum_int/sum_pred

        hR = 0;
        if (sum_true > 0):
            hR = sum_int/sum_true 

        hF1 = 0;
        if ((hP+hR) > 0):
            hF1 = (2*hP*hR)/(hP+hR)
            
        # increment micro stats aggregate values [only for leaf nodes/tips] 
        if is_tip(graph, test_class):
            micro_int += sum_int
            micro_true += sum_true
            micro_pred += sum_pred
        
        # get node depth from root 
        dist = len(root_dist(graph, test_class)) # don't adjust for root

        # store results per class
        class_dict.update( { copy.copy(test_class): 
                                       { 'hP' : copy.copy(hP),
                                         'hR' : copy.copy(hR),
                                         'hF1' : copy.copy(hF1),
                                          'n' : copy.copy(n),
                                          'nP' : copy.copy(nP),
                                          'nT' : copy.copy(nT), 
                                          'root_dist' : l_test,
                                       } 
                           } ) 

        # store in table 
        temp_pc =  {}   

        # variables 
        temp_pc.update({'node':copy.copy(test_class)})
        temp_pc.update({'root-dist':copy.copy(l_test)})
        temp_pc.update({'hP':copy.copy(hP)})
        temp_pc.update({'hR':copy.copy(hR)})
        temp_pc.update({'hF1':copy.copy(hF1)})
        temp_pc.update({'n':copy.copy(n)})
        temp_pc.update({'nP':copy.copy(nP)})
        temp_pc.update({'nT':copy.copy(nT)})
        
        # append to per class
        per_class = per_class.append(temp_pc, ignore_index = True)

    # calculate micro average
    microP = 0
    if (micro_pred > 0):
        microP = micro_int/micro_pred

    microR = 0
    if (micro_pred > 0):
        microR = micro_int/micro_true 

    microF1 = 0
    if ((microP+microR) > 0):
        microF1 = (2*microP*microR)/(microP+microR)
    
    # Get values for macro averages
    macroP_array = [];
    macroR_array = [];
    for k, v in class_dict.items():
       
        # include class in macro stats if there are >0 samples in test set
        # OR outclass samples have been classified. 
        if ( (v['nT']>0) ) or (macro_inc_all is True): # and (v['hF1']>0)
            macroP_array.append(v['hP'])
            macroR_array.append(v['hR'])
            
        #for k1, v1 in v.items():
        #    if k1 == "hP":
        #        
        #    if k1 == "hR":
        #        macroR_array.append(v1)
        #        print(k1, v1)
         
    # calculate macro average
    macroP = np.average(macroP_array)
    macroR = np.average(macroR_array)
    macroF1 = 0
    if ((macroP+macroR) > 0):
        macroF1 = (2*macroP*macroR)/(macroP+macroR)
     
    # create summary dict
    summary_dict = { "macro": {"hR": macroR, 
                              "hP": macroP, 
                              "F1": macroF1}, 
                     "micro": {"hR": microR, 
                              "hP": microP, 
                              "F1": microF1}, 
                     "per_class": class_dict }
    
    #return summary
    return(summary_dict, per_class)

#(summary, summary_table) = summary_statistics_per_class(graph, test_labels, test_classifications, penalty=False)
#print(summary)
#print(summary_table)


# In[25]:


def plot_classification_report(summary_dict):
    '''plot classification report in the style of sklearn classifiaction report.'''
    print("TODO")

    plot_classification_report(summary_dict)


# In[26]:


# fit resamplers per node and store number of sample per class
def fit_resampler_only(graph, labels, features, subsampler = ""):
    '''Fit resampler and return number of samples remaining per node in hierachy.
    Returns a dictionary.'''

    labels = copy.copy(labels)
    
    output = {}
    
    # initialise storage dictionary
    storage = {}

    # process in topographical order
    nodes = nx.topological_sort(graph) # graph.nodes()
    for current in nodes:

        print(" - starting", current)

        # check if node is tip or has only one downstream node
        dec = find_decendants(graph, current)

        isTip = is_tip(graph, current)
        if ( (dec is None) or (len(dec)==1) or (isTip == 1) ):
            print(' - tip or non-branching node')
        else:

            print(" - processing: " + str(current))

            # subset data 
            (sub_labels, sub_indices, sub_features) = subset_on_node(graph, current, copy.copy(labels), features, excludeSelf = True)
            
            # rename labels for current level in hierachy
            sub_labels = copy.copy(rename_labels_by_node(graph, current, sub_labels, byDecendants = True))
            
            # subset sample if subsampler provided - NOTE: deepcopy does not work well with Pipelines+TPOT
            if not subsampler == "":
                (sub_features, sub_labels) = subsampler.fit_resample(copy.copy(sub_features), copy.copy(sub_labels))
            
            # count number of samples per node and store
            vc = pd.Series(sub_labels).value_counts()
            for i in vc.index.values:
                output.update({copy.copy(i): copy.copy(vc.loc[i])})
    
    # return values                          
    return(output)

#fit_resampler_only(graph, train_labels, train_features, subsampler = RandomUnderSampler())


# In[1]:


def overall_summary_stats(labels, classifications, graph, penalty=False):
    
    '''calculate aggregate hierachical summary stats over entire dataset. Statitics are calculated 
    as described in "A survey of hierarchical classification across different application 
    domains". Stats are calculated for entire dataset not on a per class aggregate basis.
    This function will return hR, hP, hF1 and hAccuracy (ad-hoc)'''
    
    # initialise variables
    sum_pred = 0;
    sum_true = 0;
    sum_int = 0;
    sum_acc = 0;

    n = len(labels)

    for i,true in enumerate(labels):

        # feedback##
        #print("\n", labels[i])## 

        pred = classifications.iloc[i].values[0]

        # find path to root 
        true_path = root_dist(graph, true)
        pred_path = root_dist(graph, pred)

        # remove root stub (artifical node)
        true_path.remove('root')
        pred_path.remove('root')

        # find length of true/pred paths
        l_true = len(true_path)
        l_pred = len(pred_path)

        # find intersection of true/pred
        intersection = list(set(true_path)&set(pred_path))
        l_int = len(intersection)

        # adjust length of predicted path if it is > l_true path [optional - don't penalise specific predictions]
        if (penalty == False):
            if l_pred > l_true:
                l_pred = l_true

        # precision
        precision = 0;
        if (l_pred > 0):
            precision = l_int/l_pred

        # recall 
        recall = 0;
        if (l_true > 0):
            recall = l_int/l_true

        # aggregate result 
        sum_pred = sum_pred + l_pred;
        sum_true = sum_true + l_true;
        sum_int = sum_int + l_int;
        sum_acc = sum_acc + precision

    # calculate hierachical precision/recall/f1 over entire dataset
    hP = 0;
    if (sum_pred > 0):
        hP = sum_int/sum_pred

    hR = 0;
    if (sum_true > 0):
        hR = sum_int/sum_true 
    
    hF1 = 0;
    if ( (hP+hR) > 0 ):
        hF1 = (2*hP*hR)/(hP+hR)

    # hierachical accuracy statistic is ad-hoc 
    hAccuracy = sum_acc/n
    
    # create output dictionary 
    out_dict = {'hR': hR, 
                'hP': hP,
                'hF1': hF1,
                'hAcc': hAccuracy}
    return(out_dict)

#overall_summary_stats(labels, classifications, graph)


# In[ ]:


# null resampler
class NullSampler:
    
    def __init__(
        self,
        *,
        random_state=None,
    ):
        self.random_state = random_state
    
    def set_params(self, **kwargs):
        """ temp solution to set params after initialisation. Returns self"""
        for i in kwargs:
            self.i=kwargs[i]
        return self
    
    def fit_resample(self, X, y):
        return(X,y)


# In[ ]:


# balanced sampler
class RandomBalancingSampler:
    
    def __init__(
        self,
        *,
        sampling_strategy="mean",
        random_state=None,
    ):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.sample_indices_ = []
        
    def set_params(self, **kwargs):
        """ temp solution to set params after initialisation. Returns self"""
        for i in kwargs:
            self.i=kwargs[i]
        return self
        
    def fit_resample(self, X, y):
        
        # get class counts
        values = pd.value_counts(y)

        target = 'NA'
        if self.sampling_strategy == "mean":
            target = values.mean()
        elif self.sampling_strategy == "median":
            target = values.median()
            
        # check target value has been set
        if target == 'NA':
            print(" - ERROR: invalid sampling_strategy arguement\n")
            return

        # make dicts of upsampling and downsampling target values 
        upsample = copy.copy(values)
        for i, val in enumerate(values):
            if val < target:
                upsample[i] = target

        downsample = copy.copy(upsample)
        for i, val in enumerate(values):
            if val > target:
                downsample[i] = target

        # convert to dict 
        upsample = upsample.to_dict()
        downsample = downsample.to_dict()
        
        # upsample features/labels
        upsampler = RandomOverSampler(sampling_strategy = upsample, random_state = self.random_state)
        (out_X, out_y) = upsampler.fit_resample(copy.copy(X), copy.copy(y))
        
        # downsample features/labels
        downsampler = RandomUnderSampler(sampling_strategy = downsample, random_state = self.random_state)
        (out_X, out_y) = downsampler.fit_resample(copy.copy(out_X), copy.copy(out_y))
        
        # get indices
        up_ind = upsampler.sample_indices_
        down_ind = downsampler.sample_indices_
        sample_indices = up_ind[down_ind] # get indices for upsample using downsample
        
        # set indices 
        self.sample_indices_ = sample_indices
        
        # return transformed X/y
        return(out_X, out_y)
       
#test_sampler = RandomBalancingSampler()
#test_sampler.set_params(**{'sampling_strategy': 'mean'})
#(out_features, out_labels) = test_sampler.fit_resample(train_features, train_labels)
#print(out_features.shape, len(out_labels))


# In[ ]:


# Hierarchical Resampler 
from imblearn.under_sampling import RandomUnderSampler

class HierarchicalSampler:
    
    def __init__(
        self,
        *,
        resampler=None,
        random_state=None,
        graph=None
    ):
        self.random_state = random_state
        self.resampler = resampler
        self.graph = graph
        
    def fit_resample(self, X, y):
        
        # set data and variables 
        X_out = copy.copy(X) 
        y_out = copy.copy(y)
        graph = copy.copy(self.graph)
        sampling_counts = pd.DataFrame(columns = ['node', 'class', 'current_label', 'count'])
        node_counts = pd.DataFrame(columns = ['node', 'class_count', 'sample_count', 'resampled_count'])

        # generate nodes in graph in reverse topographical order
        rev_nodes = reversed(list(nx.topological_sort(graph)))
        
        # drop nodes in graph which do not contain samples
        inc_nodes = []
        for current in rev_nodes:
            (sub_labels_original, sub_indices, sub_features) = subset_on_node(graph, current, copy.copy(y_out), 
                                                                        copy.copy(X_out), excludeSelf = False)
            if np.size(sub_labels_original):
                inc_nodes.append(current)

        # process remaining nodes in reverse topographical order
        for current in inc_nodes:

            # check for tip and number of descendants 
            isTip = is_tip(graph, current)
            dec = find_decendants(graph, current)
            
            # process node according to type
            if (  isTip == 0 ) and ( ( dec is not None ) and ( len(dec)>1 ) ):

                 # subset data on node
                (sub_labels_original, sub_indices, sub_features) = subset_on_node(graph, current, copy.copy(y_out), 
                                                                            copy.copy(X_out), excludeSelf = False)
            
                # rename labels for current level in hierachy
                sub_labels = rename_labels_by_node(graph, current, copy.copy(sub_labels_original), byDecendants = True)
            
                # get number of samples before resampling 
                no_samples_b4 = len(sub_labels)
                
                # find number of unique classes 
                l_unique = len(np.unique(sub_labels))
                
                # only process nodes with > 1 classes 
                if l_unique != 1:
                    
                    # rebalance samples in node using appropriate sampling scheme
                    resampler = self.resampler.set_params(**{'random_state': self.random_state})
                    (new_X, temp_y) = resampler.fit_resample(copy.copy(sub_features), copy.copy(sub_labels))
                    
                    #  extract resampler indices - make unique list for dropping samples
                    resampler_ind = copy.copy(resampler.sample_indices_)

                    # use indices to recover original labels
                    sub_labels_rebalanced = sub_labels_original[resampler_ind]

                    # recover original index values from features
                    original_X_outdex = sub_features.index.values[resampler_ind]

                    # rename duplicates in y_labels with .1, .2 etc
                    renamed_X_outdex = []
                    for i in original_X_outdex:
                        temp = copy.copy(i)
                        temp_val = 1
                        temp_new = "%s.%i" % (temp, temp_val) 
                        while temp_new in renamed_X_outdex:
                            temp_val = temp_val+1
                            temp_new = "%s.%i" % (temp, temp_val) 
                        renamed_X_outdex.append(copy.copy(temp_new))

                    # add renamed index to new_X
                    new_X.index = copy.copy(renamed_X_outdex)

                    # get index lengths for sanity checking
                    ri = len(new_X.index.values)
                    riu = len(np.unique(new_X.index.values))
                    if ri != riu:
                         raise ValueError(" - ERROR: length of replaced X index contains non-unique values in node %s" % (current))

                    # check previous indices for duplicates
                    ind_vals = len(X_out.index.values)
                    ind_vals_unique = len(np.unique(X_out.index.values))
                    if ind_vals != ind_vals_unique:
                         raise ValueError(" - ERROR: length of input X index contains non-unique values in node %s" % (current))

                    # check for intersection between new and old indices
                    intersect = X_out.index.intersection(new_X.index.values)
                    if (len(intersect)>0):
                        raise ValueError(" - ERROR: interesection between old and new sample indices in node %s" % (current))

                    # replace original features with resampled features
                    X_prev_len =  X_out.shape[0] # original #samples 
                    X_outd = X_out.index.values[sub_indices]  # get index of X samples to remove - convert numerical row index to index value from subset_on_node
                    X_out = X_out.drop(X_outd, axis = 0) # drop original rows in X
                    X_outt_len =  X_out.shape[0]  # intermediate #samples 
                    X_out = X_out.append(new_X, ignore_index = False) # append rebalanced samples back to working X

                    # remove and append new y labels  
                    y_temp = np.delete(y_out, sub_indices)
                    y_out = np.append(y_temp, sub_labels_rebalanced)

                    # sanity check - intermediate X should match expected number of rows 
                    exp_rows = X_prev_len - len(sub_indices)
                    if exp_rows != X_outt_len:
                        raise ValueError(" - ERROR: intermediate X features (%i) does not have expected number of samples (%i) after dropping tips in node %s" % (X_outt_len, exp_rows, current))

                    # sanity check - y length should match rownumber in X
                    y_len = len(y_out)
                    X_len = X_out.shape[0]
                    if y_len != X_len:
                        raise ValueError(" - ERROR: length of y labels (%i) does not match size of X features (%i) in node %s" % (y_len, X_len, current))

                    # check processed indices for duplicates
                    ind_vals = len(X_out.index.values)
                    ind_vals_unique = len(np.unique(X_out.index.values))
                    if ind_vals != ind_vals_unique:
                         raise ValueError(" - ERROR: length of processed X index contains non-unique values in node %s" % (current))

                    # store before and after info about number of samples
                    temp_vc = pd.pandas.Series(y_out).value_counts()
                    temp_counts = pd.DataFrame(columns = ['node', 'class', 'current_label', 'count'])
                    temp_counts['class'] = np.array(temp_vc.index.values)
                    current_labels = rename_labels_by_node(graph, current, copy.copy(np.array(temp_vc.index.values)), byDecendants = True)
                    temp_counts['current_label'] =  np.array(current_labels)
                    temp_counts['count'] = np.array(temp_vc)
                    temp_counts = temp_counts.assign(node = current)
                    sampling_counts = sampling_counts.append(temp_counts, ignore_index=True)
                    
                    # get summary totals
                    temp_sum = {'node': current, 
                                'class_count': l_unique,
                                'sample_count': no_samples_b4,
                                'resampled_count': len(temp_y)}
                    node_counts = node_counts.append(temp_sum, ignore_index = True)
                   
        # update self with sampling counts/node_summary
        self.sampling_counts = sampling_counts
        self.node_counts = node_counts
                       
        return(X_out, y_out)
    
#sampling_counts.to_csv('test_sampling_counts.tsv', sep = "\t")

# subset data 
#(sub_labels, sub_indices, sub_features) = subset_on_node(graph, 'Africa', train_labels, train_features, excludeSelf = True)
            

#HS = HierarchicalSampler(resampler = RandomBalancingSampler(sampling_strategy = 'mean'), random_state = 34)
#HS = HierarchicalSampler(resampler = RandomOverSampler(), random_state = 34, graph = graph)
#(X_out, y_out) = HS.fit_resample(X = sub_features, y = sub_labels)
#print(X_out.shape)
#print(len(y_out))
#print(HS.sampling_counts)
#print(HS.node_counts)


# In[28]:


def test_run():
    
    # generate example data
    (features, labels, metadata) = generate_example_data()

    # create DAG from metadata
    graph = metadata_to_DAG(metadata, columns = ['a','b','c'], visualise = True)
       
    # check graph has 'root' node 
    check_root(graph)
    
    # eset example variables
    target = 'a'
    tip = 'a1a'
    
    # check sub functions
    if find_ancestors(graph, target) != 'root':
        print(' - find_ancestors failed')
    if not np.array_equal(find_decendants(graph, target), ['a1', 'a2']): 
        print(' - find_decendants failed')
    if is_tip(graph, tip) != 1:
        print(' - is_tip failed')
    if root_dist(graph, tip) != ['root', 'a', 'a1', 'a1a']:
        print(' - root_dist failed')
    if get_class_labels(graph, target) != ['a1', 'a2', 'a1a', 'a1b']:
        print(' - get_class_labels failed')
    (test1, test2) = subset_on_node(graph, target, labels)
    if (not np.array_equal( test1, ['a', 'a2', 'a1', 'a1a', 'a1', 'a1b', 'a1a', 'a1b', 'a1a', 'a1b', 'a1a', 'a1b']) ):
        print(' - subset_on_node failed (1)')
    if (not np.array_equal( test2, [0, 1, 6,  7,  8,  9, 10, 11, 12, 13, 14, 15] ) ):
        print(' - subset_on_node failed (2)')
    if not np.array_equal(rename_labels_by_node(graph, target, copy.copy(labels), byDecendants = True), ['a', 'a2', 'b', 'b1', 'b', 'b2', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1',
 'a1', 'c1', 'c2']):
        print(' - rename_labels_by_node failed')    
    
    # set classifier
    classifier = DummyClassifier(strategy="most_frequent")

    # fit classifier
    models = fit_hierarchical_classifier(graph, copy.copy(labels), copy.copy(features), 
                                         classifier, verbose = True)
    
    # classify training samples
    (classification_table, classifications) = classify_samples_in_hierarchy(graph, features, 
                                                                       models,
                                                                       mode = 'max', 
                                                                       threshold = 0.51, 
                                                                       verbose = True) 
    #print(classifications)
    
    # generate confusion matrix
    confusion_mat = hierarchical_confusion_matrix (graph, classifications, labels)
    print(confusion_mat)
    
    # generate hierachical summary stats per class
    summary = summary_statistics_per_class(graph, labels, classifications, penalty=False)
    print(summary)
    
    # generate flat summary stats for each model per node
    (summary_per_node, summary_per_class, clf_tables) = per_node_summary_stats(graph, labels, features, models)
    print(summary_per_node)
    print(summary_per_class)
    print(clf_tables)
        
#test_run()


# In[29]:


def process_SNPAddress_to_features (meta_file="", DAG_vars = "", remove_count = 0):
    
    # check required vars provided
    if DAG_vars == "":
        print(" - ERROR: DAG_vars required");
        return(None)
    if meta_file == "":
        print(" - ERROR: metadata file required");
        return(None)

    # read metadata
    meta_subset = pd.read_csv(meta_file, sep='\t').set_index('SRA Accession', drop = False)
    meta = meta_subset

    # remove NaNs from metadata - replace with empty strings
    meta = meta.fillna("")
    
    # genrate columns for DAG creation 
    DAG_meta = meta[DAG_vars]

    # make temp labels variable - these will be modified later by relabel_sample_by_count_per_node
    temp_labels = create_labels(DAG_meta, columns = DAG_vars)
    #pd.DataFrame(meta[var], columns = [var]).rename(columns={var: "label"}).set_index(meta['SRA Accession'])

    # identify indeices/classes to remove due to low sample size from input data
    temp_graph = metadata_to_DAG(DAG_meta, visualise = True)###########FAlse
    (renamed_labels, rm_ind, rm_classes) = relabel_samples_by_count_per_node(temp_graph, temp_labels, threshold=remove_count)

    #  add corrected labels variable to metadata 
    meta['corrected_labels'] = copy.copy(renamed_labels)

    # remove samples identified by relabel_samples_by_count_per_node
    print(" - classes remove or relabelled:" + str(rm_classes))
    print("# meta before filtering", meta.shape)
    meta = meta.drop(meta.index[rm_ind])
    print("# meta after filtering", meta.shape)

    # save final meta
    meta.to_csv('meta.filtered_renamed.tsv', sep = "\t",  index = False, header = True)

    # regenerate graph 
    DAG_meta = meta[DAG_vars]
    for i in rm_classes:
        DAG_meta = DAG_meta.replace(i,"")    
    graph = metadata_to_DAG(DAG_meta, visualise = True)

    # save DAG metadata
    DAG_meta.to_csv('DAG_meta.tsv', sep = "\t",  index = False, header = False)

    # add stratified SNP address info to metadata

    # split SNP Address into multiple columns 
    SNP_split = meta.SNP.str.split(".", expand=True)

    # join columns to create unique SNP addresses
    meta['SNP250'] = SNP_split[0]
    meta['SNP100'] = SNP_split[[0, 1]].agg('.'.join, axis=1)
    meta['SNP50'] = SNP_split[[0, 1, 2]].agg('.'.join, axis=1)
    meta['SNP25'] = SNP_split[[0, 1, 2, 3]].agg('.'.join, axis=1)
    meta['SNP10'] = SNP_split[[0, 1, 2, 3, 4]].agg('.'.join, axis=1)
    meta['SNP5'] = SNP_split[[0, 1, 2, 3, 4, 5]].agg('.'.join, axis=1)
    # meta['SNP'] = SNP_split[[0, 1, 2, 3, 4, 5, 6]].agg('.'.join, axis=1) # only one representative per cluster- exclude

    # The majority of SNP addresses are unique and potentially uninformative. Use SNP5 or greater

    # one hot encode SNP Address excluding SNP0 info 
    SNPAddress = meta[['SNP5','SNP10','SNP25','SNP50','SNP100','SNP250']]

    # make features variable
    features = pd.get_dummies(SNPAddress,dummy_na=True).set_index(meta.index)
    print("# features before filtering", features.shape)

    # filter features which occur only once
    rm_col = features.columns[features.sum()==1]
    features = features.drop(columns=rm_col)
    print("# features after filtering", features.shape, "\n")

    # make labels as dataframe, later convert to np.array
    labels =  pd.DataFrame(meta['corrected_labels'], columns = ['corrected_labels']).rename(columns={'corrected_labels': "label"}).set_index(meta['SRA Accession'])

    # make other category for any classes below a set frequency
    classes = labels.label.value_counts()
    print("Class counts before filtering:\n", labels.label.value_counts(), "\n")

    # Convert labels to np.array
    labels = labels['label'].values

    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, 
    #                                                                            test_size = 0.25, 
    #                                                                            stratify=labels, # stratify on country
    #                                                                            random_state = seed)

    # feedback
    #print('Features Shape:', features.shape)
    #print('Labels Shape:', labels.shape, "\n")
    #print('Training Features Shape:', train_features.shape)
    #print('Training Labels Shape:', train_labels.shape)
    #print('Testing Features Shape:', test_features.shape)
    #print('Testing Labels Shape:', test_labels.shape, "\n")
    
    # return vars
    return(graph, features, labels, meta)

