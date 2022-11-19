#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
#import pylab
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
# import mplcursors
# import plotly
# import pickle
# import streamlit as st
import plotly.express as px
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=50.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2,initial_dims=56, perplexity=29.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 500
    initial_momentum = 0.2 #0.5
    final_momentum = 0.9 #0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y
def standardize_features_all(df):
    """
    Standardizes the features in the feature matrix by subtracting the mean and dividing by the standard deviation. ()

        Parameters:
                df (pandas.DataFrame): the feature matrix

        Returns:
                df (pandas.DataFrame): the feature matrix with standardized values.
    """
    # This might be faster if we used a numpy array.
    # for feature in ['mesh_area', "mesh_compactness", "mesh_rectangularity","mesh_diameter", "mesh_eccentricity"]:
    #     df[feature] = (df[feature] - df[feature].mean() / df[feature].std())
    # return df
    df['mesh_area'] = ((df['mesh_area'] - df['mesh_area'].mean()) / df['mesh_area'].std())
    df['mesh_compactness'] = ((df['mesh_compactness'] - df['mesh_compactness'].mean()) / df['mesh_compactness'].std())
    df['mesh_rectangularity'] = ((df['mesh_rectangularity'] - df['mesh_rectangularity'].mean()) / df['mesh_rectangularity'].std())
    df['mesh_diameter'] = ((df['mesh_diameter'] - df['mesh_diameter'].mean()) / df['mesh_diameter'].std())
    df['mesh_eccentricity'] = ((df['mesh_eccentricity'] - df['mesh_eccentricity'].mean()) / df['mesh_eccentricity'].std())
    
    return df
def remove_outliers(df):
    df["mesh_eccentricity"] = np.where(df["mesh_eccentricity"] > 100, 50,df["mesh_eccentricity"])
    df["mesh_compactness"] = np.where(df["mesh_compactness"] > 1000, 300,df["mesh_compactness"])
    #print(df.describe())
    return df

def df_to_feature_matrix(dataframe):
    """
    Transforms a pandas DataFrame into a numpy feature matrix. 

        Parameters:
                dataframe (pandas.DataFrame) the dataframe as created by features.py

        Returns:
                feature_matrix (numpy.ndarray) numpy array with only the values representing the features and the ID
    """
    histogram_vectors = histograms_to_vector(dataframe)
    dataframe = dataframe[['ID'] + FEATURES]
    # TODO add histograms
    feature_matrix = dataframe.to_numpy()
    feature_matrix = np.hstack((feature_matrix, histogram_vectors))
    return feature_matrix
FEATURES = ['mesh_area', 'mesh_compactness', 'mesh_rectangularity', 'mesh_diameter', 'mesh_eccentricity']

def histograms_to_vector(df):
    column_list = ["mesh_a3", "mesh_d1", "mesh_d2","mesh_d3", "mesh_d4"]
    histogram_lists = []
    for columns in column_list:
        histogram_list = np.vstack(df[columns].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' ')))
        histogram_lists.append(histogram_list)
    histograms = np.hstack(histogram_lists)
               
    return histograms

if __name__ == "__main__":
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")

    with open(r'C:\Users\niels\MR\INFOMR-Multimedia-Retrieval\labels\id_to_label_49.json', 'rb') as file:
        id_to_label = json.load(file)    
    df = pd.read_csv(r"C:\Users\niels\Downloads\df.csv", index_col=0)    
    with open(r'C:\Users\niels\MR\INFOMR-Multimedia-Retrieval\labels\label_to_id_49.json', 'rb') as file:
        label_to_id = json.load(file)
    
    # with open(r'C:\Users\niels\MR\INFOMR-Multimedia-Retrieval\labels\id_to_label_7.json', 'rb') as file:
    #     id_to_label = json.load(file)    
    # df = pd.read_csv("feature_vectors_2.csv", index_col=0)    
    # with open(r'C:\Users\niels\MR\INFOMR-Multimedia-Retrieval\labels\label_to_id_7.json', 'rb') as file:
    #     label_to_id = json.load(file)

    # TODO THIS SHOULD NOT BE DONE IN QUERY
    #print(list(label_to_id.keys()))
    list_names = list(df['ID'])
    df = remove_outliers(df)
    df=df.dropna()
    
    df_copy = df.copy()
    df2 = standardize_features_all(df_copy)
    
    feature_matrix = df_to_feature_matrix(df2)
    #print(feature_matrix.shape)
    label_to_color = dict()
    # cnames = {
    # 'aliceblue':            '#F0F8FF',
    # 'aqua':                 '#00FFFF',
    # 'aquamarine':           '#7FFFD4',
    # 'azure':                '#F0FFFF',
    # 'beige':                '#F5F5DC',
    # 'bisque':               '#FFE4C4',
    # 'black':                '#000000',
    # 'blanchedalmond':       '#FFEBCD',
    # 'blue':                 '#0000FF',
    # 'blueviolet':           '#8A2BE2',
    # 'brown':                '#A52A2A',
    # 'burlywood':            '#DEB887',
    # 'cadetblue':            '#5F9EA0',
    # 'chartreuse':           '#7FFF00',
    # 'chocolate':            '#D2691E',
    # 'coral':                '#FF7F50',
    # 'cornflowerblue':       '#6495ED',
    # 'cornsilk':             '#FFF8DC',
    # 'crimson':              '#DC143C',
    # 'cyan':                 '#00FFFF',
    # 'darkblue':             '#00008B',
    # 'darkcyan':             '#008B8B',
    # 'darkgoldenrod':        '#B8860B',
    # 'darkgray':             '#A9A9A9',
    # 'darkgreen':            '#006400',
    # 'darkkhaki':            '#BDB76B',
    # 'darkmagenta':          '#8B008B',
    # 'darkolivegreen':       '#556B2F',
    # 'darkorange':           '#FF8C00',
    # 'darkorchid':           '#9932CC',
    # 'darkred':              '#8B0000',
    # 'darksalmon':           '#E9967A',
    # 'darkseagreen':         '#8FBC8F',
    # 'darkslateblue':        '#483D8B',
    # 'darkslategray':        '#2F4F4F',
    # 'darkturquoise':        '#00CED1',
    # 'darkviolet':           '#9400D3',
    # 'deeppink':             '#FF1493',
    # 'deepskyblue':          '#00BFFF',
    # 'dimgray':              '#696969',
    # 'dodgerblue':           '#1E90FF',
    # 'firebrick':            '#B22222',
    # 'forestgreen':          '#228B22',
    # 'fuchsia':              '#FF00FF',
    # 'gainsboro':            '#DCDCDC',
    # 'gold':                 '#FFD700',
    # 'goldenrod':            '#DAA520',
    # 'gray':                 '#808080',
    # 'green':                '#008000',
    # 'greenyellow':          '#ADFF2F',
    # 'honeydew':             '#F0FFF0',
    # 'hotpink':              '#FF69B4',
    # 'indianred':            '#CD5C5C',
    # 'indigo':               '#4B0082',
    # 'ivory':                '#FFFFF0',
    # 'khaki':                '#F0E68C',
    # 'lavender':             '#E6E6FA',
    # 'lavenderblush':        '#FFF0F5',
    # 'lawngreen':            '#7CFC00',
    # 'lemonchiffon':         '#FFFACD',
    # 'lightblue':            '#ADD8E6',
    # 'lightcoral':           '#F08080',
    # 'lightcyan':            '#E0FFFF',
    # 'lightgoldenrodyellow': '#FAFAD2',
    # 'lightgreen':           '#90EE90',
    # 'lightgray':            '#D3D3D3',
    # 'lightpink':            '#FFB6C1',
    # 'lightsalmon':          '#FFA07A',
    # 'lightseagreen':        '#20B2AA',
    # 'lightskyblue':         '#87CEFA',
    # 'lightslategray':       '#778899',
    # 'lightsteelblue':       '#B0C4DE',
    # 'lightyellow':          '#FFFFE0',
    # 'lime':                 '#00FF00',
    # 'limegreen':            '#32CD32',
    # 'linen':                '#FAF0E6',
    # 'magenta':              '#FF00FF',
    # 'maroon':               '#800000',
    # 'mediumaquamarine':     '#66CDAA',
    # 'mediumblue':           '#0000CD',
    # 'mediumorchid':         '#BA55D3',
    # 'mediumpurple':         '#9370DB',
    # 'mediumseagreen':       '#3CB371',
    # 'mediumslateblue':      '#7B68EE',
    # 'mediumspringgreen':    '#00FA9A',
    # 'mediumturquoise':      '#48D1CC',
    # 'mediumvioletred':      '#C71585',
    # 'midnightblue':         '#191970',
    # 'mintcream':            '#F5FFFA',
    # 'mistyrose':            '#FFE4E1',
    # 'moccasin':             '#FFE4B5',
    # 'navy':                 '#000080',
    # 'oldlace':              '#FDF5E6',
    # 'olive':                '#808000',
    # 'olivedrab':            '#6B8E23',
    # 'orange':               '#FFA500',
    # 'orangered':            '#FF4500',
    # 'orchid':               '#DA70D6',
    # 'palegoldenrod':        '#EEE8AA',
    # 'palegreen':            '#98FB98',
    # 'paleturquoise':        '#AFEEEE',
    # 'palevioletred':        '#DB7093',
    # 'papayawhip':           '#FFEFD5',
    # 'peachpuff':            '#FFDAB9',
    # 'peru':                 '#CD853F',
    # 'pink':                 '#FFC0CB',
    # 'plum':                 '#DDA0DD',
    # 'powderblue':           '#B0E0E6',
    # 'purple':               '#800080',
    # 'red':                  '#FF0000',
    # 'rosybrown':            '#BC8F8F',
    # 'royalblue':            '#4169E1',
    # 'saddlebrown':          '#8B4513',
    # 'salmon':               '#FA8072',
    # 'sandybrown':           '#FAA460',
    # 'seagreen':             '#2E8B57',
    # 'seashell':             '#FFF5EE',
    # 'sienna':               '#A0522D',
    # 'silver':               '#C0C0C0',
    # 'skyblue':              '#87CEEB',
    # 'slateblue':            '#6A5ACD',
    # 'slategray':            '#708090',
    # 'snow':                 '#FFFAFA',
    # 'springgreen':          '#00FF7F',
    # 'steelblue':            '#4682B4',
    # 'tan':                  '#D2B48C',
    # 'teal':                 '#008080',
    # 'thistle':              '#D8BFD8',
    # 'tomato':               '#FF6347',
    # 'turquoise':            '#40E0D0',
    # 'violet':               '#EE82EE',
    # 'wheat':                '#F5DEB3',
    # 'yellow':               '#FFFF00',
    # 'yellowgreen':          '#9ACD32'}
    
    # labels = []
    # colors = list(cnames.keys())
    # for f in feature_matrix[:200]:
    #     label = id_to_label[str(int(f[0]))]
    #     if(label in label_to_color.keys()):
    #         labels.append(label_to_color[label])
    #     else:
    #         label_to_color[label] = colors.pop()
    #         labels.append(label_to_color[label])
    
    labels = []
    for f in feature_matrix:
        labels.append(id_to_label[str(int(f[0]))])
    ids = []
    for f in feature_matrix:
        ids.append((f[0]))
    
    
    # legend_test = []
    # for f in feature_matrix[:100]:
    #     legend_test.append(id_to_label[str(int(f[0]))])     
    # print(labels)

    
    #print(label_to_id.keys())
    # colordict = {'animal': 'yellow','building':'grey','vehicle':'red','plant':'green','furniture':'black','household':'brown','-1':'purple'}

    colordict = {'winged_vehicle': 'gray', 'balloon_vehicle': 'silver', 'helicopter':'darkslategray', 'arthropod':'darkolivegreen', 'human':'saddlebrown', 'flying_creature':'seagreen', 
                 'quadruped':'forestgreen', 'snake':'maroon', 'underwater_creature':'olive', 'head':'darkslateblue', 'hand':'cadetblue', 'building':'steelblue', 'city':'navy', 'display_device':'chocolate', 'door':'yellowgreen', 
                 'fireplace':'silver', 'cabinet':'sienna', 'seat':'seashell', 'shelves':'seagreen', 'table':'sandybrown', 'geographic_map':'salmon', 'hat':'saddlebrown', 
                 'ladder':'indianred', 'lamp':'limegreen', 'liquid_container':'goldenrod', 
                 'mailbox':'darkseagreen', 'musical_instrument':'darkmagenta', 'plant':'tan', 'satellite_dish':'darkorchid', 
                 'sea_vessel':'red', 'sign':'darkturquoise', 'sink':'darkorange', 'slot_machine':'gold', 'staircase':'lime', 'handheld':'mediumspringgreen', 
'car':'crimson', 'cycle':'deepskyblue', 
'wheel':'blue', 'trex':'pink', 'blade':'greenyellow', 'skeleton':'tomato', 
'torso':'orchid', 'bridge':'fuchsia', 'chess_piece':'dodgerblue', 'chest':'palevioletred', 'dragon':'khaki', 'bed':'aquamarine', 
'gun':'plum', 'microchip':'lightgreen', 'shoe':'deeppink', 'snowman':'mediumslateblue', 'swingset':'lightsalmon', 'train':'paleturquoise'}
#     fake_handles = [mpatches.Patch(color=item) for item in colordict.values()]
#     plt.rcParams['figure.figsize'] = [15, 15]
#     plt.rcParams['font.size'] = 8
#     label = colordict.keys()
# #     plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})   
#     print((feature_matrix.shape))
#     print('0',feature_matrix[300][0])
#     print('1',feature_matrix[300][55])
#     plt.title('t-SNE plot of the Princeton dataset feature vectors projected in 2D.')
    for i in range(0,len(feature_matrix)):
        #print(f)
        
        feature_matrix[i][0] = 0
        #1546, 56
        #0
    print((feature_matrix.shape))
    #Y = tsne(feature_matrix, 2)
    # with open('tsne_data.pkl','wb') as f:
    #     pickle.dump(Y, f)
    #print(Y)
    Y = np.load('tsne_data.pkl', allow_pickle=True)
    #fig = plt.figure(figsize=(20, 20))
    
    #Uncomment block below
    
    # a = sns.scatterplot(data=Y, x=Y[:, 0],y=Y[:, 1],palette=colordict,hue=labels,hue_order=label) #hue=labels,
    # #a = plt.scatter(x=Y[:, 0],y=Y[:, 1],c=colordict.values())
    # #pylab.scatter(Y[:, 0], Y[:, 1], c=dict.get(labels)) #c=labels
    # #plt.legend(label_to_id.keys())
    # #plt.legend(label)
    # # by default the tooltip is displayed "onclick"
    # # we can change it by setting hover to True
    # cursor = mplcursors.cursor(a, hover=True)
    # # by default the annotation displays the xy positions
    # # this is to change it to the countries name
    # print(len(labels),len(list_names))
    # @cursor.connect("add")
    # def on_add(sel):
    #     sel.annotation.set(text=f'{labels[sel.index]},{list_names[sel.index]}') #sel.target.index
    #     #sel.annotation.set(text=list_names[sel.target.index])
    # sns.move_legend(a, "center left", bbox_to_anchor=(1, 0.5))
    # #st.pyplot(fig)
    # pylab.show()
    
    
    fig = px.scatter(Y, x=Y[:, 0],y=Y[:, 1],color_discrete_map =colordict,color=labels,hover_name=ids,labels=labels) #color=labels,hover_name=ids,labels=labels
    #fig.show()
    st.write(fig)
    
    
    # scatter_x = np.array([1,2,3,4,5])
    # scatter_y = np.array([5,4,3,2,1])
    # group = np.array([1,3,2,1,3])
    # cdict = {1: 'red', 2: 'blue', 3: 'green'}

    # fig, ax = plt.subplots()
    # for g in np.unique(group):
    #     ix = np.where(group == g)
    #     ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
    # ax.legend()
    # plt.show()    
