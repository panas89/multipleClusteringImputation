
import numpy as np
from IPython.display import clear_output
import itertools as it
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.neighbors import BallTree
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def pipelineOfVariation(caseInd, dfTrain, dfTest, printOutput, tolerance_Value, categorical, 
                        continuous, radius, alpha, variations, partialLinear, linearVarCols):
    """
    Method that perfroms a pipeline of imputing the data and returns imputed data
    caseInd: index of test case
    dfTrain: training set dataframe
    dfTest:test set dataframe
    printOutput: if for one case maybe want o print the ouput of number if similar cases
    tolerance_Value: represents the tolerance value
    categorical: list of categorical variables
    continuous: list of continuous variables
    radius: scalar that defines the radius of the hypersphere of the BallTree algorithm used to find similar cases
    alpha: the confidence interval value
    variations: boolean that represents whether to generate C.I.s for imputations
    partialLinear: boolean that represents whether we generate simulated data points in uncertain values
    linearVarCols: list of columns fo the variables to generate simulated data points in uncertain values
    
    Retruns:
    the imputed test case, in the case of missing continuous variables it returns the possible imputations of these continuous variables
    """
    
    nullMatrix = dfTest.isnull().as_matrix()
    row = nullMatrix[caseInd,:]
    combs = getCombinations(row,dfTrain,tolerance_Value=tolerance_Value)
    
    if sum(row)>0: #if there are missing values
        dfAllNNs, _ , _ = getNNs(dfTrain, dfTest, combs, row, radius=radius, printOutput=printOutput, caseInd=caseInd)
    else:
        dfAllNNs = None
        
    if np.sum(row)==0:
        x = getDatasetOnlyVariationsLinear(dfTest=dfTest, row=row, caseInd=caseInd, categorical=categorical, 
                                   continuous=continuous, variations=variations, 
                                   partialLinear=partialLinear,linearVarCols = linearVarCols)
    else:
        x = getDatasetOfVariations(dfAllNNs, dfTest, row=row,caseInd=caseInd, categorical=categorical, 
                                   continuous=continuous, alpha=alpha, variations=variations, 
                                   partialLinear=partialLinear,linearVarCols = linearVarCols)
    
    if printOutput==False:
        clear_output()
    
    return x
    


def queryNN(X_train, X_test, radius, leaf_size):
    """
    Method that identifies from a dataset the NN most similar cases (Nearest neighbors).
    X_train: dataset to find neighbours
    X_test: dataset to find neighbors for
    BallTree_leaf_size: leaf size of kd tree
    radius: radius in high dimensional space to search for NNs
    
    Returns:
    counts: count of NNs for each datapoint
    indices: indices of NNs from dataset X_train
    """
   
    tree = BallTree(X_train, leaf_size=leaf_size) 
    counts = tree.query_radius(X_test, r=radius, count_only=True)
    indices = tree.query_radius(X_test, r=radius)
    return counts, indices

def getVariablesCI(X,alpha):
    """
    Method that computes the mean of the NN and then uses that mean to define an interval based on a normal distr.
    Returns that interval.
    X: data
    alpha: value of C.I.
    """
    if X.ndim > 1:
        confs = []
        X = X.T

        for i in X:

            mean, sigma,conf_int = confidenceInterval(X= i[~np.isnan(i)],alpha=alpha)
            #mean, sigma = np.mean(X[indices,i]), np.std(X[indices,i])
            #conf_int = stats.norm.interval(alpha, loc=mean, scale=sigma)
            confs.append(conf_int)

        return confs
    else:
        mean, sigma,conf_int = confidenceInterval(X=X[~np.isnan(X)],alpha=alpha)
        return conf_int

def getVariablesLI(X,alpha):
    """
    Method that can be used to define a "manual" interval of a variable based on the initial value of the datapoint
    X: data
    alpha: interval range
    
    Example:
    [1,2,3] if alpha  = 0.1 
    [0.9,1.1] is the interval of the first variable
    [1.9,2.1]
    [2.9,3.1]
    """
    
    if not isinstance(X, list):
        return (X-alpha,X+alpha)
    else:
        confs = []
        for i in X.shape[0]:
            conf_int = np.array([X[i]-alpha,X[:,i]+alpha]) # +- percentage of variable value
            confs.append(conf_int)

        return confs        

def confidenceInterval(X,alpha,median=True):
    """
    Method: that compute sthe C.I. of a normal distribution.
    
    X:data
    
    Returns:
    mean: mean of distribution
    sigma: standard deviation
    conf_int: the confidence interval
    """
    
    if median:
        median, sigma = np.median(X), np.std(X)
        conf_int = stats.norm.interval(alpha, loc=median, scale=sigma)
        return median, sigma, conf_int
    else:
        mean, sigma = np.mean(X), np.std(X)
        conf_int = stats.norm.interval(alpha, loc=mean, scale=sigma)
        return mean, sigma, conf_int

# function to create all combinations of variables intervals
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
           
    Obtained and validated from:
    https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def spaceSteps(step_size,confs):
    """
    Method: to compute all intervals in a linear steps.
    
    step_size: is the the linear step size for each interval.
    confs: is the set of confidence intervals to be used.
    
    Returns:
    intervals: vectors of the intervals for each variable
    """
    
    for i in range(0,len(confs)):
        conf_int = confs[i]
        if i==0:
            intervals = np.linspace(conf_int[0],conf_int[1],step_size)
        else:
            interval = np.linspace(conf_int[0],conf_int[1],step_size)
            intervals = np.column_stack((intervals,interval))
    
    return intervals

def getNNs(dfTrain, dfTest, combs, row, radius, printOutput, caseInd):
    
    """
    Method that identifies the similar cases of a test case and returns them.
    dfTrain: training dataframe
    dfTest: testing dataframe
    combs: combinations of complete variables
    row: boolean vector representing the missing variables
    radius: radius of the hypersphere used by the BallTree algorithm
    printOutput: boolean if True used to printhe indices and similar cases
    caseInd: integer index of the test case
    
    Returns:
    dfAllNNs: dataframe of Nearest Neighbors
    allCounts: count of similar cases
    allIndices: indices of all simialr cases
    """
    
    #######################################################################
    #create dataframe to get similar cases with replacement
    df = dfTrain.copy()
    df.set_index('oldIndex',inplace=True)

    #######################################################################
    allCounts = []
    allIndices = []
    dfAllNNs = pd.DataFrame()

    for i in combs:
        ##### get dataframe of training
        X_train = dfTrain[['oldIndex'] + i].dropna(axis=0)
        cols = [k for k in X_train.columns if k not in list(dfTrain.columns[row])]
        X_train = X_train[cols].as_matrix()
        
        if X_train.shape[1]>1: #case in which too many missing values
            ##### get dataframe of testing
            X_test = dfTest.loc[caseInd,cols].as_matrix()
            
            ##### normalize
            scaler = MinMaxScaler()
            scaler.fit(X_train[:,1:])

            X_test[1:] = scaler.transform(X_test[1:].reshape(1,-1))

            ##### get NNs
            counts, indices = queryNN(X_train[:,1:],[X_test[1:]],radius=radius*len(i)*0.1,leaf_size=10)
            allCounts.append(counts)
            allIndices.append(indices)

            ##### save NNs
            dfTemp = df.loc[np.asarray(X_train[list(indices[0]),0], dtype=int),:]
            dfTemp.reset_index(inplace=True) #reset index because index of df is oldIndex set up on top
            dfAllNNs = dfAllNNs.append(dfTemp,ignore_index=True)
        
    if printOutput:
        print('Number of NNs: ' , np.sum(allCounts))
        print('Indices: ' ,allIndices)
        
    return dfAllNNs, allCounts, allIndices

def booleanRow(columns, cols):
    """
    Method that gets two string lists and checks if cols elements are in columns, returns boolean vector that represents the element
    of columns in which the elemetn of cols is.
    
    columns: list of string
    cols: lis of strings
    
    Returns:
    Boolean list
    """
    
    boolRow = []
    for i in columns:
        if i in cols:
            boolRow.append(True)
        else:
            boolRow.append(False)

    return boolRow

def getDatasetOfVariations(dfAllNNs,dfTest, row, caseInd, categorical, continuous, alpha, 
                           variations, partialLinear, linearVarCols):
    
    """
    Method that generates the dataset fo similar cases, called variations for the case of continuous variables as
    all possible cases that fall insed the range of similar cases values are returned.
    dfAllNNs: dataframe of all similar cases
    dfTest: test set dataframe
    row: boolean vector representing the variables with missing data as True
    caseInd: index of test case
    categorical: list of categorical variables
    continuous: list of continuous variables
    alpha: the confidence interval value
    variations: boolean that represents whether to generate C.I.s for imputations
    partialLinear: boolean that represents whether we generate simulated data points in uncertain values
    linearVarCols: list of columns fo the variables to generate simulated data points in uncertain values
    
    Returns:
    A numpy array of the imputed test case
    """

    #######################################################################
    
    x = dfTest.loc[caseInd].as_matrix()
       
    if sum(row)>0: #if there are missing values
        boolCategorical = booleanRow(dfAllNNs.columns,categorical)
        boolContinuous = booleanRow(dfAllNNs.columns,continuous)

        catColumns = np.logical_and(boolCategorical,row) #oldIndex not present in dfAllNNs
        contColumns = np.logical_and(boolContinuous,row)
                
        if (np.sum(catColumns)>0): 
            cols = dfAllNNs.columns[catColumns]
            freqValues = [dfAllNNs[i].value_counts().index[0] for i in cols]
            ######## impute categorical values
            ind = np.array(catColumns)
            x[ind] = freqValues
        if(np.sum(contColumns)>0):
            cols = dfAllNNs.columns[contColumns]
            if partialLinear:# and 'C_currentage' in cols:
                confs = []
                for j in cols:
                    if j in linearVarCols and ~row[list(dfAllNNs.columns).index(j)]:
                        confs.append(getVariablesLI(dfTest.loc[caseInd,j],alpha=1.0))
                    else:
                        confs.append(getVariablesCI(dfAllNNs[j].as_matrix(),alpha=alpha))
                x = getVariations(x=x, variations=variations, contColumns=contColumns, confs=confs, step_size=10) 
            else:
                confs = []
                for j in cols:
                    confs.append(getVariablesCI(dfAllNNs[j].as_matrix(),alpha=alpha))
                x = getVariations(x=x, variations=variations, contColumns=contColumns, confs=confs, step_size=10)
        else:
            contColumns = booleanRow(dfAllNNs.columns,linearVarCols)
            cols = dfAllNNs.columns[contColumns]
            if partialLinear:# and 'C_currentage' in cols:
                confs = []
                for j in cols:
                    if j in linearVarCols and ~row[list(dfAllNNs.columns).index(j)]:
                        confs.append(getVariablesLI(dfTest.loc[caseInd,j],alpha=1.0))
                x = getVariations(x=x, variations=variations, contColumns=contColumns, confs=confs, step_size=10) 
            
                
    return x


def getDatasetOnlyVariationsLinear(dfTest, row, caseInd, categorical, continuous, 
                                    variations, partialLinear, linearVarCols):
    """
    Method that generates simulated data points based on uncertain values of continuous cariables.
    Input:
    dfTest: test set dataframe
    row: boolean vector representing the variables with missing data as True
    caseInd: index of test case
    categorical: list of categorical variables
    continuous: list of continuous variables 
    variations: boolean that represents whether to generate C.I.s for imputations
    partialLinear: boolean that represents whether we generate simulated data points in uncertain values
    linearVarCols: list of columns fo the variables to generate simulated data points in uncertain values
    
    Returns:
    numpy array with simulated data points (imputed data points)
    """
    
    x = dfTest.loc[caseInd].as_matrix()
    
    if sum(row)==0 and partialLinear:
        cols = [i for i in dfTest.columns]
        boolCategorical = booleanRow(cols,categorical)
        boolContinuous = booleanRow(cols,continuous)

        if partialLinear:
            boolCols = booleanRow(cols,linearVarCols)
            row = boolCols | row

        catColumns = np.logical_and(boolCategorical,row) #oldIndex not present in dfAllNNs
        contColumns = np.logical_and(boolContinuous,row)

        confs = []
        for j in linearVarCols:
            confs.append(getVariablesLI(dfTest.loc[caseInd,j],alpha=1.0))
        x = getVariations(x=x, variations=variations, contColumns=contColumns, confs=confs, step_size=10) 
         
    return x

def getVariations(x, variations, contColumns,confs,step_size):
    
    """
    Method that generates all variations of simulated data points by computing all combinations of imputed values.
    
    Input:
    x: numpy array of a data point
    variations: boolean that represents whether to generate C.I.s for imputations
    contColumns: list of booleans that represents the index of continuous columns
    confs: list of tuples that contains the range of values for each continuous variable
    step_size: the number of steps in the conf range
    
    Returns:
    x all variations of simulated data from imputed range of values.
    """
    
    if variations==True:
        ########### Define intervals in linear steps
        intervals = spaceSteps(step_size=step_size, confs=confs)
        ########### All combinations of variables intervals
        intsCombs = cartesian(intervals.T)

        ind = np.array(contColumns)

        if(intsCombs.shape[0] > 1):
            x = np.tile(x, (intsCombs.shape[0],1))
            x[:,ind] = intsCombs
        else:
            x = np.tile(x, (intsCombs.shape[1],1))
            x[:,ind] = intsCombs.T
    else:
        confs_mean = [np.mean(i) for i in confs]
        ind = np.array(contColumns)
        x[ind] = confs_mean
        
    return x

def getCombinations(row, df, tolerance_Value):
    """
    Method: computes all the combinations of the remaining complete variables of the test set, it uses a tolerance value in each iteration
    that represents the percentage of variables included to compute all of the possibel combinations.
    
    row: boolean row representing were there is a mising value in the variables
    df: training dataset
    tolerance_Value: tolerance value represents the tolerance to missing data
    
    Returns:
    All the combinations of possible variables based on the tolerance
    """
    
    cols = ['MRN_D','G_5yearscore','oldIndex'] + list(df.columns[row])
    arrays = [i for i in df.columns if i not in cols]
    combs = []
    for i in range(int(np.round(len(arrays)*tolerance_Value)),len(arrays)+1):
        combs = combs + [list(x) for x in it.combinations(arrays, i)]
    
    return combs

###################################################################################################################
###################################################################################################################

def train_test_split(df, testSetSize, extTestSetSize,external_validation, as_dataframe):
    """
    Method that splits a dataset into random training and test set or random training, test and external test set
    df: datafram that constins the data
    testSetSize: defines the size of the test size
    extTestSetSize: defines the size of the external test size
    external_validation: boolean if true generates an external test set
    as_dataframe: boolean if true returns training and test sets as pandas dataframes
    
    Returns:
    The training and test set or the training, test and external test set.
    """
    
    test_ind = np.random.RandomState(200).choice(len(df), size= int(np.round(len(df)*testSetSize)),replace=False)
    
    df.reset_index(inplace=True)
    df = df.rename(index=int,columns={'index':'oldIndex'})
    train_ind = [i for i in range(len(df)) if i not in test_ind]
    
    if external_validation:
        extTest_ind = np.random.RandomState(200).choice(len(train_ind), size= int(np.round(len(test_ind)*(1+extTestSetSize-testSetSize))),replace=False)
        extTest_ind = [train_ind[i] for i in extTest_ind]
        train_ind = [i for i in train_ind if i not in extTest_ind]
        
        cols = [i for i in df.columns]
        if as_dataframe:
            X_train = df.loc[train_ind,cols]
            X_train.reset_index(inplace=True,drop=True)
            X_test = df.loc[test_ind,cols]
            X_test.reset_index(inplace=True,drop=True)
            X_extTest = df.loc[extTest_ind,cols]
            X_extTest.reset_index(inplace=True,drop=True)
        else:
            X_train = df.loc[train_ind,cols].as_matrix()
            X_test = df.loc[test_ind,cols].as_matrix()
            X_extTest = df.loc[extTest_ind,cols].as_matrix()
            
        return X_train, X_test, X_extTest
    else:
        cols = [i for i in df.columns]
        if as_dataframe:
            X_train = df.loc[train_ind,cols]
            X_train.reset_index(inplace=True,drop=True)
            X_test = df.loc[test_ind,cols]
            X_test.reset_index(inplace=True,drop=True)
        else:
            X_train = df.loc[train_ind,cols].as_matrix()
            X_test = df.loc[test_ind,cols].as_matrix()
            
        return X_train, X_test
    
def assignRandomValues(df,indexColumns,labelColumns,partialVarColumns,missingValues):
    """
    Method that generates random missing values to a dataset
    Inputs:
    df: input data dataframe
    indexColumns: columns that are indices not used in random generation of missing values
    labelColumns: label columns not to be used in tandom gneeration
    partialVarColumns: list of the columns that will undergo partial variability (range), so random missing values are 
                        not generated on these columns
    missingValues: boolean indicating whether to introduce missing values or not
    
    Return:
    Output:
    df: dataframe with missing values
    """
    if missingValues:

        cols = [i for i in df.columns if i not in indexColumns + labelColumns]

        rand = np.random.RandomState(200).randint(len(cols), size=(len(df),len(cols)))

        for i in range(len(df)):
            rand[i][rand[i]==np.random.RandomState(200).randint(len(cols))] = -1

        indxs = np.where( rand < 0 )

        dfMiss = df.copy()

        for i,j in zip(indxs[0],indxs[1]):
            dfMiss.iloc[i,j+len(indexColumns)] = np.nan

        for i in partialVarColumns:
            dfMiss[partialVarColumns] = df[partialVarColumns]

        return dfMiss
    else:
        return df
    
###################################################################################################################
###################################################################################################################    
    
def getRace(x):
    """
    Method to impute race based on most frequent race
    x: input dataframe
    
    Returns:
    dataframe updated to include the variable race.
    """
    ### picking the most frequent race
    for i in tqdm_notebook(range(len(x))):
        if x.loc[i,'white']==1:
            x.loc[i,'race'] = 'white'

        elif x.loc[i,'hispanic']==1:
            x.loc[i,'race'] = 'hispanic'

        elif x.loc[i,'black']==1:
            x.loc[i,'race'] = 'black'

        elif x.loc[i,'asian'] == 1:
            x.loc[i,'race'] = 'asian'

    #    elif x.loc[i,'amindian'] == 1:
    #        x.loc[i,'race'] = 'white'

        else:
            x.loc[i,'race'] = 'white'
    return x

###################################################################################################################
###################################################################################################################  

def plotDistributionCI(X, theshold, ymax, bins):
    """
    Method that plots the distribution of the probabilities and the C.I. and mean.
    
    X: data 1*d
    threshold: threshold value
    
    Returns:
    Plot of the distribution
    """
    
    plt.style.use('ggplot')

    plt.figure(figsize=(12,6))
    plt.hist(X,alpha=0.5,bins=bins,color='blue',label='5 year risk distribution')

    plt.vlines( x=theshold,ymin=0, ymax=ymax, alpha=0.8, color='red', linewidth='3', label='Threshold')

    plt.title('Distribution of Risk with 1.67 threshold')
    plt.xlabel('5 Year Risk value')
    plt.ylabel('density')
    plt.legend()
    plt.show()

def plotDistributionCIRisk(X, ymax, bins):
    """
    Method that plots the distribution of the probabilities and the C.I. and mean.
    
    X: data 1*d
    threshold: threshold value
    
    Returns:
    Plot of the distribution
    """

    plt.style.use('ggplot')

    plt.figure(figsize=(18,6))
    plt.hist(X,alpha=0.5,bins=bins,color='blue',label='5 year risk distribution')
    
    mean, _, conf_int = confidenceInterval(X,alpha=0.95)
    print('Mean: ', mean)
    print('Median: ', np.median(X))
    print('Number of risk values: ', len(X))
    print('C.I.: ', conf_int)
    plt.vlines( x=conf_int[0],ymin=0, ymax=ymax, alpha=0.8, color='green', linewidth='3',label='95% C.I. of median')
    plt.vlines( x=conf_int[1],ymin=0, ymax=ymax, alpha=0.8, color='green', linewidth='3')
    plt.vlines( x=np.median(X),ymin=0, ymax=ymax, alpha=0.8, color='black', linewidth='3'
               , label= 'median = ' + str(np.round(mean,2)))
    plt.vlines( x=1.67,ymin=0, ymax=ymax, alpha=0.8, color='red', linewidth='3', label='1.67 Threshold')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Distribution of Risk with 1.67 threshold', fontsize=20)
    plt.xlabel('5 Year Risk value', fontsize=16)
    plt.ylabel('density', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.show()