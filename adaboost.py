def find_stump(df):
    
    ### This performs a greedy search to find the best stump classifier
    ###
    ### Input: df = pd df with columns L, W, Y, and a column of weights
    ###
    ### Output: 3-tuple of ints (var, index, value)
    ###         var = the X the stump is built on
    ###         index = the row of the cut value
    ###         val = the cut value
    ###
    ### This fcn will be called in adaboost_trainer()
    
    #copy the df
    train = df.copy()
    
    #keep this list so we can iterate over the features
    preds = ["L", "W"]
    
    #Initialize a stump for the greedy search
    best_stump = {"variable" : "foo",
                  "index" : -1,
                  "cutpoint" : -1,
                  "Yhat" : pd.DataFrame(),
                  "w_err" : math.inf}
                  
    for pred in preds:
             
        #For each row of that variable
        for index, row in train.iterrows():
            
            #The value of this variable in this row is the cut point
            cut = row[pred]
            #We will compare all values of this variable to this cut point
            val = train[pred]
                
            #Terminal node containing y=1,y=0 counts
            #For those rows with var value > value of current row
            y1_gt = sum(np.where((val >= cut) & (train['Y'] == 1), 1, 0))
            y0_gt = sum(np.where((val >= cut) & (train['Y'] == -1), 1, 0))

            #Terminal node containing y=1,y=0 counts
            #For those rows with var values < value of current row
            y1_lt = sum(np.where((val < cut) & (train['Y'] == 1), 1, 0))
            y0_lt = sum(np.where((val < cut) & (train['Y'] == -1), 1, 0))
            
            
            #Find the class assignment of each node
            if y1_gt >= y0_gt:
                gtnode = 1
            else:
                gtnode = -1
            
            if y1_lt >= y0_lt:
                ltnode = 1
            else:
                ltnode = -1
            
            #Classify all rows based on this stump
            yhat = np.where(val >= cut, gtnode, ltnode)
            
            #Weighted error of this stump
            err_num = sum(train['weight'] * (train['Y'] != yhat))
            err_denom = sum(train['weight'])
            w_err = err_num  / err_denom
            
            
            #create a dict to hold the info of this stump
            this_stump = {"variable" : pred,
                          "index" : index,
                          "cutpoint" : cut,
                          "Yhat" : yhat,
                          "w_err" : w_err,
                          "y1gt" : y1_gt,
                          "y0gt" : y0_gt,
                          "y1lt" : y1_lt,
                          "y0lt" : y0_lt,
                          "gtnode" : gtnode,
                          "ltnode" : ltnode}
            
            
            if this_stump['w_err'] < best_stump['w_err']:
                best_stump = this_stump
                
            
    return(best_stump)

#print(find_stump(train1))


def adaboost_trainer(training, B):
        
    ### This fcn uses boosted stumps to classify testing data
    ###
    ### Input: training = pd df with 4 numeric columns
    ###        testing = pd df with 4 numeric columns (same as train)
    ###        B = int (number of iterations)
    ###
    ### Output: a list of stumps (dictionaries)
    ###         a list of weights for the stumps (ints)
    ###         
    ### This fcn will be called by adaboost()
    
    # Copy the training and testing data
    train = training.copy()
    
    # Initialize weights at 1/N for all rows
    train['weight'] = 1 / len(train)
        
    # Initialize a column:
    # For the mth iteration, save am * Yhat
    train['amGm'] = 0

    #Initialize a list of stumps (the weak learners)
    stump_list = []
    
    #Initialize a list of importance values
    am_list = []
    
    # Iterate a user given # times
    for i in range(0, B):
        
        # Find the best stump in the training data
        stump = find_stump(train)
        
        #append this stump to the list of stumps
        stump_list.append(stump)
        
        
        #unpack the stump a bit
        err = stump['w_err']
        train['Yhat'] = stump['Yhat']
        
        #update the weights, add am to running list
        am =  (1/2)*math.log((1 - err) / err)
        exp_am_err = np.exp(am * (train['Yhat'] != train['Y']))
        train['weight'] = train['weight'] * exp_am_err 
        
        #Append to the list of importance values
        am_list.append(am)
        
        #update the running sum of am*Gm
        train['amGm'] = train['amGm'] + (am * train['Yhat'])
        
    
    #Do a final update of the predictions
    #for each row, Yhat = sign(sum(amGm))
    train['Yhat_new'] = np.where(train['amGm'] > 0, 1, -1)
    
    
    #train_err = sum(train['Y'] != train['Yhat_new']) / len(train)
    
        
    return(am_list, stump_list)



def adaboost(training, testing, B):
    
    ### This fcn performs classification on testing data based on
    ### a set of stumps and importances generated using the adaboost_trainer
    ###
    ### Input: training = pd df to train the algorithm
    ###        testing = pd df to test the classifier on
    ###
    ### Output: int (the error rate for testing classification)

    
    #copy the data frames
    train = training.copy()
    test = testing.copy()
    
    #first call adaboost_trainer
    classifier = adaboost_trainer(train, B)
    
    #extract the list of importances
    am_list = classifier[0]
    
    #extract the list of stumps
    stump_list = classifier[1]
    
    #initialize the weighted sum of weak classifiers
    test['amGm'] = 0
    
    #iterate over the length of the am_list (same as stump_list)
    #we classify using the stump, then weight the result with am
    for i in range(0, len(am_list)):
        
        cut = stump_list[i]['cutpoint']
        var = stump_list[i]['variable']
        gtnode = stump_list[i]['gtnode']
        ltnode = stump_list[i]['ltnode']
        am = am_list[i]
        
        
        test['amGm'] += am * np.where(test[var] >= cut, gtnode, ltnode)
    
    
    test['Yhat'] = np.where(test['amGm'] > 0, 1, -1)
    
    err = sum(test['Yhat'] != test['Y']) / len(test)
    
    return(err)