def perceptron (training, testing, lr = 1, w1 = 1, w2 = 1):
    
    ### This fcn trains a perceptron and classifies testing data
    ###
    ### Input: training = pd df with 4 numeric columns
    ###        testing = pd df with 4 numeric columns (same as train)
    ###        lr = int (the learning rate)
    ###        w1 = int (initial weight for L)
    ###        w2 = int (initial weight for W)
    ###
    ### Output: error = int (prediction error rate for testing data)
    ###         
    
    
    #copy the training df
    train = training.copy()
    #copy the testing df
    test = testing.copy()
    
    #predicted ys are sum of weighted values
    #(yhat = 1*Bias + 1*L + 1*W) for each row
    train['Yhat'] = (train['Bias'] 
                  + (w1 * train['L']) 
                  + (w2 * train['W']) >= 0)
        
    #get the initial error column and number of misclassifications
    train['err'] = train['Y'] - train['Yhat']
    n_misclass = sum(abs(train['err']))
    
    num_updates = 0

    #continue to update the weights
    #stop when we have 0 misclassified
    while n_misclass != 0:
        
        num_updates +=1

        #For each row in the df do:
        for index, row in train.iterrows():
            
            #find the change in weights based on current row
            dw1 = (lr * row['L'] * row['err'])
            dw2 = (lr * row['W'] * row['err'])
            
            
            #then update all weights with this change
            #if row was correctly classified, dw1 = dw2 = 0 so no update
            w1 = w1 + dw1
            w2 = w2 + dw2
        
        
            #New predicted ys from new sum of weighted values
            train['Yhat'] = (train['Bias'] 
                        + (w1 * train['L'])
                        + (w2 * train['W']) >= 0)
        
            
            #Updated error column 
            #and updated number of misclassifications
            train['err'] = train['Y'] - train['Yhat']
            n_misclass = sum(abs(train['err']))
            
            #if we have 0 misclassifications before
            #finishing all the rows, break out of for loop
            if n_misclass == 0:
                break
        
    
    #apply weights to testing set to get predictions
    test['Yhat'] = (test['Bias']
                    + (w1 * test['L'])
                    + (w2 * test['W']) >= 0)
    
    
    test_err_rt = sum(abs(test['Y'] - test['Yhat'])) / len(test)
    
    return(test_err_rt, num_updates)