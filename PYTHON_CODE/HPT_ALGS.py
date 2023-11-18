#####################################################################################################
######################################## HPT Algorithms code ########################################
#####################################################################################################
############################## useful libraries ################################
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import skopt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error,r2_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from tpot import TPOTClassifier
######################### function for confusion matrices #######################
def conf_heatmap(Ytest,y_pred,dataset,model_name,tuner): 
    plt.figure()
    sns.heatmap(confusion_matrix(Ytest,y_pred), annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Y-predicted')
    plt.ylabel('Y-test')
    plt.title('Confusion Matrix-'+dataset+'-'+model_name+'-'+tuner)
    plt.savefig('ALL_CONFUSION_MATRICES\\Confusion Matrix_'+dataset+'_'+model_name+'_'+tuner+'.png')
############################## GBC parameters #################################
GBCParamGrid={
        'n_estimators': list(np.concatenate((np.arange(1,11),np.arange(20, 101, 10)))),
        'max_depth': list(np.arange(1, 5)),
        'learning_rate': list(np.round(np.arange(0.01, 0.51, 0.02), 2)),
        'max_features':list(np.round(np.arange(0.1, 1.01, 0.05), 2)),
        'min_samples_split': list(np.concatenate((np.arange(2,11),np.arange(20, 101, 10)))),
        'min_samples_leaf': list(np.concatenate((np.arange(1,11),np.arange(20, 101, 10)))),
        }
############################## RF parameters ###################################
RFParamGrid={
    'n_estimators': list(np.concatenate((np.arange(1,11),np.arange(20, 101, 10)))),
    'max_depth': list(np.arange(1, 5)),
    'max_features':list(np.round(np.arange(0.1, 1.01, 0.05), 2)),
    'min_samples_split': list(np.concatenate((np.arange(2,11),np.arange(20, 101, 10)))),
    'min_samples_leaf': list(np.concatenate((np.arange(1,11),np.arange(20, 101, 10))))
    }
############################## XGB parameters ##################################
XGBParamGrid={
    'n_estimators':list(np.concatenate((np.arange(1,11),np.arange(20, 101, 10)))),
    'max_depth': list(np.arange(1, 5)),
    'learning_rate': list(np.round(np.arange(0.01, 0.51, 0.02), 2)),
    'colsample_bylevel':list(np.round(np.arange(0.25, 1.01, 0.05), 2)),
    'colsample_bytree': list(np.round(np.arange(0.25, 1.01, 0.05), 2))
    }
################################ models ####################################
model_xgb=[xgb.XGBClassifier(),XGBParamGrid,"xgboost.XGBClassifier","XGB"]
model_rf=[RandomForestClassifier(),RFParamGrid,"sklearn.ensemble.RandomForestClassifier","RF"]
model_GBC=[GradientBoostingClassifier(),GBCParamGrid,"sklearn.ensemble.GradientBoostingClassifier","GBC"]
ml_list=[model_xgb,model_rf,model_GBC]# creation of the list of models
############################ Data Management ##############################
test_size=0.3
datadir="transformed_data"
data_list=os.listdir(datadir)# creation of the list of datasets
################ creating experiment folders #################
os.mkdir("ALL_METRICS")
os.mkdir("ALL_BARPLOTS")
os.mkdir("ALL_TUNING_CURVES")
os.mkdir("ALL_INDIVIDUALS_CSV_FILES")
os.mkdir("ALL_PARAMETERS")
os.mkdir("ALL_CONFUSION_MATRICES")
############## creating the experiment files #################
for i in ["XGB","RF","GBC"]:
    fcol=[i+"_TIME","TPOT","RCV","BAYES","dataset_mean",
          i+"_TRAIN","TPOT","RCV","BAYES","dataset_mean",
          i+"_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_VALIDATION","TPOT","RCV","BAYES","dataset_mean",
          i+"_MAE_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_R^2_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_PEARSON_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_P-VALUE_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_PRECISION_TEST","TPOT","RCV","BAYES","dataset_mean",
          i+"_RECALL_TEST","TPOT","RCV","BAYES","dataset_mean"]
    exl=pd.DataFrame(fcol)
    exl.to_excel('ALL_METRICS\\'+i+'_METRICS.xlsx',index=False,)
    fcol=[i+"_PARAMETERS","TPOT","RCV","BAYES"]
    exl=pd.DataFrame(fcol)
    exl.to_excel('ALL_PARAMETERS\\'+i+'_PARAMETERS.xlsx',index=False,)
print("$$$$ EXCEL FILES CREATED $$$$")
############################## Loop ##################################
col=1
for data in data_list:# selecting dataset
    # loading the dataset and doing train_test_split
    exl=pd.read_excel(datadir+"\\"+data)
    exl=exl.sample(frac=1.0)
    X=exl.iloc[:, :-1]
    Y=exl.iloc[:,-1]
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=test_size,)
    
    data=os.path.splitext(data)[0]
    data
    start_hpt=time.time()

    for model,grid,tpot_model,model_name in ml_list:# selecting a model
        # useful lists for saving the new models iformation
        times=[data]
        train_accs=[data]
        test_accs=[data]
        validation_acc=[data]
        mae=[data]
        r2=[data]
        pearson=[data]
        pvalue=[data]
        parameters=[data]
        curves=[]
        precision=[data]
        recall=[data]
        ################################
        # TPOTClassifier() #
        TPOT=TPOTClassifier(generations= 5,
                            population_size=30,
                            offspring_size=15,
                            n_jobs=1,verbosity= 2,
                            template='Classifier',
                            config_dict = {tpot_model:grid},
                            cv = 5, scoring = 'accuracy')
        start_time = time.time()
        TPOT.fit(Xtrain,Ytrain)
        end_time = time.time()
        execution_time = end_time - start_time
        times.append(execution_time)
        train_accs.append(TPOT.score(Xtrain,Ytrain))
        test_accs.append(TPOT.score(Xtest,Ytest))
        validation_acc.append(TPOT._optimized_pipeline_score)
        y_pred = TPOT.predict(Xtest)
        conf_heatmap(Ytest,y_pred,data,model_name,'TPOT')
        precision.append(precision_score(Ytest, y_pred))
        recall.append(recall_score(Ytest, y_pred))
        mae.append(mean_absolute_error(Ytest, y_pred))
        r2.append(r2_score(Ytest, y_pred))
        p,v = pearsonr(Ytest, y_pred)
        pearson.append(p)
        pvalue.append(v)
        parameters.append(str(TPOT.fitted_pipeline_.steps))
        evaluated_individuals=pd.DataFrame.from_dict(TPOT.evaluated_individuals_.values())
        n_iter=evaluated_individuals.shape[0]
        evaluated_individuals.to_csv('ALL_INDIVIDUALS_CSV_FILES\\eval_indiv_TPOT_'+model_name+'_'+data+'.csv',
                                     index=False)
        curves.append(sorted(list(evaluated_individuals['internal_cv_score'])))
        
        # RandomizedSearchCV() #
        # setting the searching parameters
        RS=RandomizedSearchCV(estimator=model,
                              param_distributions=grid,
                              n_iter=n_iter,
                              n_jobs=1,
                              cv=5,scoring='accuracy',
                              verbose=1)
        start_time = time.time()
        RS.fit(Xtrain,Ytrain)# searching for best model
        end_time = time.time()
        execution_time = end_time - start_time
        times.append(execution_time)# keeps time
        train_accs.append(RS.score(Xtrain,Ytrain))# keeps accuracy on train set
        test_accs.append(RS.score(Xtest,Ytest))# keeps accuracy on test set
        validation_acc.append(RS.best_score_)# keeps best model's cross validation accuracy
        y_pred = RS.predict(Xtest)# gets predictions
        conf_heatmap(Ytest,y_pred,data,model_name,'RS')# creates confusion matrix
        precision.append(precision_score(Ytest, y_pred))# keeps precision
        recall.append(recall_score(Ytest, y_pred))# keeps recall
        mae.append(mean_absolute_error(Ytest, y_pred))# keeps mean absolute error
        r2.append(r2_score(Ytest, y_pred))# keeps R^2
        p,v = pearsonr(Ytest, y_pred)
        pearson.append(p)# keeps pearson correlation coefficient
        pvalue.append(v)# keeps p-value
        parameters.append(str(RS.best_params_))# keeps best model's parameters
        evaluated_individuals=pd.DataFrame.from_dict(RS.cv_results_.values())# saves CSVs
        evaluated_individuals.to_csv('ALL_INDIVIDUALS_CSV_FILES\\eval_indiv_RS_'+model_name+'_'+data+'.csv',
                                     index=False)
        curves.append(sorted(list(RS.cv_results_['mean_test_score'])))# keeping values for plots

        # skopt.BayesSearchCV() #
        BAYES=skopt.BayesSearchCV(estimator=model, 
                                  search_spaces=grid,
                                  n_iter=n_iter,
                                  n_jobs=1,cv=5,
                                  scoring='accuracy',
                                  verbose=1)
        start_time = time.time()
        BAYES.fit(Xtrain,Ytrain)
        end_time = time.time()
        execution_time = end_time - start_time
        times.append(execution_time)
        train_accs.append(BAYES.score(Xtrain,Ytrain))
        test_accs.append(BAYES.score(Xtest,Ytest))
        validation_acc.append(BAYES.best_score_)
        y_pred = BAYES.predict(Xtest)
        conf_heatmap(Ytest,y_pred,data,model_name,'BAYES')
        precision.append(precision_score(Ytest, y_pred))
        recall.append(recall_score(Ytest, y_pred))
        mae.append(mean_absolute_error(Ytest, y_pred))
        r2.append(r2_score(Ytest, y_pred))
        p,v = pearsonr(Ytest, y_pred)
        pearson.append(p)
        pvalue.append(v)
        parameters.append(str(dict(BAYES.best_params_)))
        evaluated_individuals=pd.DataFrame.from_dict(BAYES.cv_results_.values())
        evaluated_individuals.to_csv('ALL_INDIVIDUALS_CSV_FILES\\eval_indiv_BAYES_'+model_name+'_'+data+'.csv',
                                     index=False)
        curves.append(sorted(list(BAYES.cv_results_['mean_test_score'])))
        ############################################################################################
        ######################### create list of new metrics ############################
        addition=[]
        for i in [times,train_accs,test_accs,validation_acc,mae,r2,pearson,pvalue,precision,recall]:
            i.append(np.mean(i[1:]))
            addition=addition+i
        ############ update files by adding the new metrics and parameters ##############
        all_metrics=pd.read_excel('ALL_METRICS\\'+model_name+'_METRICS.xlsx')
        all_metrics[col]=addition
        all_metrics.to_excel('ALL_METRICS\\'+model_name+'_METRICS.xlsx',index=False)
        all_params=pd.read_excel('ALL_PARAMETERS\\'+model_name+'_PARAMETERS.xlsx')
        all_params[col]=parameters
        all_params.to_excel('ALL_PARAMETERS\\'+model_name+'_PARAMETERS.xlsx',index=False)
        ###################################### VISUALIZATION #######################################
        ############ tuning history - line diagrams ###########
        plt.figure()
        linewidth=4
        for curve in curves:
            x=range(1, len(curve)+1)
            plt.plot(x,curve,marker="x",linewidth=linewidth,linestyle=':')
            linewidth=linewidth-1
        plt.xlabel('MODELS TRAINED')
        plt.ylabel('VALIDATION ACCURACY')
        plt.legend(['TPOT tuning curve','RS tuning curve','BAYES tuning curve'])
        plt.title('TUNING HISTORY FOR '+model_name+' ON '+data+' DATASET')
        plt.savefig('ALL_TUNING_CURVES\\TUC_'+model_name+'_'+data+'.png')
        ################### time barplots ###################
        plt.figure()
        labels = ['TPOT','RS','BAYES']
        plt.bar(labels,times[1:4])
        plt.xlabel('HPT ALGORITHMS')
        plt.ylabel('TUNING TIME')
        plt.title(model_name+' '+data+' DATASET')
        plt.savefig('ALL_BARPLOTS\\TIB_'+model_name+'_'+data+'.png')
    col=col+1
    finish_hpt=time.time()
    print("The models for this dataset trained in :",finish_hpt-start_hpt,"seconds")
print("$$$$ HyperParameter tuning finished $$$$")       
#####################################################################################################
############################################### FINISH ##############################################
#####################################################################################################