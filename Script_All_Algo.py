import scipy
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy import stats
import scipy.stats as st
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, coo_matrix
from numpy.linalg import qr as QR_decomp


############################################################
def getPivotMonths(DF, time_column, N_TMonths):
    pivotMonths_list = []
    ts = DF[time_column]
    for n in range(1, N_TMonths+1):
        pivotMonth = ts.max() - pd.DateOffset(months=n)
        pivotMonths_list.append(pivotMonth)
    return pivotMonths_list

def Time_DataSplit(DF,time_column,pivotMonths_list,N_TMonths,n_train):
    ΔA_list = []
    ts = DF[time_column]
    A0_df = DF.loc[ts < pivotMonths_list[-1]]
    ΔA1 = DF.loc[ts >= pivotMonths_list[0]]
    ΔA_list.append(ΔA1)
    for i in range(N_TMonths-1):
        ΔA = DF.loc[(ts >= pivotMonths_list[i+1]) & (ts < pivotMonths_list[i])]
        ΔA_list.append(ΔA)
    ΔA_list =  ΔA_list[::-1]          ##reverse order..
    ΔA_train = ΔA_list[:n_train]
    ΔA_test = ΔA_list[n_train:]
    return A0_df,ΔA_train,ΔA_test

def TestTrain_DataSplit(DF, user_column, time_column, pivotMonths_list, ΔA_test):
    SVDTrain_list  = []
    PSITest_list   = []
    HOLDOUT_list   = []
    ts = DF[time_column]
    for test_ in ΔA_test:
        train_ = DF.loc[ts < test_[time_column].min()]
        test_sorted = test_.sort_values(time_column)
        test_idx = [x[-1] for x in test_sorted.index.groupby(test_sorted[user_column]).values()]

        holdout = test_.loc[test_idx]
        psi_test = test_.drop(test_idx)
        svdtrain = pd.concat([train_,test_.drop(test_idx)], axis = 0)

        SVDTrain_list.append(svdtrain)
        PSITest_list.append(psi_test)
        HOLDOUT_list.append(holdout)
    return SVDTrain_list,PSITest_list,HOLDOUT_list

#################################################################################

def SingleRatingMatrix(DF,user_column,product_column,rating_column,rows_,cols_):  ##rows_ = n_users,cols_ = n_items
    rows0 = DF[user_column].values
    cols0 = DF[product_column].values
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        # there were duplicates accumulated by .tocsr() -> need to make it implicit
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix


def AllRatingMatrices(DFList,user_column,product_column,rating_column,rows_ ,cols_):
    Rating_matrix_list = []
    for df in DFList:
        df_Mat = SingleRatingMatrix(df,user_column,product_column,rating_column, rows_, cols_)
        Rating_matrix_list.append(df_Mat)
    return Rating_matrix_list               #return the list of Rating matrices


#################################################################################
def pureSVD(SVDRatingMatrices, k):
    Vsvd_list = []
    for Rating_Mat in SVDRatingMatrices:
        Usvd, Ssvd, VTsvd = svds(Rating_Mat, k=k)
        Vsvd = VTsvd.T

        Vsvd_list.append(Vsvd)
    return Vsvd_list


######################################################################
                                                                                ##INPUTS: factorization of the rank-r matrix Y(0) = USV and the increment ΔA 
def integrator(U0,S0,V0,ΔA):
    K1 = U0 @ S0 + ΔA @ V0                #1st step is to find K1 from inital inputs...
    U1,S1_cap =  QR_decomp(K1)            #compute the QR_decomposition of K1 
    S0_tilde = S1_cap - U1.T @ ΔA @ V0
    L1 = V0 @ S0_tilde.T + ΔA.T @ U1
    V1,S1_T =  QR_decomp(L1)                  #compute the QR_decomposition of L1
    S1 = S1_T.T
    return U1,S1,V1

def getStartingValues(A0,k):
    U, S, VT = svds(A0,k=k)
    V = VT.T
    S = np.diag(S)
    return U,S,V


def integratorOnMat(A0,ΔA_train_matrix,ΔA_test_matrix,k):
    U,S,V = getStartingValues(A0,k)          ##technically the starting point U, S, V here are U0, S0, V0T
    for ΔA in ΔA_train_matrix:
        U,S,V = integrator(U,S,V,ΔA)            ##the last U,S,V from this ΔA_train are the starting elements for the ΔA_test  
        
    V_list = []  
    for ΔA in ΔA_test_matrix:
        U,S,V = integrator(U,S,V,ΔA)
        V_list.append(V)
    return V_list

############################################################################################
def topsort(a, n):
    parted = np.argpartition(a, -n)[-n:]
    return parted[np.argsort(-a[parted])]

topN_Index = topsort

def TopNPred(RatingMat,holdout,V, user_column, N):  ##N == Top_N
    TestUsers = holdout[user_column]
    HOLDOUT_usersMat = RatingMat[TestUsers,:]         ##this doubles as the "previously seen items"
    PVVT =  HOLDOUT_usersMat.dot(V).dot(V.T)
    users_column = HOLDOUT_usersMat.nonzero()[0]
    items_column = HOLDOUT_usersMat.nonzero()[1]
    args = np.array([users_column,items_column])
    np.put(PVVT, np.ravel_multi_index(args, PVVT.shape),-np.inf)   ##downsample previously seen items
    TopN_pred = np.apply_along_axis(topsort, 1,PVVT,n = N)
    return TopN_pred

def getALLTopNPred(RatingMat_List,HOLDOUT_list,V_list,user_column,N):
    All_TOPN_PRED = []
    for RatingMat,holdout,V in zip(RatingMat_List,HOLDOUT_list,V_list):
        TopN_pred =  TopNPred(RatingMat,holdout,V, user_column, N)
        All_TOPN_PRED.append(TopN_pred)
    return All_TOPN_PRED

#################################################################################################

def Sample_Hitrate(Holdout,TopN_pred,user_column,item_column):
    Eval_itemsVector  =  Holdout[[item_column]].to_numpy()
    HitRate_arr   = (TopN_pred == Eval_itemsVector).sum(axis=1)  ##sum along row...
    HitCount = np.count_nonzero(HitRate_arr == 1)
    HitRate_ = HitRate_arr.mean()
    return HitRate_


def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        mean, se = np.mean(a), st.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        lower_band =  mean-h
        upper_band =  mean+h
        return lower_band, mean,upper_band                ##LowerBand || Mean || UpperBand


def SAMPLE_All_HitRate(HOLDOUT_list,All_TOPN_PRED,user_column,item_column):
    AllSteps_Hitrate = []
    for Holdout, TopN_pred in zip(HOLDOUT_list,All_TOPN_PRED):
        HitRate_ = Sample_Hitrate(Holdout,TopN_pred,user_column,item_column)
        AllSteps_Hitrate.append(round(HitRate_,6))
    LowerBand, Avg_HitRate, UpperBand  = mean_confidence_interval(AllSteps_Hitrate, confidence=0.95)
    LowerBand   =  round(LowerBand,6)
    Avg_HitRate = round(Avg_HitRate,6)
    UpperBand   =  round(UpperBand,6)
    return AllSteps_Hitrate, LowerBand, Avg_HitRate, UpperBand

#########################################################################################################

def SVDoptimalSearch(SVDtrain_MatList,HOLDOUT_list, start_value,MAX_RANK,increment,user_column,item_column,N):
    AllSteps_HITRATE_ = []
    AVG_HITRATE_LIST  = []
    LOWERBAND_LIST    = []
    UPPERBAND_LIST    = []
    RANK_LIST = []
    print("HitRate Evaluation : ")   
    for rank in tqdm(range(start_value,MAX_RANK+1,increment)):
        Vsvd_list = pureSVD(SVDtrain_MatList,k=rank)
        ALLPredsvd = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vsvd_list,user_column,N)
        AllSteps_Hitrate, LowerBand, Avg_HitRate, UpperBand = SAMPLE_All_HitRate(HOLDOUT_list,ALLPredsvd,user_column,item_column)

        RANK_LIST.append(rank)
        AllSteps_HITRATE_.append(AllSteps_Hitrate)
        LOWERBAND_LIST.append(LowerBand)
        AVG_HITRATE_LIST.append(Avg_HitRate)
        UPPERBAND_LIST.append(UpperBand)
    return AllSteps_HITRATE_, RANK_LIST,LOWERBAND_LIST, AVG_HITRATE_LIST,UPPERBAND_LIST

def PSIoptimalSearch(SVDtrain_MatList,HOLDOUT_list,A0_Rating_matrix,PSI_train_matrix,PSI_test_matrix,start_value,MAX_RANK,increment,user_column,item_column,N):
    AllSteps_HITRATE_ = []
    AVG_HITRATE_LIST  = []
    LOWERBAND_LIST    = []
    UPPERBAND_LIST    = []
    RANK_LIST = []
    print("HitRate Evaluation : ") 
    for rank in tqdm(range(start_value,MAX_RANK+1,increment)):
        Vpsi_list =  integratorOnMat(A0_Rating_matrix,PSI_train_matrix,PSI_test_matrix,k=rank)
        ALLPredPSI = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vpsi_list,user_column,N)  ##SVDtrain_MatList == UserHistMat
        AllSteps_Hitrate, LowerBand, Avg_HitRate, UpperBand  = SAMPLE_All_HitRate(HOLDOUT_list,ALLPredPSI,user_column,item_column)

        RANK_LIST.append(rank)
        AllSteps_HITRATE_.append(AllSteps_Hitrate)
        LOWERBAND_LIST.append(LowerBand)
        AVG_HITRATE_LIST.append(Avg_HitRate)
        UPPERBAND_LIST.append(UpperBand)
    return AllSteps_HITRATE_, RANK_LIST,LOWERBAND_LIST, AVG_HITRATE_LIST,UPPERBAND_LIST

############################# Random Recommendation #############################################

def TopN_RandomPred(RatingMat,user_column,N):  ##RandRec for all users
    N_users,N_items = RatingMat.shape   
    PVVT_RandScores = np.random.rand(N_users,N_items)   ##Assigns random scores to items
    users_column = RatingMat.nonzero()[0]
    items_column = RatingMat.nonzero()[1]
    args = np.array([users_column,items_column])
    np.put(PVVT_RandScores, np.ravel_multi_index(args, PVVT_RandScores.shape),-np.inf)   ##downsample previously seen items
    TopN_pred = np.apply_along_axis(topsort, 1,PVVT_RandScores,n = N)
    return TopN_pred

def get_ALLRandPred(RatingMat_List,user_column,N):
    All_RandPred = []
    for RatingMat in tqdm(RatingMat_List):  
        Rand_pred =  TopN_RandomPred(RatingMat,user_column, N)
        All_RandPred.append(Rand_pred)
    return All_RandPred

def getAll_RandomHitRate(HOLDOUT_list,All_RandPred,user_column,item_column):
    AllSteps_Hitrate = []
    for Holdout, Random_pred in zip(HOLDOUT_list,All_RandPred):  
        TestUsers = Holdout[user_column]
        HOLDOUT_RandPred = Random_pred[TestUsers,:]    
        HitRate_ = Sample_Hitrate(Holdout,HOLDOUT_RandPred,user_column,item_column)
        AllSteps_Hitrate.append(HitRate_)

    LowerBand, Avg_HitRate, UpperBand  = mean_confidence_interval(AllSteps_Hitrate, confidence=0.95)
    return AllSteps_Hitrate, LowerBand, Avg_HitRate, UpperBand    

def getAll_RandomRecMRR(HOLDOUT_list,All_RandPred,user_column,item_column):
    AllSteps_MRR = []
    for Holdout, Random_pred in tqdm(zip(HOLDOUT_list,All_RandPred)):  
        TestUsers = Holdout[user_column]
        HOLDOUT_RandPred = Random_pred[TestUsers,:]    
        MRR_ = MRR_Eval(Holdout,HOLDOUT_RandPred,item_column)
        AllSteps_MRR.append(round(MRR_,6))

    LowerBand, Avg_MRR, UpperBand  = mean_confidence_interval(AllSteps_MRR, confidence=0.95)
    return AllSteps_MRR, LowerBand, Avg_MRR, UpperBand       


########################### MOST POPULAR_ITEM RECOMMENDATION ###################################

def getMOSTPOP_Pred(DF,user_column,item_colum,Nusers,N):  ##get the most popular item at a particular step
    top_counts= DF.groupby(item_colum)[user_column].count()    
    top_items = top_counts.sort_values(ascending=False) 
    MostPOP_Items = top_items[:N].index.values
    MostPOP_Pred = np.array([MostPOP_Items,]*Nusers)      ##ASSIGN FOR ALL USER
    return MostPOP_Pred


def getAll_MOSTPOP_Pred(DF_list,user_column,item_colum,Nusers,N):
    All_MostPOPRED_List = []
    for DF in tqdm(DF_list):
        MostPOP_Pred = getMOSTPOP_Pred(DF,user_column,item_colum,Nusers,N)
        All_MostPOPRED_List.append(MostPOP_Pred) 
    return All_MostPOPRED_List


def getAll_MostPOPHitRate(HOLDOUT_list,All_MostPOPRED_List,user_column,item_column):
    AllSteps_Hitrate = []
    for Holdout, MostPOP_pred in zip(HOLDOUT_list,All_MostPOPRED_List): 
        TestUsers = Holdout[user_column]
        Holdout_MostPOPred =  MostPOP_pred[TestUsers,:]  
        HitRate_ = Sample_Hitrate(Holdout,Holdout_MostPOPred,user_column,item_column)
        AllSteps_Hitrate.append(HitRate_)
    LowerBand, Avg_HitRate, UpperBand  = mean_confidence_interval(AllSteps_Hitrate, confidence=0.95)
    return AllSteps_Hitrate, LowerBand, Avg_HitRate, UpperBand    


def getAll_MostPOP_MRR(HOLDOUT_list,All_MostPOPRED_List,user_column,item_column):
    AllSteps_MRR = []
    for Holdout, MostPOP_pred in tqdm(zip(HOLDOUT_list,All_MostPOPRED_List)): 
        TestUsers = Holdout[user_column]
        Holdout_MostPOPred =  MostPOP_pred[TestUsers,:]   
        MRR_ = MRR_Eval(Holdout,Holdout_MostPOPred,item_column)
        AllSteps_MRR.append(MRR_)
    LowerBand, Avg_MRR, UpperBand  = mean_confidence_interval(AllSteps_MRR, confidence=0.95)
    return AllSteps_MRR, LowerBand, Avg_MRR, UpperBand  

###########################################################################################




                        ############ STABILITY #################
        
def CORRTestTrain_DataSplit(DF,user_column,time_column,pivotMonths_list,ΔA_test,Noffset):
    SVD_CORRTr_list  = []
    PSI_CORRTst_list   = []
    Corr_HOLDOUT   = []
    ts = DF[time_column]
    for test_ in ΔA_test:
        train_ = DF.loc[ts <= test_[time_column].max()]   ###get everydata untill the end of the time step
        train_sorted = train_.sort_values(time_column)
        holdout_idx = train_sorted[user_column].drop_duplicates(keep='last').index
        
        holdout = train_.loc[holdout_idx]
        svdtrain = train_.drop(holdout_idx)
        StopMonth = svdtrain[time_column].max() - pd.DateOffset(months=Noffset)
        psi_test = svdtrain.loc[svdtrain[time_column]>= StopMonth]
        
        SVD_CORRTr_list.append(svdtrain)
        PSI_CORRTst_list.append(psi_test)
        Corr_HOLDOUT.append(holdout)
    return SVD_CORRTr_list, PSI_CORRTst_list,Corr_HOLDOUT

def getDiff_Users(SVD_CORRTr_list,ΔA_test,user_column,Nsteps):
    ACTIVE_USERS_list  =  []
    INACTIVE_USERS_list = []   
    for i in range(Nsteps):
        All_users = SVD_CORRTr_list[i][user_column].unique()
        Active_users = ΔA_test[i][user_column].unique()
        Inactive_users = np.setdiff1d(All_users,Active_users)  

        ACTIVE_USERS_list.append(Active_users)
        INACTIVE_USERS_list.append(Inactive_users)
    return ACTIVE_USERS_list,INACTIVE_USERS_list
                   

def TopNPred_ALLUSERS(RatingMat,V, N):  ##Prediction for all users ...||Not just Holdout
    PVVT =  RatingMat.dot(V).dot(V.T) 
    users_column = RatingMat.nonzero()[0]
    items_column = RatingMat.nonzero()[1]
    args = np.array([users_column,items_column])
    np.put(PVVT, np.ravel_multi_index(args, PVVT.shape),-np.inf)   ##downsample previously seen items
    TopN_pred = np.apply_along_axis(topsort, 1,PVVT,n = N)
    return TopN_pred

    
def getALLTopNPred_ALLUSERS(RatingMat_List,V_list,N):
    All_TOPN_PRED = []
    for RatingMat,V in zip(RatingMat_List,V_list):   
        TopN_pred =  TopNPred_ALLUSERS(RatingMat,V, N)
        All_TOPN_PRED.append(TopN_pred)
    return All_TOPN_PRED    


def no_copy_csr_matrix(data, indices, indptr, shape, dtype):
        # set data and indices manually to avoid index dtype checks
        # and thus prevent possible unnecesssary copies of indices
        matrix = csr_matrix(shape, dtype=dtype)
        matrix.data = data
        matrix.indices = indices
        matrix.indptr = indptr
        return matrix

def build_rank_weights_matrix(recommendations, shape):
        recommendations = np.atleast_2d(recommendations)
        n_users, topn = recommendations.shape
        weights_arr = 1. / np.arange(1, topn+1) # 1 / rank
        weights_mat = np.lib.stride_tricks.as_strided(weights_arr, (n_users, topn), (0, weights_arr.itemsize))

        data = weights_mat.ravel()
        indices = recommendations.ravel()
        indptr = np.arange(0, n_users*topn + 1, topn)

        weight_matrix = no_copy_csr_matrix(data, indices, indptr, shape, weights_arr.dtype)
        return weight_matrix

def rank_weighted_jaccard_index(inds1, inds2):
        shape = inds1.shape[0], max(inds1.max(), inds2.max())+1
        weights1 = build_rank_weights_matrix(inds1, shape)
        weights2 = build_rank_weights_matrix(inds2, shape)
        jaccard_index = weights1.minimum(weights2).sum(axis=1) / weights1.maximum(weights2).sum(axis=1)
        return np.asarray(jaccard_index).squeeze()


def getAll_AvgCorr(All_PRED,Corr_steps_):
    All_Corr_List = []
    for step in Corr_steps_:
        corr_result = rank_weighted_jaccard_index(All_PRED[step-1], All_PRED[step])  ##corr values for all users at each dual step
        All_Corr_List.append(corr_result) 
    AUserC_arry = np.array(All_Corr_List).T      
    return AUserC_arry      ##corr values for all users at every step for a single rank  ##rows == users || column : steps


#####All Ranks CorrrScores 
def SVDCorr_4AllRanks(SVDTrCorr_MatList,Corr_steps_,start_value,MAX_RANK,increment,N, SAVE_name):
    print("Correlation for Allusers: ")
    All_UsersCorrSVD = []
    for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
        Vsvd_list = pureSVD(SVDTrCorr_MatList,k=rank) 
        ALLPredsvd = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vsvd_list,N)
        AUsersCorr_ = getAll_AvgCorr(ALLPredsvd,Corr_steps_)  #Avg_AAUsersSVD
        All_UsersCorrSVD.append(AUsersCorr_)   ##this returns list of array
    np.savez_compressed(SAVE_name,All_UsersCorrSVD)     
    return All_UsersCorrSVD 


def PSICorr_4AllRanks(SVDTrCorr_MatList,A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,Corr_steps_,start_value,MAX_RANK,increment,N,SAVE_name):
    print("Correlation for Allusers: ")
    All_UsersCorrPSI = []
    for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
        Vpsi_list =  integratorOnMat(A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,k=rank) 
        ALLPredPSI = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vpsi_list,N)   ##SVDtrain_MatList == UserHistMat
        AUsersCorr_ = getAll_AvgCorr(ALLPredPSI,Corr_steps_)    #Avg_AAUsersPSI
        All_UsersCorrPSI.append(AUsersCorr_)
    np.savez_compressed(SAVE_name,All_UsersCorrPSI) 
    return All_UsersCorrPSI
##############################################################################

                                        ## COVERAGE  RATIO  ##

def StepCoverage_Ratio(DF,Step_Pred,item_column):
    nItems_tot  = DF[item_column].nunique()
    nPred_items = len(np.unique(Step_Pred))
    coverage_ratio = round((nPred_items/nItems_tot),6)
    return coverage_ratio

def AllSteps_Coverage_Ratio(DF,All_Pred,item_column):
    All_CoverageRatio_List = []
    for Step_Pred in All_Pred:
        step_CI = StepCoverage_Ratio(DF,Step_Pred,item_column)
        All_CoverageRatio_List.append(step_CI)
    return All_CoverageRatio_List


def SVD_AllRankCovRatio(DF,SVDtrain_MatList,HOLDOUT_list,user_column,item_column,start_value,max_rank,increment,N):
    CoverageRatio_List = []
    print("Coverage Ratio : ")  
    for rank in tqdm(range(start_value,max_rank+1,increment)): 
        Vsvd_list = pureSVD(SVDtrain_MatList,k=rank) 
        ALLPredsvd = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vsvd_list,user_column,N)
        SVD_CR =  AllSteps_Coverage_Ratio(DF,ALLPredsvd,item_column)
        CoverageRatio_List.append(SVD_CR)
    return CoverageRatio_List


def PSI_AllRankCovRatio(DF,SVDtrain_MatList,HOLDOUT_list,A0_Rat_mat,PSI_train_mat,PSI_test_mat,user_column,item_column,start_value,max_rank,increment,N):
    CoverageRatio_List = []
    print("Coverage Ratio : ")
    for rank in tqdm(range(start_value,max_rank+1,increment)): 
        Vpsi_list =  integratorOnMat(A0_Rat_mat,PSI_train_mat,PSI_test_mat,k=rank) 
        ALLPredPSI = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vpsi_list,user_column,N)   ##SVDtrain_MatList == UserHistMat
        PSI_CR =  AllSteps_Coverage_Ratio(DF,ALLPredPSI,item_column)
        CoverageRatio_List.append(PSI_CR)
    return CoverageRatio_List
#################################################################################

             ########### MEAN RECIPROCAL RANK #################

def MRR_Eval(Holdout,TopN_pred,item_column):
    Ntest_users = Holdout.shape[0]
    Eval_itemsVector  =  Holdout[[item_column]].to_numpy()
    item_pos = np.where(Eval_itemsVector == TopN_pred)[1] +1  #eval item pos in pred_index1 
    if item_pos.size:  ##if any hit
       Hit_RR = (1/item_pos)  
       MRR_ = np.divide(np.sum(Hit_RR),Ntest_users)  #mean of all reciprocal rank
       return MRR_    
    return 0

def getAll_MRR_Eval(HOLDOUT_list,All_TOPN_PRED,item_column):
    AllSteps_MRR = []
    for Holdout, TopN_pred in zip(HOLDOUT_list,All_TOPN_PRED):  
        MRR_ = MRR_Eval(Holdout,TopN_pred,item_column)
        AllSteps_MRR.append(round(MRR_,6))

    LowerBand, Avg_MRR, UpperBand  = mean_confidence_interval(AllSteps_MRR, confidence=0.95)
    LowerBand = round(LowerBand,6)
    Avg_MRR  = round(Avg_MRR,6)
    UpperBand =round(UpperBand,6)
    return AllSteps_MRR, LowerBand, Avg_MRR, UpperBand    
            

def SVDAllRank_MRR(SVDtrain_MatList,HOLDOUT_list,start_value,max_rank,increment,user_column,item_column,N):
    AllSteps_MRR_  = []
    AVG_MRR_LIST   = []
    LOWERBAND_LIST = []
    UPPERBAND_LIST = []
    RANK_LIST = []
    print("Mean Reciprocal Rank: ")  
    for rank in tqdm(range(start_value,max_rank+1,increment)): 
        Vsvd_list = pureSVD(SVDtrain_MatList,k=rank) 
        ALLPredsvd = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vsvd_list,user_column,N)
        AllSteps_MRR, LowerBand, Avg_MRR, UpperBand  = getAll_MRR_Eval(HOLDOUT_list,ALLPredsvd,item_column)

        RANK_LIST.append(rank)
        AllSteps_MRR_.append(AllSteps_MRR)
        LOWERBAND_LIST.append(LowerBand)
        AVG_MRR_LIST.append(Avg_MRR)
        UPPERBAND_LIST.append(UpperBand)
    return AllSteps_MRR_, RANK_LIST,LOWERBAND_LIST, AVG_MRR_LIST,UPPERBAND_LIST


def PSIAllRank_MRR(SVDtrain_MatList,HOLDOUT_list,A0_Rat_mat,PSI_tr_mat,PSI_tst_mat,start_value,max_rank,increment,user_column,item_column,N):
    AllSteps_MRR_ = []
    AVG_MRR_LIST  = []
    LOWERBAND_LIST    = []
    UPPERBAND_LIST    = []
    RANK_LIST = []
    print("Mean Reciprocal Rank: ") 
    for rank in tqdm(range(start_value,max_rank+1,increment)): 
        Vpsi_list =  integratorOnMat(A0_Rat_mat,PSI_tr_mat,PSI_tst_mat,k=rank) 
        ALLPredPSI = getALLTopNPred(SVDtrain_MatList,HOLDOUT_list,Vpsi_list,user_column,N)  ##SVDtrain_MatList == UserHistMat
        AllSteps_MRR, LowerBand, Avg_MRR, UpperBand  = getAll_MRR_Eval(HOLDOUT_list,ALLPredPSI,item_column)

        RANK_LIST.append(rank)
        AllSteps_MRR_.append(AllSteps_MRR)
        LOWERBAND_LIST.append(LowerBand)
        AVG_MRR_LIST.append(Avg_MRR)
        UPPERBAND_LIST.append(UpperBand)
    return AllSteps_MRR_, RANK_LIST,LOWERBAND_LIST, AVG_MRR_LIST,UPPERBAND_LIST


##############################################################################

def SingelStep_StagnantCheck(All_PRED,DF_name,Nusers):
    First_Pred = All_PRED[0] 
    Last_Pred = All_PRED[-1]
    corr_result = [stats.spearmanr(First_Pred[i, :],Last_Pred[i,:],axis=1) [0] for i in range(Nusers)]
    corr_resultDF = pd.DataFrame(corr_result)
    corr_resultDF = corr_resultDF.rename(columns={0:'Users_FirstLastCorr'})
    corr_resultDF.to_csv(DF_name, index=False)
    return corr_resultDF    

def getHitRate_csv(RANK_LIST,LOWERBAND_LIST, AVG_HITRATE_LIST,UPPERBAND_LIST, DF_name):
    COMBINED_DF = pd.DataFrame(list(zip(RANK_LIST,LOWERBAND_LIST,AVG_HITRATE_LIST,UPPERBAND_LIST)),
                                                columns =['Rank_Values','HitR_LowerBand','AvgHitRate','HitR_UpperBand'])

    COMBINED_DF.to_csv(DF_name, index=False)
    return COMBINED_DF


def getAll_csv(RANK_LIST,LOWERBAND_LIST, AVG_HITRATE_LIST,UPPERBAND_LIST,DF_name):
    COMBINED_DF = pd.DataFrame(list(zip(RANK_LIST,LOWERBAND_LIST,AVG_HITRATE_LIST,UPPERBAND_LIST)), 
                                                columns =['Rank_Values','LowerBand','Avg_values','UpperBand'])

    COMBINED_DF.to_csv(DF_name, index=False)
    return COMBINED_DF


def get_StepwiseCSV(Stepwise_List,step_label,RANK_list,DF_name): 
    STEP_list = ['Step_%s' % s for s in step_label]
    DF_ = pd.DataFrame(Stepwise_List)
    DF_.columns=  STEP_list 
    DF_['Step_Avg'] = DF_.mean(axis=1)
    DF_.insert(loc=0, column='Rank', value=RANK_list)
    DF_.to_csv(DF_name, index=False)
    return DF_   


def get_SingleStep_CSV(SingleSteps_value,LowerBand_HR, Avg_HR, UpperBand_HR,step_label,DF_name): 
    STEP_list = ['Step_%s' % s for s in step_label]
    DF_ = pd.DataFrame(SingleSteps_value).T
    DF_.columns=  STEP_list 
    DF_['HR_LowerBand'] = LowerBand_HR
    DF_['Avg_HitRate']  = Avg_HR
    DF_['HR_UpperBand'] = UpperBand_HR
    DF_.to_csv(DF_name, index=False)
    return DF_

def get_Any_SingleStepCSV(SingleSteps_value,step_label,DF_name): 
    STEP_list = ['Step_%s' % s for s in step_label]
    DF_ = pd.DataFrame(SingleSteps_value).T
    DF_.columns=  STEP_list 
    DF_['Step_Avg'] = DF_.mean(axis=1)
    DF_.to_csv(DF_name, index=False)
    return DF_

        
#####################################################################################
def get_args():
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--algorithm')
        
        parser.add_argument('--data_path')
        
        parser.add_argument('--df_')
        
        parser.add_argument('--n_tmonths', type=int)
        
        parser.add_argument('--Nsteps',default=7, type=int)

        parser.add_argument('--time_column',default='timestamp')

        parser.add_argument('--n_train',type= int)

        parser.add_argument('--n',default=10,type= int)
        
        parser.add_argument('--start_value',default=10,type= int)

        parser.add_argument('--increment', default=5, type= int)

        parser.add_argument('--max_rank', default=80, type= int)
        
        parser.add_argument('--Noffset', default=1, type= int)

        parser.add_argument('--user_column')

        parser.add_argument('--item_column')

        parser.add_argument('--step_start', type=int, default=1)
        
        parser.add_argument('--step_end', type=int, default=9 )

        return parser.parse_args()



def main():

    args = get_args()
    algorithm = args.algorithm
    data_path = args.data_path 
    df_ =   args.df_
    time_column = args.time_column
    n_tmonths = args.n_tmonths
    Nsteps = args.Nsteps
    n_train = args.n_train
    max_rank = args.max_rank
    increment = args.increment
    start_value = args.start_value
    user_column = args.user_column
    item_column = args.item_column
    n = args.n

    Noffset = args.Noffset
    step_start =  args.step_start
    step_end =  args.step_end

    DF = pd.read_csv(data_path)
    DF[time_column] =  pd.to_datetime(DF[time_column])

    Nusers = DF['userId'].nunique()     
    rows_ =  DF['userId'].max() + 1
    cols_ =  DF[item_column].max() + 1
    step_label = range(step_start, step_end)
    Corr_steps_ = range(step_start, step_end-1)
    RANK_list = list(range(start_value, max_rank+1,increment))
    
    pivotMonths_list = getPivotMonths(DF, time_column, n_tmonths)
    A0_df, ΔA_train, ΔA_test = Time_DataSplit(DF, time_column, pivotMonths_list, n_tmonths, n_train)
    SVDTrain_list, PSITest_list, HOLDOUT_list = TestTrain_DataSplit(DF,'userId','timestamp', pivotMonths_list, ΔA_test)
    SVDtrain_MatList = AllRatingMatrices(SVDTrain_list,'userId',item_column,'rating', rows_ ,cols_)
    
    SVD_CORRTr_list, PSI_CORRTst_list,Corr_HOLDOUT = CORRTestTrain_DataSplit(DF,'userId','timestamp',pivotMonths_list,ΔA_test,Noffset)
    SVD_CORRTr_MATlist = AllRatingMatrices(SVD_CORRTr_list,'userId',item_column,'rating',rows_ ,cols_)
    AU_list, IU_list   =   getDiff_Users(SVD_CORRTr_list[:-1],ΔA_test[:-1],'userId',Nsteps) 



    
                                    #####################  SVD #########################  
    if algorithm == 'SVD':
        AllSteps_HITRATE_SVD, RANK_LIST_SVD, LOWERBAND_LIST_SVD, AVG_HITRATE_LIST_SVD, UPPERBAND_LIST_SVD =                             SVDoptimalSearch(SVDtrain_MatList, HOLDOUT_list, start_value, max_rank, increment, user_column, item_column, n)
        #1-HitRate        
        SVD_COMBINED_DF =  getHitRate_csv(RANK_LIST_SVD, LOWERBAND_LIST_SVD, AVG_HITRATE_LIST_SVD, UPPERBAND_LIST_SVD, df_+'SVD_AvgHirate.csv')
        SVD_StepwiseHitRate = get_StepwiseCSV(AllSteps_HITRATE_SVD,step_label,RANK_list, df_+'SVD_StepHirate.csv') 
        
        #2-Correlation          
        AllUsers_CorrSVD = SVDCorr_4AllRanks(SVD_CORRTr_MATlist,Corr_steps_,start_value,max_rank,increment,n, df_+'SVD_AllUsersCORR')   
        
        #3-COVERAGE RATIO     
        SVD_CoverageRatio =   SVD_AllRankCovRatio(DF,SVDtrain_MatList,HOLDOUT_list,'userId',item_column,start_value,max_rank,increment,n) 
        SVD_CoverageRatioDF =  get_StepwiseCSV(SVD_CoverageRatio,step_label,RANK_list,df_+'SVD_CovRatio.csv')

            #3-MEAN RECIPROCAL RANK
        AllSteps_MRR_svd, RANK_svd, LOWERBAND_svd, AVG_MRR_svd, UPPERBAND_svd = SVDAllRank_MRR(SVDtrain_MatList,HOLDOUT_list,start_value,max_rank,increment,'userId',item_column,n)
        
        SVD_MRR_DF = getAll_csv(RANK_svd,LOWERBAND_svd, AVG_MRR_svd,UPPERBAND_svd, df_+'SVD_MRR.csv')
        SVD_stepMRR_DF = get_StepwiseCSV(AllSteps_MRR_svd,step_label,RANK_list, df_+'SVD_StepMRR.csv')

            
            
    
                                #######################  PSI #######################  
    if algorithm == 'PSI':
        A0_Rating_matrix = SingleRatingMatrix(A0_df,      'userId',item_column,'rating',rows_, cols_)
        PSI_train_matrix = AllRatingMatrices(ΔA_train,    'userId',item_column,'rating',rows_ ,cols_)
        PSI_test_matrix =  AllRatingMatrices(PSITest_list,'userId',item_column,'rating',rows_ ,cols_)
        
        #1-HitRate        
        AllSteps_HITRATE_PSI, RANK_LIST_PSI, LOWERBAND_LIST_PSI, AVG_HITRATE_LIST_PSI, UPPERBAND_LIST_PSI = PSIoptimalSearch(SVDtrain_MatList, HOLDOUT_list, A0_Rating_matrix, PSI_train_matrix, PSI_test_matrix, start_value, max_rank,increment, user_column, item_column, n)
        PSI_COMBINED_DF = getHitRate_csv(RANK_LIST_PSI, LOWERBAND_LIST_PSI, AVG_HITRATE_LIST_PSI, UPPERBAND_LIST_PSI, df_+'PSI_AvgHirate.csv') 
        PSI_StepwiseHitRate = get_StepwiseCSV(AllSteps_HITRATE_PSI,step_label,RANK_list,df_+'PSI_StepHirate.csv')
        
        #2-Correlation
        PSI_TestCorr_Mat =  AllRatingMatrices(PSI_CORRTst_list,'userId',item_column,'rating',rows_ ,cols_)
        PSI_AllUserCorr = PSICorr_4AllRanks(SVD_CORRTr_MATlist,A0_Rating_matrix,PSI_train_matrix,PSI_TestCorr_Mat,Corr_steps_,start_value,max_rank,increment,n,df_+'PSI_AllUsersCORR')
        
         
        #3-COVERAGE RATIO  
        PSI_CoverageRatio = PSI_AllRankCovRatio(DF,SVDtrain_MatList,HOLDOUT_list,A0_Rating_matrix,PSI_train_matrix, PSI_test_matrix,'userId',item_column,start_value,max_rank,increment,n)                              
        PSI_CoverageRatioDF =  get_StepwiseCSV(PSI_CoverageRatio,step_label,RANK_list,df_+'PSI_CovRatio.csv')
        
        #1-MRR     
        AllSteps_MRR_psi, RANK_psi, LOWERBAND_psi, AVG_MRR_psi, UPPERBAND_psi = PSIAllRank_MRR(SVDtrain_MatList,HOLDOUT_list,A0_Rating_matrix, PSI_train_matrix, PSI_test_matrix,start_value,max_rank,
                                                                increment,'userId',item_column,n)
        
        PSI_MRR_DF =     getAll_csv(RANK_psi,LOWERBAND_psi, AVG_MRR_psi,UPPERBAND_psi,df_+'PSI_MRR.csv') 
        PSI_StepMRR_DF = get_StepwiseCSV(AllSteps_MRR_psi,step_label,RANK_list,df_+'PSI_StepMRR.csv')
           
            
            
                                ##################  Random Recommendation ########################
    if algorithm == 'RandRec':
        #1-HitRate  
        print("HitRate Evaluation : ") 
        All_RandPred = get_ALLRandPred(SVDtrain_MatList,'userId',n)
        RR_AllStepsHR, RR_LowerBand, RR_AvgHR, RR_UpperBand  = getAll_RandomHitRate(HOLDOUT_list,All_RandPred,'userId',item_column)         
        RR_HRDF = get_SingleStep_CSV(RR_AllStepsHR,RR_LowerBand, RR_AvgHR, RR_UpperBand,step_label,df_+'RRec_AvgHirate.csv')
        
        #2-Correlation
        print("Correlation for Allusers : ")
        All_RandPredCorr = get_ALLRandPred(SVD_CORRTr_MATlist,'userId',n)
        RR_AllUsersCorr =   getAll_AvgCorr(All_RandPredCorr,Corr_steps_)
        np.savez_compressed(df_+'RRec_AllUsersCORR',RR_AllUsersCorr)      
        
        #3-COVERAGE RATIO 
        print("Coverage Ratio : ")
        RR_CoverageRatio =   AllSteps_Coverage_Ratio(DF,All_RandPred,item_column)
        RR_CoverageRatioDF =  get_Any_SingleStepCSV(RR_CoverageRatio,step_label,df_+'RRec_CovRatio.csv') 
        
        #1-MRR  
        print("MRR Evaluation : ") 
        RR_AllSteps_MRR, LowerBandmrr, RR_Avg_MRR, UpperBandmrr = getAll_RandomRecMRR(HOLDOUT_list,All_RandPred,'userId',item_column)       
        RR_MRR_DF = get_SingleStep_CSV(RR_AllSteps_MRR, LowerBandmrr, RR_Avg_MRR,UpperBandmrr,step_label,df_+'RRec_MRR.csv')



                                ##################  MOST POPULAR ITEMS Rec ########################
    if algorithm == 'POPRec':
        #1-HitRate 
        print("HitRate Evaluation : ") 
        All_MostPOPRED_List = getAll_MOSTPOP_Pred(PSITest_list,user_column,item_column,Nusers,n)
        MP_HitR, MP_LowerBand, MP_AvgHR, MP_UpperBand = getAll_MostPOPHitRate(HOLDOUT_list,All_MostPOPRED_List,user_column,item_column)
        MP_HRDF = get_SingleStep_CSV(MP_HitR, MP_LowerBand, MP_AvgHR, MP_UpperBand,step_label,df_+'MPRec_AvgHirate.csv') 
        
        #2-Correlation
        print("Correlation for Allusers : ")
        All_MostPOPRED_Corr = getAll_MOSTPOP_Pred(PSI_CORRTst_list,'userId',item_column,Nusers,n)
        MP_AllUsersCorr =   getAll_AvgCorr(All_MostPOPRED_Corr,Corr_steps_)
        np.savez_compressed(df_+'MPRec_AllUsersCORR',MP_AllUsersCorr) 
        
        
        #3-COVERAGE RATIO  
        print("Coverage Ratio : ")
        MP_CoverageRatio =   AllSteps_Coverage_Ratio(DF,All_MostPOPRED_List,item_column)
        MP_CoverageRatioDF =  get_Any_SingleStepCSV(MP_CoverageRatio,step_label,df_+'MPRec_CovRatio.csv')
        
        #4-MRR  
        print("MRR Evaluation : ") 
        MP_Stepmrr, MP_LowerBand, MP_Avg_mrr, MP_UpperBand = getAll_MostPOP_MRR(HOLDOUT_list,All_MostPOPRED_List,user_column,item_column)
        MP_mrrDF = get_SingleStep_CSV(MP_Stepmrr, MP_LowerBand, MP_Avg_mrr, MP_UpperBand,step_label,df_+'MPRec_MRR.csv') 


                                
if __name__ == "__main__":
    main()
    
    
  
    
    

    
        
# def twoSteps_Corr(All_PRED,step,Nusers):
#     Step1_Pred = All_PRED[step-1] 
#     Step2_Pred = All_PRED[step]
#     corr_result = [stats.spearmanr(Step1_Pred[i, :],Step2_Pred[i,:],axis=1) [0] for i in range(Nusers)]
#     return corr_result
    
    
#####uSERS Activeness Ranks CorrrScores 
# def UserActiveness_Corr(All_TOPN_PRED,Corr_steps_,UsersList):
#     USER_StepAvgList = []
#     for step, Users_, in zip(Corr_steps_,UsersList):
#         USER_LastPred =    All_TOPN_PRED[step-1][Users_,:]  ##previous step prediction
#         USER_CurrentPred = All_TOPN_PRED[step][Users_,:]    ##Current step prediction
#         Users_corr =  [stats.spearmanr(USER_LastPred[i, :],USER_CurrentPred[i,:],axis=1)[0] for i in range(len(Users_))]
#         USERs_StepAvg = np.mean(Users_corr)                 ##Avg_ for all users..
#         USER_StepAvgList.append(USERs_StepAvg)
#     return USER_StepAvgList 


# def All_UserActiveness_Corr(All_TOPN_PRED,Corr_steps_,ActiveUsersL,InActiveUsersL):
#     ActiveUSER_AvgList =  UserActiveness_Corr(All_TOPN_PRED,Corr_steps_,ActiveUsersL)
#     InActiveUSER_AvgList = UserActiveness_Corr(All_TOPN_PRED,Corr_steps_,InActiveUsersL)
#     return ActiveUSER_AvgList,InActiveUSER_AvgList



# #######All UsersActivenessRanks CorrrScores 

# def SVD_AllRankUserCorr(ActiveUsersL,InActiveUsersL,SVDTrCorr_MatList,Corr_steps_,start_value,MAX_RANK,increment,N):
#     ActiveUSER_AvgList = []
#     InActiveUSER_AvgList = []
#     print("Correlation for Active and Inactive Users: ")
#     for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
#         Vsvd_list = pureSVD(SVDTrCorr_MatList,k=rank) 
#         ALLPredsvd = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vsvd_list,N)
#         ActiveUSER_AvgSVD, InActiveUSER_AvgSVD = All_UserActiveness_Corr(ALLPredsvd,Corr_steps_,ActiveUsersL,InActiveUsersL)
                
#         ActiveUSER_AvgList.append(ActiveUSER_AvgSVD)
#         InActiveUSER_AvgList.append(InActiveUSER_AvgSVD)
#     return ActiveUSER_AvgList, InActiveUSER_AvgList


# def PSI_AllRankUserCorr(ActiveUsersL,InActiveUsersL,SVDTrCorr_MatList,A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,Corr_steps_,start_value,MAX_RANK,increment,N):
#     ActiveUSER_AvgList = []
#     InActiveUSER_AvgList = []
#     print("Correlation for Active and Inactive Users: ")      
#     for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
#         Vpsi_list =  integratorOnMat(A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,k=rank) 
#         ALLPredPSI = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vpsi_list,N)    ##SVDtrain_MatList == UserHistMat
#         ActiveUSER_AvgPSI, InActiveUSER_AvgPSI = All_UserActiveness_Corr(ALLPredPSI,Corr_steps_,ActiveUsersL,InActiveUsersL)

#         ActiveUSER_AvgList.append(ActiveUSER_AvgPSI)
#         InActiveUSER_AvgList.append(InActiveUSER_AvgPSI)
#     return ActiveUSER_AvgList, InActiveUSER_AvgList



     #### Stanancy Check #### 
# def Stagnacy_Check(All_PRED,nusers):
#     First_Pred = All_PRED[0] 
#     Last_Pred = All_PRED[-1]
#     corr_result = [stats.spearmanr(First_Pred[i, :],Last_Pred[i,:],axis=1) [0] for i in range(nusers)]
#     AA_users = np.mean(corr_result)
#     return AA_users    


# def SVDStagnantCheck_AllRanks(SVDTrCorr_MatList,Corr_steps_,start_value,MAX_RANK,increment,N,Nusers,DF_name):
#     print("Stagnancy Check : ")
#     All_corr_values = []
#     Rank_list = []
#     for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
#         Vsvd_list = pureSVD(SVDTrCorr_MatList,k=rank) 
#         ALLPredsvd = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vsvd_list,N)
#         corr_value = Stagnacy_Check(ALLPredsvd,Nusers)
#         All_corr_values.append(corr_value)
#         Rank_list.append(rank)
#     corr_DF = pd.DataFrame(All_corr_values)
#     corr_DF.insert(loc=0, column='Rank', value=Rank_list)
#     corr_DF.to_csv(DF_name, index=False)
#     return corr_DF


# def PSIStagnantCheck_AllRanks(SVDTrCorr_MatList,A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,Corr_steps_,start_value,MAX_RANK,increment,N,Nusers,DF_name):
#     print("Stagnancy Check : ")
#     All_corr_values = []
#     Rank_list = []
#     for rank in tqdm(range(start_value,MAX_RANK+1,increment)): 
#         Vpsi_list =  integratorOnMat(A0_Corr_Mat,PSI_TrCorr_Mat,PSI_TestCorr_Mat,k=rank) 
#         ALLPredPSI = getALLTopNPred_ALLUSERS(SVDTrCorr_MatList,Vpsi_list,N)      ##SVDtrain_MatList == UserHistMat
#         corr_value = Stagnacy_Check(ALLPredPSI,Nusers)
#         All_corr_values.append(corr_value) 
#         Rank_list.append(rank)
#     corr_DF = pd.DataFrame(All_corr_values)
#     corr_DF.insert(loc=0, column='Rank', value=Rank_list)
#     corr_DF.to_csv(DF_name, index=False)
#     return corr_DF

    
