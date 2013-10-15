"""
This script implement imputation of the 10,000 dataset according to
the questionnaires as well as domain knowledge provided by experts.
Note that after this step there are still some missing values in the dataset.
Then we can assume the missingness are present randomly and we could employ
some statistical/machine learning algorithms to do further imputation,
resulting in a complete dataset.

Inputs:
-------
Final10000_Dataset_12_MAR13.txt: 
    txt file containing the 10,000 dataset
    
Final10000_DataDictionary_12MAR13_MH_PJC_Annotation_June2013.csv:
    csv file containing PJC's annotation of all the features

Outputs:
--------
data_imputation_questionnaire.pkl:
    python pickle file containing the dataset after imputation

features_imputation_questionnaire.csv
    csv file containing information of the remaining features after imputation
    including: name, data type, the number of missing values
"""
print __doc__

from python.COPDGene.utils.read_txt_data import read_txt_data
import numpy as np
from time import time
import csv
from python.COPDGene.utils.check_features_type import check_features_type
import pickle
from python.COPDGene.utils.compute_missingness import compute_missingness

t0 = time()
file_name_data = "/home/changyale/dataset/COPDGene/Final10000_Dataset_12MAR13.txt"
file_name_features = \
        "/home/changyale/dataset/COPDGene/Final10000_DataDictionary_12MAR13_MH_PJC_Annotation_June2013.csv"

# Choose the dataset to use: 'include' or 'include_maybe' or 'whole'
flag_data = 'whole'

#=============Read information about features from file_name_features=========
csvfile = open(file_name_features,'rb')
reader = csv.reader(csvfile)
lines = [line for line in reader]
n_row = len(lines)
n_col = len(lines[0])

# array containg information about all the features
features = np.empty((n_row,n_col),dtype=list)
for i in range(0,n_row):
    features[i,:] = lines[i]

# list containing the names of attributes of features, such as varnum,form,type
f_features = lines[0]

#============Read information about dataset from file_name_data===============
data_1 = read_txt_data(file_name_data)
n_instances_1,n_features_1 = data_1.shape

#print n_instances_1,n_features_1
#for j in range(n_features_1):
#    print j,data_1[0,j]

# Remove instances with CTMissing_Reason != 0
n_instances = 0
for j in range(n_features_1):
    if data_1[0,j] == "CTMissing_Reason":
        for i in range(n_instances_1):
            if data_1[i,j] == '0':
                n_instances += 1

# Include the first row containing feature names
n_instances += 1
n_features = n_features_1
data = np.empty((n_instances,n_features),dtype=list)
index = 0
data[index,:] = data_1[0,:]
index += 1
for j in range(n_features_1):
    if data_1[0,j] == "CTMissing_Reason":
        for i in range(1,n_instances_1):
            if data_1[i,j] == '0':
                data[index,:] = data_1[i,:]
                index += 1

#print index, n_instances

# Compute missingness of dataset after removing samples according to the value
# of "CTMissing_Reason"
n_missing_before_imputation = compute_missingness(data[1:n_instances,:])

# Imputation according to the questionnaires
for i in range(1,n_instances):
    for j in range(1,n_features):
        if data[0,j] == 'Walk_Limit' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['WalkSymp_BackPain','WalkSymp_JointPain',\
                        'WalkSymp_Legs_Fatigue','WalkSymp_ShortnessBreath']:
                    data[i,k] = '0'
        # COPDGene_Demographics_Physical/Oxygen Saturation and Therapy
        if data[0,j] == 'O2_Therapy' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['O2_Hours_Day','O2_Years','O2use_rest',\
                        'O2use_exercise','O2use_sleep']:
                    data[i,k] = '0'
        # COPDGene_Medical_History/Medical History/8-alpha-1 test
        ######################################################################
        if data[0,j] == 'Alpha1Test' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] == 'PhenoGenoDK':
                    data[i,k] = '1'
        # COPDGene_Medical_History/Medical History/9-smoke cigarettes
        # There're parent-daughter structure here,but since the parent feature
        # "Do you currently smoke cigarettes" are missing from the provided 
        # feature list, so no imputation here.
        # PJC: Imputation for SmokeMenthol:important in subset of smoke only,
        # can be set to zero
        if data[0,j] == 'SmokeMenthol' and data[i,j] == '':
            data[i,j] = '0'

        # COPDGene_Medical_History/Medical History/10-joints painful
        if data[0,j] == 'PainfulJoints' and data[i,j] == '':
            data[i,j] = '0'
        if data[0,j] == 'PainfulJoints' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['PainJointType_Shoulder',\
                        'PainJointType_Elbow','PainJointType_Wrist',\
                        'PainJointType_Hip','PainJointType_Knee',\
                        'PainJointType_Foot']:
                    data[i,k] = '0'
        if data[0,j] == 'LowerBackPain' and data[i,j] == '':
            data[i,j] = '0'
        # COPDGene_Medication_History/Medication History/1
        if data[0,j] == 'CurrentMedUse' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['bagonist','bagonistlongact','CombcCSBagon',\
                        'combivent','Cortsterinhal','CortsterOral','ipratrop',\
                        'nebulizer','theoph','tiotrop']:
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease/Respiratory Symptoms/1-cough
        if data[0,j] == 'HaveCough' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'Cough4Plus':
                    data[i,k] = '0'
            # COPDGene_Respiratory_Disease/Respiratory Symptoms/3-cough
            # Only when answers to 'HaveCough','CoughAM_yn','CoughRestDay' are
            # all 'NO', then could we do imputation for 'Cough3Mo','CoughNumYr'
            for m in range(1,n_features):
                if data[0,m] == 'CoughAM_yn' and data[i,m] == '0':
                    for n in range(1,n_features):
                        if data[0,n] == 'CoughRestDay' and data[i,n] == '0':
                            for p in range(1,n_features):
                                if data[0,p] in ['Cough3Mo','CoughNumYr']:
                                    data[i,p] = '0'
        # COPDGene_Respiratory_Disease/Respiratory Symptoms/4-phlegm
        if data[0,j] == 'HavePhlegm' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'PhlegmOften':
                    data[i,k] = '0'
            # COPDGene_Respiratory_Disease/6-phlegm
            # Only when answers to 'HavePhlegm', 'PhlegmAM', 'PhlegmRestDay'
            # are all 'NO', then could we do imputation for 'Phlegm3Mo',
            # 'PhlegmNumYr'
            for m in range(1,n_features):
                if data[0,m] == 'PhlegmAM' and data[i,m] == '0':
                    for n in range(1,n_features):
                        if data[0,n] == 'PhlegmRestDay' and data[i,n] == '0':
                            for p in range(1,n_features):
                                if data[0,p] in ['Phlegm3Mo','PhlegmNumYr']:
                                    data[i,p] = '0'
        # COPDGene_Respiratory_Disease/7-episodes
        if data[0,j] == 'EpisodeCghPhlm' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['NumEpisodeLastYr','YrsEpisodeLast']:
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease/8-wheezing or whistling
        ######################################################################
        # How to set age when age-related features have missing values?
        # We could simply ignore the age-related features
        # If answer to Q8 is NO, then jump to Q11
        # Since features related to age have been excluded, so it doesn't
        # matter what values to fill in
        if data[0,j] == 'ChstWheezyWhist' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'AgeWheezWhistDK':
                    data[i,k] = '1'
                if data[0,k] == 'AgeFirstAttackDK':
                    data[i,k] = '1'
                if data[0,k] in ['ShrtBrthAttk','ShrtBrth2Plus',\
                        'MedorTreatAttack','ChestWheez12mo','WithCold',\
                        'ApartFromCold','MoreOnceAWeek','MostDaysNights']:
                    data[i,k] = '0'
        if data[0,j] == 'WithCold' and data[i,j] == '':
            for k in range(1,n_features):
                if data[0,k] == 'ChestWheez12mo' and data[i,k] == '0':
                    data[i,j] = '0'
        # If answer to Q8 is YES
        if data[0,j] == 'ChstWheezyWhist' and data[i,j] == '1':
            for k in range(1,n_features):
                # If answer to Q9 is NO
                if data[0,k] == 'ShrtBrthAttk' and data[i,k] == '0':
                    for m in range(1,n_features):
                        if data[0,m] == 'AgeFirstAttackDK':
                            data[i,m] = '1'
                        if data[0,m] in ['ShrtBrth2Plus','MedorTreatAttack']:
                            data[i,m] = '0'
                # If answer to Q10 is NO
                if data[0,k] == 'ChestWheez12mo' and data[i,k] == '0':
                    for m in range(1,n_features):
                        if data[0,m] in ['withCold','ApartFromCold',\
                                'MoreOnceAWeek','MostDaysNights']:
                            data[i,m] = '0'
        # If answer to Q16 is NO
        if data[0,j] == 'LtdUphill' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in['LtdLevelSlow','LtdLevelStop','LtdLvlStop100',\
                        'LtdNotLeaveHm']:
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease/COPD Exacerbations in the Last Year
        if data[0,j] == 'ERLungProb' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'ERLungProbtimes':
                    data[i,k] = '0'
        if data[0,j] == 'TreatChestIll' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'TreatChestTimes':
                    data[i,k] = '0'
        if data[0,j] == 'TreatSteroids' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] == 'TreatSteroiTimes':
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease/Severity of COPD Exacerbations in the
        # Last Year
        # For this code-daughter variables can be set to zero
        # PJC: "In addition, because the episodes questions were multiple
        # responses and not very useful as is, we created a new set of
        # variables, called COPDExacTrt1-COPDExacTrt5, which count the number
        # of episodes in which a given treatment was checked(Row 253-2527)"
        if data[0,j] == 'FlareupChestTrb' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['COPDExacTrt1','COPDExacTrt2','COPDExacTrt3',\
                        'COPDExacTrt4''COPDExacTrt5']:
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease/Respiratory Conditions
        if data[0,j] == 'Asthma' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['AsthmaAgeDK','AsthmaDxByDr',\
                        'AsthmaStillHave','AsthmaStopChild',\
                        'AsthmaTreat']:
                    data[i,k] = '0'
        if data[0,j] == 'HayFev' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['HayFevAgeDK','HayFevDxByDr',\
                        'HayFevStillHave','HayFevStill','HayFevTreat']:
                    data[i,k] = '0'
        if data[0,j] == 'BronchAttack' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['BronchDxByDr','BronchAgeDK','BronchTimes']:
                    data[i,k] = '0'
        if data[0,j] == 'Pneumonia' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['PneuDxByDr','PneuAgeDK','PneuTimes']:
                    data[i,k] = '0'
        if data[0,j] == 'ChronBronch' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['ChronBroncDxByDr','ChronBrncStillHv',\
                        'ChronBroncTreat']:
                    data[i,k] = '0'
        if data[0,j] == 'Emphysema' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['EmphDxByDr','EmphStillHave','EmphTreat']:
                    data[i,k] = '0'
        if data[0,j] == 'COPD' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['CopdDxByDr','CopdStillHave','CopdTreat']:
                    data[i,k] = '0'
        if data[0,j] == 'SleepApnea' and data[i,j] in ['0','3']:
            for k in range(1,n_features):
                if data[0,k] in ['SleepApDxByDr','SleepApStillHav',\
                        'SleepApTreat']:
                    data[i,k] = '0'
        # COPDGene_Respiratory_Disease\Environment Exposures
        # Cigarette Smoking
        if data[0,j] == 'EverSmokedCig' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['SmokCigNow','CigperDaySmokNow',\
                        'CigPerDaySmokAvg','CigSmok24hrs_NA','CigSmok24hrs',\
                        'CigSmok2hrs','CigSmokHalfHr']:
                    data[i,k] = '0'
        if data[0,j] == 'EverSmokedCig' and data[i,j] == '1':
            for k in range(1,n_features):
                if data[0,k] == 'SmokCigNow' and data[i,k] == '0':
                    for m in range(1,n_features):
                        if data[0,m] == 'CigPerDaySmokNow':
                            data[i,m] = '0'
        # Pipe Smoking
        if data[0,j] == 'SmokPipeReg' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['SmokPipeNow','QtySmokPipeTob',\
                        'SmokTobPerWeek']:
                    data[i,k] = '0'
        if data[0,j] == 'SmokPipeReg' and data[i,j] == '1':
            for k in range(1,n_features):
                if data[0,k] == 'SmokPipeNow' and data[i,k] == '0':
                    for m in range(1,n_features):
                        if data[0,m] == 'QtySmokPipeTob':
                            data[i,m] = '0'
        # Cigar Smoking
        if data[0,j] == 'SmokCigarReg' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['SmokCigarNow','CigarPerDayNow',\
                        'SmokCigarPerWeek']:
                    data[i,k] = '0'
        if data[0,j] == 'SmokCigarReg' and data[i,j] == '1':
            for k in range(1,n_features):
                if data[0,k] == 'SmokCigarNow' and data[i,k] == '0':
                    for m in range(1,n_features):
                        if data[0,m] == 'CigarPerDayNow':
                            data[i,m] = '0'
        # Second-hand Smoke Exposure
        # NO conditional relationship between this question block
        # Educational and Occupational History
        if data[0,j] == 'WorkNow' and data[i,j] == '0':
            for k in range(1,n_features):
                if data[0,k] in ['WorkDustyJobNow','ExpFumesNow']:
                    data[i,k] = '0'

# parse the dataset according to whether to keep certain features
# information about features are necessary
# features[i,6]=1->include,features[i,7]=1->maybe,features[i,8]=1->exclude
# First extract the information of features
features_include = [0]*n_features
features_maybe = [0]*n_features
features_exclude = [0]*n_features
for i in range(1,n_row):
    if features[i,6] == '1':
        features_include[i-1] = 1
    if features[i,7] == '1':
        features_maybe[i-1] = 1
    if features[i,8] == '1':
        features_exclude[i-1] = 1

# Exclude the features with more than 10% missing values
for j in range(n_features):
    if data[0,j] in ['Number_Nodules','Largest_Nodule','WallAreaPct_subseg',\
            'SF36_PF_score','SF36_RP_score','SF36_RE_score','SF36_SF_score',\
            'SF36_BP_score','SF36_VT_score','SF36_MH_score','SF36_GH_score',\
            'SF36_PF_t_score','SF36_RP_t_score','SF36_RE_t_score',\
            'SF36_SF_t_score','SF36_BP_t_score','SF36_VT_t_score',\
            'SF36_MH_t_score','SF36_GH_t_score','SF36_PCS_score',\
            'SF36_MCS_score'] and features_include[j] == 1:
        features_include[j] = 0
        features_exclude[j] = 1

n_features_include = sum(features_include)
n_features_maybe = sum(features_maybe)
n_features_exclude = sum(features_exclude)

# Second parse the dataset according to information of features
data_include = np.empty((n_instances,n_features_include),dtype=list)
data_include_maybe = np.empty((n_instances,n_features_include+\
        n_features_maybe),dtype=list)
for i in range(n_instances):
    index_include = 0
    index_include_maybe = 0
    for j in range(n_features):
        if features_include[j] == 1:
            data_include[i,index_include] = data[i,j]
            index_include += 1
            data_include_maybe[i,index_include_maybe] = data[i,j]
            index_include_maybe += 1
        if features_maybe[j] == 1:
            data_include_maybe[i,index_include_maybe] = data[i,j]
            index_include_maybe += 1

if flag_data == 'include':
    data_use = data_include
elif flag_data == 'include_maybe':
    data_use = data_include_maybe
elif flag_data == 'whole':
	data_use = data
else:
    print("Error in choosing data to use, please input again!")
    exit

# Compute the number of missing values after imputation according to the
# questionnaries
n_missing_after_imputation = compute_missingness(data)

# Compute the number of missing value for each feature
# Here we use data_use
features_type = check_features_type(data)
pickle.dump([data_use,features_type],open("data_imputation_questionnaire.pkl","wb"))
pickle.dump([list(data[0,:]),
             n_missing_before_imputation,
             n_missing_after_imputation,
             features_type,
             features_include],open("info_missing_type_include.pickle","wb"))

print len(list(data[0,:]))
print len(n_missing_before_imputation)
print len(n_missing_after_imputation)
print len(features_type)
print len(features_include)

n_instances,n_features = data_use.shape
n_missing = [0]*n_features

n_missing_10plus = 0
for j in range(n_features):
    for i in range(1,n_instances):
        if data_use[i,j] == '':
            n_missing[j] += 1
    if n_missing[j]>0.1*n_instances:
        if n_missing_10plus == 0:
            print("Features with more than 10% missing values are as following:")
        print data_use[0,j]
        n_missing_10plus += 1
features = list(data_use[0,:])
final = [features,features_type,n_missing]
item_length = n_features
file_name_result = "features_imputation_questionnaire.csv"
result = open(file_name_result,'wb')
file_writer = csv.writer(result)
file_writer.writerow(["FeatureName","FeatureType","#Missing"])
for i in range(item_length):
    file_writer.writerow([x[i] for x in final])

print(['Running time(min):',(time()-t0)/60.0])
