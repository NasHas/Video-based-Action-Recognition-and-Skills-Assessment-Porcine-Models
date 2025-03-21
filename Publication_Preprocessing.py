'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

import csv
import itertools
import pathlib

import pandas as pd
import os
import glob
import pathlib
from sys import maxsize
import numpy as np
from numpy.ma import column_stack


def convert_to_csv(file_path,dir_path,dataframe_sum_numbers):
    ### Suturing ######################################################################################
    df = pd.read_excel(file_path)
    file_name = os.path.basename(file_path)
    file_name = file_name.replace(".xlsx","")
    df1 = df[(df['Behavior'] == 'S hånd') | (df['Behavior'] == 'S hÃ¥nd') | (df['Behavior'] == 'S fort') | (
                df['Behavior'] == 'S enkelt') | (df['Behavior'] == 'S sutur')] # Behavior
    df1 = df1.drop(df.columns[[0,1, 2, 3, 4,5, 6, 7, 8,10,11,13,14,15,16]], axis=1)
    df1['Time'] = df1['Time'].astype(float)
    df1[df1.columns[1]] = df1[df1.columns[1]].round(1)
    # Unique values
    df_uniq_values = df1['Behavior'].unique()
    df_dict = {}
    for val in df_uniq_values:
        # Name to dataframe autogen-name
        df_dict[val] = df1.loc[df1['Behavior'] == val]
    # Data dictionary to fill of intervals
    temp_dict = {}
    # Loop through datafram and fill data intervals
    for key in df_dict:
        listtemp = df_dict[key]["Time"].values.tolist() #Time
        for i in range(0, len(listtemp), 2):
            if (i < (len(listtemp) - 1)):
                if listtemp[i + 1] - listtemp[i] > 5:
                    if((listtemp[i + 1] - listtemp[i])%5) != 0:
                        tal = list(range(int(listtemp[i]), int(listtemp[i + 1] - (listtemp[i + 1] - listtemp[i]) % 5) + 1))  # Fills interval through range(start,stop)
                        df_temp = pd.DataFrame()
                        df_temp["time"] = tal
                        df_temp["Behavior"] = key
                        temp_dict[f"{key}+{i}"] = df_temp
                    else:
                        tal = list(range(int(listtemp[i]), int(listtemp[i + 1]+1)))  # Fills interval through range(start,stop)
                        df_temp = pd.DataFrame()
                        df_temp["time"] = tal
                        df_temp["Behavior"] = key
                        temp_dict[f"{key}+{i}"] = df_temp
    df_final_sutur = pd.DataFrame()
    # merge all dataframe that are filled with data
    for key in temp_dict:
        df_final_sutur = pd.concat([df_final_sutur, temp_dict[key]], axis=0, ignore_index=True)


    ### Dissection ######################################################################################
    df2 = df[(df['Behavior'] == 'D alm') | (df['Behavior'] == 'D klips') | (df['Behavior'] == 'D hæm') | (
                df['Behavior'] == 'D hÃ¦m')]
    df2 = df2.drop(df.columns[[0,1, 2, 3, 4,5, 6, 7, 8,10,11,13,14,15,16]], axis=1)
    df2['Time'] = df2['Time'].astype(float)
    df2[df2.columns[1]] = df2[df2.columns[1]].round(1)
    # Unique values
    df_uniq_values = df2['Behavior'].unique()
    df_dict = {}
    for val in df_uniq_values:
        # Name to dataframe autogen-name
        df_dict[val] = df2.loc[df2['Behavior'] == val]
    # Data dictionary to fill of intervals
    temp_dict = {}
    # Loop through datafram and fill data intervals
    for key in df_dict:
        listtemp = df_dict[key]["Time"].values.tolist()
        for i in range(0, len(listtemp), 2):
            if (i < (len(listtemp) - 1)):# Not last element
                if listtemp[i + 1] - listtemp[i] > 5:# Sequence of 5 frames
                    if ((listtemp[i + 1] - listtemp[i]) % 5) != 0: # Check that frame inverval is of 5 equal frames
                        tal = list(range(int(listtemp[i]), int(listtemp[i + 1] - (listtemp[i + 1] - listtemp[i]) % 5) + 1))  # Fills interval through range(start,stop)
                        df_temp = pd.DataFrame()
                        df_temp["time"] = tal
                        df_temp["Behavior"] = key
                        temp_dict[f"{key}+{i}"] = df_temp
                    else:
                        tal = list(range(int(listtemp[i]),int(listtemp[i + 1] + 1)))  # Fills interval through range(start,stop)
                        df_temp = pd.DataFrame()
                        df_temp["time"] = tal
                        df_temp["Behavior"] = key
                        temp_dict[f"{key}+{i}"] = df_temp
    df_final_dissektion = pd.DataFrame()
    # merge all dataframe that are filled with data
    for key in temp_dict:
        df_final_dissektion = pd.concat([df_final_dissektion, temp_dict[key]], axis=0, ignore_index=True)


    #####################################Union join ######################33
    # df_join_all = pd.concat([df_final_dissektion,df_final_sutur],axis=0,join='outer',ignore_index=True, sort=False)
    if not df_final_dissektion.empty and not df_final_sutur.empty:
        df_join_all = pd.merge(df_final_sutur, df_final_dissektion, on="time", how="outer")
    elif df_final_sutur.empty:
        df_final_dissektion.insert(1, 'Behavior_y', 0)
        df_join_all = df_final_dissektion
    if df_final_dissektion.empty:
        df_final_sutur.insert(2, 'Behavior_y', 0)
        df_join_all = df_final_sutur
    df_join_all = df_join_all.fillna(0)
    # Replace all strings values with one
    for i in df_join_all.columns:
        if i != df_join_all.columns[0]:
            df_uniq_values1 = df_join_all[i].unique()
            for val in df_uniq_values1:
                if type(val) == type('sdsds'):
                    df_join_all = df_join_all.replace(val, 1)
    # Remove 1 og 1 in both columns in Behavier column
    if len(list(df_join_all.columns)) > 1:
        df_join_all = df_join_all.drop(df_join_all[(df_join_all[df_join_all.columns[1]] == 1) & (df_join_all[df_join_all.columns[2]] == 1)].index)
    df_final = pd.DataFrame()
    # df_final['combinaed'] = f"{file_name}"+' frame'+df_join_all[df_join_all.columns[0]].astype(str)+'.jpg,'+df_join_all[df_join_all.columns[1]].astype(str)+','+df_join_all[df_join_all.columns[2]].astype(str)
    df_final['combined'] = f"{file_name}" + ' frame' + df_join_all[df_join_all.columns[0]].astype(str) + '.jpg'
    df_final.insert(1, "sutur", df_join_all[df_join_all.columns[1]])
    df_final.insert(2, "dissektion", df_join_all[df_join_all.columns[2]])
    df_final.to_csv(dir_path+f'\\{file_name}.csv', index=False, header=False)
    new_row = pd.DataFrame({'Filname':[file_name], 'Sutur':[df_final.sum()[1]], 'Dissektion':[df_final.sum()[2]]})
    dataframe_sum_numbers = pd.concat([dataframe_sum_numbers,new_row],ignore_index=True)
    return dataframe_sum_numbers

#Find max sum in subarray function
def find_largest_sum_subarray_closest_to_target(df, column_name, target):
    df = df.sample(frac=1)
    df = df.drop(df[(df[column_name]==0)].index)
    closest_sum = float('inf')
    max_sum = float('-inf')
    current_sum = 0

    start_index = 0
    end_index = 0
    current_start = 0

    for i, value in enumerate(df[column_name]):
        current_sum += value

        if abs(current_sum - target) < abs(closest_sum - target):
            closest_sum = current_sum
            start_index = current_start
            end_index = i

        if current_sum > max_sum:
            max_sum = current_sum

        if current_sum < 0:
            current_sum = 0
            current_start = i + 1

    largest_subarray_data = df[column_name][start_index:end_index + 1]

    return start_index, end_index, closest_sum, largest_subarray_data, max_sum

# Define this directory path VERY IMPORTANT!!!
directory_path = "PATH"

#Get list of .xlsx files
excel_files_list = glob.glob(os.path.join(directory_path, "*.xlsx"))

#Dataframe to save sums
dataframe_sum_numbers = pd.DataFrame(columns=['Filname','Sutur', 'Dissektion'])
#Loop thorugh the xlsx files
for file in excel_files_list:
    dataframe_sum_numbers = convert_to_csv(file,directory_path,dataframe_sum_numbers)

#Sum of sum numbers to find min. value of the 2 classes
#Dataframe_sum_numbers1= dataframe_sum_numbers.sum()

#Random shuffle
shuffeled_DF = dataframe_sum_numbers.sample(frac=1)

#Delete same dissektion and sutur data
#shuffeled_DF = shuffeled_DF.drop(shuffeled_DF[((shuffeled_DF["Dissektion"]>0) & (shuffeled_DF["Sutur"]>0))].index) # fjerner alle hvor der både er dissektion og sutur

#Min. number of the 2 classes
dataframe_sum_min_number = shuffeled_DF.sum().drop("Filname").min()

#Vrain split number
train_number = dataframe_sum_min_number *0.80

#Valid split
valid_number = dataframe_sum_min_number * 0.10

#Test split number
test_number = dataframe_sum_min_number * 0.10

#Find max subarray test sample dissektion
column_name = dataframe_sum_numbers.columns[2]  # Replace with the column name you want to analyze
target = test_number/2  # Replace with the target number
start_index, end_index, closest_sum, largest_subarray_Dessektion_test_sample, max_sum = find_largest_sum_subarray_closest_to_target(shuffeled_DF.drop(shuffeled_DF[((shuffeled_DF["Dissektion"]>0) & (shuffeled_DF["Sutur"]>0))].index), column_name, target)

if start_index is not None and end_index is not None:
    print(f"Largest sum contiguous subarray in column '{column_name}' closest to the target {target}")
    print("Start Index:", start_index)
    print("End Index:", end_index)
    print("Closest Sum:", closest_sum)
    print("Maximum Sum:", max_sum)
    print("Largest Subarray Data:")
    print(largest_subarray_Dessektion_test_sample)
else:
    print(f"No positive subarray found in column '{column_name}'")

#Data_frame test set
Data_fram_test_setD = pd.DataFrame()

#Merge
Data_fram_test_setD = pd.merge(shuffeled_DF, largest_subarray_Dessektion_test_sample, left_index=True,right_index=True, how="inner")
Data_fram_test_setD = Data_fram_test_setD.drop(Data_fram_test_setD.columns[3],axis=1)
sum_DF_Temp = Data_fram_test_setD["Sutur"].sum()
Data_fram_test_setD = Data_fram_test_setD.rename(columns={"Dissektion_x": "Dissektion"})

#Find max subarray test sample sutur
column_name = dataframe_sum_numbers.columns[1]  # Replace with the column name you want to analyze
target = test_number/2  # Replace with the target number
start_index, end_index, closest_sum, largest_subarray_sutur_test_sample, max_sum = find_largest_sum_subarray_closest_to_target(shuffeled_DF.drop(shuffeled_DF[((shuffeled_DF["Dissektion"]>0) & (shuffeled_DF["Sutur"]>0))].index), column_name, target)

if start_index is not None and end_index is not None:
    print(f"Largest sum contiguous subarray in column '{column_name}' closest to the target {target}")
    print("Start Index:", start_index)
    print("End Index:", end_index)
    print("Closest Sum:", closest_sum)
    print("Maximum Sum:", max_sum)
    print("Largest Subarray Data:")
    print(largest_subarray_sutur_test_sample)
else:
    print(f"No positive subarray found in column '{column_name}'")

#Merge sutur####
Data_fram_test_setS = pd.merge(shuffeled_DF, largest_subarray_sutur_test_sample, left_index=True,right_index=True, how="inner")
Data_fram_test_setS = Data_fram_test_setS.drop(Data_fram_test_setS.columns[3],axis=1)
sum_DF_Temp = Data_fram_test_setS["Dissektion"].sum()
Data_fram_test_setS= Data_fram_test_setS.rename(columns={"Sutur_x": "Sutur"})

Final_test_set = pd.concat([Data_fram_test_setD, Data_fram_test_setS], axis=0)
Final_test_set= Final_test_set.sample(frac=1)
Final_test_set.to_csv(directory_path+'\\Final_test_set.csv', columns=['Filname'], index=False, header=False)

# Find out which participants' videos are present in the test-set
list_of_names = pd.DataFrame({"Names":["NAME1","NAME2"]}) #INSERT NAMES
matching_names = []

#Loop to match names and test sample
for substring in list_of_names['Names']:
    matching_filenames = Final_test_set[Final_test_set['Filname'].str.contains(substring, case=False, regex=True)]
    if not matching_filenames.empty:
        matching_names.append(substring)
    #matching_filenames1 = pd.concat([matching_filenames1,substring])

#Dataframe to save matching names
matching_filenames1 = pd.DataFrame({"Names":matching_names})

matching_filenames2 = pd.DataFrame()
for substring in matching_filenames1['Names']:
    matching_filenames = dataframe_sum_numbers[dataframe_sum_numbers['Filname'].str.contains(substring, case=False, regex=True)]
    matching_filenames2 = pd.concat([matching_filenames2, matching_filenames],axis=0)

dataframe_sum_numbers2 = dataframe_sum_numbers.merge(matching_filenames2["Filname"], on="Filname")

df_final_test_sumnumbers = dataframe_sum_numbers[~dataframe_sum_numbers['Filname'].isin(dataframe_sum_numbers2['Filname'])]

############ Validation set ####################
df_final_test_sumnumbers_shuffled = df_final_test_sumnumbers.sample(frac=1)

#Find max subarray validation sample dissektion
column_name = df_final_test_sumnumbers.columns[2]  # Replace with the column name you want to analyze
target = valid_number/2  # Replace with the target number
start_index, end_index, closest_sum, largest_subarray_Dessektion_val_sample, max_sum = find_largest_sum_subarray_closest_to_target(df_final_test_sumnumbers_shuffled.drop(df_final_test_sumnumbers_shuffled[((df_final_test_sumnumbers_shuffled["Dissektion"]>0) & (df_final_test_sumnumbers_shuffled["Sutur"]>0))].index), column_name, target)

if start_index is not None and end_index is not None:
    print(f"Largest sum contiguous subarray in column '{column_name}' closest to the target {target}")
    print("Start Index:", start_index)
    print("End Index:", end_index)
    print("Closest Sum:", closest_sum)
    print("Maximum Sum:", max_sum)
    print("Largest Subarray Data:")
    print(largest_subarray_Dessektion_val_sample)
else:
    print(f"No positive subarray found in column '{column_name}'")

#Data_frame val set
Data_fram_val_setD = pd.DataFrame()

#Merge
Data_fram_val_setD = pd.merge(df_final_test_sumnumbers_shuffled, largest_subarray_Dessektion_val_sample, left_index=True,right_index=True, how="inner")
Data_fram_val_setD = Data_fram_val_setD.drop(Data_fram_val_setD.columns[3],axis=1)
sum_DF_Temp_val = Data_fram_val_setD["Sutur"].sum()
Data_fram_val_setD = Data_fram_val_setD.rename(columns={"Dissektion_x": "Dissektion"})

#Find max subarray test sample sutur
column_name = df_final_test_sumnumbers_shuffled.columns[1]  # Replace with the column name you want to analyze
target = valid_number/2  # Replace with the target number
start_index, end_index, closest_sum, largest_subarray_sutur_val_sample, max_sum = find_largest_sum_subarray_closest_to_target(df_final_test_sumnumbers_shuffled.drop(df_final_test_sumnumbers_shuffled[((df_final_test_sumnumbers_shuffled["Dissektion"]>0) & (df_final_test_sumnumbers_shuffled["Sutur"]>0))].index), column_name, target)

if start_index is not None and end_index is not None:
    print(f"Largest sum contiguous subarray in column '{column_name}' closest to the target {target}")
    print("Start Index:", start_index)
    print("End Index:", end_index)
    print("Closest Sum:", closest_sum)
    print("Maximum Sum:", max_sum)
    print("Largest Subarray Data:")
    print(largest_subarray_sutur_val_sample)
else:
    print(f"No positive subarray found in column '{column_name}'")

#Merge sutur####
Data_fram_val_setS = pd.merge(df_final_test_sumnumbers_shuffled, largest_subarray_sutur_val_sample, left_index=True,right_index=True, how="inner")
Data_fram_val_setS = Data_fram_val_setS.drop(Data_fram_val_setS.columns[3],axis=1)
sum_DF_Temp = Data_fram_val_setS["Dissektion"].sum()
Data_fram_val_setS= Data_fram_val_setS.rename(columns={"Sutur_x": "Sutur"})

Final_val_set = pd.concat([Data_fram_val_setD, Data_fram_val_setS], axis=0)
Final_val_set= Final_val_set.sample(frac=1)
Final_val_set.to_csv(directory_path+'\\Final_val_set.csv', columns=['Filname'], index=False, header=False)

############### TRAINING SET ####################

#Loop to match names and test sample
for substring in list_of_names['Names']:
    matching_filenames = Final_val_set[Final_val_set['Filname'].str.contains(substring, case=False, regex=True)]
    if not matching_filenames.empty:
        matching_names.append(substring)
    #matching_filenames1 = pd.concat([matching_filenames1,substring])

#Dataframe to save matching names
matching_filenames3 = pd.DataFrame({"Names":matching_names})

matching_filenames4 = pd.DataFrame()
for substring in matching_filenames3['Names']:
    matching_filenames = df_final_test_sumnumbers_shuffled[df_final_test_sumnumbers_shuffled['Filname'].str.contains(substring, case=False, regex=True)]
    matching_filenames4 = pd.concat([matching_filenames4, matching_filenames],axis=0)

dataframe_sum_numbers3 = df_final_test_sumnumbers_shuffled.merge(matching_filenames4["Filname"], on="Filname")

df_final_val_sumnumbers = df_final_test_sumnumbers_shuffled[~df_final_test_sumnumbers_shuffled['Filname'].isin(dataframe_sum_numbers3['Filname'])]


df_final_val_sumnumbers= df_final_val_sumnumbers.sort_values(by='Sutur', ascending=False)

if abs(df_final_val_sumnumbers[df_final_val_sumnumbers.columns[1]].sum() - df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].sum()) <= 1000:
    df_final_val_sumnumbers.to_csv(directory_path + '\\Final_train_set.csv', columns=['Filname'], index=False, header=False)

while (df_final_val_sumnumbers[df_final_val_sumnumbers.columns[1]].sum() - df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].sum()) > 2000:
    index_to_remove = df_final_val_sumnumbers[df_final_val_sumnumbers.columns[1]].idxmax()  # You can use another condition here
    df_final_val_sumnumbers = df_final_val_sumnumbers.drop(index_to_remove)

if (df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].sum() - df_final_val_sumnumbers[df_final_val_sumnumbers.columns[1]].sum()) > 2000:
    index_to_remove = df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].idxmax()  # You can use another condition here
    df_final_val_sumnumbers = df_final_val_sumnumbers.drop(index_to_remove)

Final_train_set = df_final_val_sumnumbers
Final_train_set= Final_train_set.sample(frac=1)
Final_train_set.to_csv(directory_path + '\\Final_train_set.csv', columns=['Filname'], index=False, header=False)

# for index, row in df_final_val_sumnumbers.iterrows():
#     if (df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].sum() - df_final_val_sumnumbers[df_final_val_sumnumbers.columns[1]].sum()) >= 200:
#         Final_train_set= df_final_val_sumnumbers[df_final_val_sumnumbers.columns[2]].drop(index=0, inplace=True)
#     else:
#         break


#export function dataframe to csv

def Export_to_CSV(directory_path,DataframeFinal,Name,Dataframe_old):
    '''

    :param directory_path: path to data folder
    :param DataframeFinal: Name of file to export to csv. must be an empty dataframe
    :param Dataframe_old: Dataframe to save to CSV
    :return:
    '''
    for filename_dir in glob.glob(os.path.join(directory_path, "*.csv")):
        filename_withoutCSV = os.path.basename(filename_dir)
        for filename_DF in Dataframe_old[Dataframe_old.columns[0]]:
            if filename_DF in filename_withoutCSV:
                dftest = pd.read_csv(filename_dir, header=None)
                DataframeFinal = pd.concat([DataframeFinal, dftest], axis=0)
    DataframeFinal.to_csv(directory_path + f'\\{Name}.csv', index=False, header=False)

#Save test set
Finally_test_set = pd.DataFrame()
Export_to_CSV(directory_path,Finally_test_set,"Finally_test_set",Final_test_set)

Finally_val_set = pd.DataFrame()
Export_to_CSV(directory_path,Finally_val_set,"Finally_val_set",Final_val_set)

Finally_train_set = pd.DataFrame()
Export_to_CSV(directory_path,Finally_train_set,"Finally_train_set",Final_train_set)





