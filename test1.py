import os
import shutil
from pds4_tools import pds4_read
import untangle

# Read MCS weather data
path_MCS_test = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\SA_Sample_Data\level2_data\MCS'
path_MCS_test_mod = os.path.join(path_MCS_test, 'Modified')

path_files_2C = []
path_files_2C_out = []
path_files_2CL = []

# Collect file paths in arrays
for file in os.listdir(path_MCS_test):
    path_file = os.path.join(path_MCS_test, file)
    path_file_out = os.path.join(path_MCS_test_mod, file)
    if file.endswith('.2C'):
        path_files_2C.append(path_file)
        path_files_2C_out.append(path_file_out)
    if file.endswith('.2CL'):
        path_files_2CL.append(path_file)

# Change NUL values in data to 999 to avoid pds4_read issues
for i in range(len(path_files_2C)):
    with open(path_files_2C[i], 'rt') as fin:
        with open(path_files_2C_out[i], 'wt') as fout:
            #print("Exporting", path_files_2C_out[i])
            shutil.copy(path_files_2CL[i], path_MCS_test_mod)
            for line in fin:
                fout.write(line.replace('NUL', '999'))