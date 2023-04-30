import glob, os
import shutil

# Define and change to the path of MSCam data folder
path_MSCam = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\MSCam'
os.chdir(path_MSCam)

# Define path to generic label file
path_generic_2BL = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\MSCam\MSCam Unused Files\generic.2BL'

p = []
for filename in glob.glob('*.2B'):
    filename_bare = filename.removesuffix('.2B')
    if (filename_bare + '.2BL') not in glob.glob('*.2BL'):
        path_2BL_out = path_generic_2BL.replace('\MSCam Unused Files\generic.2BL', (filename_bare + '.2BL'))
        print('Creating label for', path_2BL_out, end=' ')
        shutil.copyfile(path_generic_2BL, path_2BL_out)
        print('DONE!')
print('2BL label generation complete.')