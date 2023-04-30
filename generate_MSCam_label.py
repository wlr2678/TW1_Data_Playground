import glob, os
import shutil
from lxml import etree

# Define and change to the path of MSCam data folder
path_MSCam = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\MSCam'
os.chdir(path_MSCam)

# Define path to generic label file
path_generic_2BL = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\MSCam\MSCam Unused Files\generic_M.2BL'

p = []
for filename in glob.glob('*.2B'):
    filename_bare = filename.removesuffix('.2B')
    if (filename_bare + '.2BL') not in glob.glob('*.2BL'):
        path_2BL_out = path_MSCam + '\\' + filename_bare + '.2BL'
        print('Creating label for', filename_bare, end=' ')
        # Duplicate generic 2BL file and rename in association with the .2B file
        shutil.copyfile(path_generic_2BL, path_2BL_out)
        # Modify important elements so that they reflect the .2B file names
        et = etree.parse(path_2BL_out)
        root = et.getroot()
        root[0][0].text = filename_bare[0:filename_bare.rindex('_')] + '.2B'
        root[1][0][0].text = filename_bare + '.2A'
        root[2][5][0].text = filename_bare + '.2B'
        root[3][0][0].text = filename_bare + '.2B'
        et.write(path_2BL_out, encoding="utf-8", xml_declaration=True, pretty_print=True)
        print('DONE!')
print('2BL label generation complete.')