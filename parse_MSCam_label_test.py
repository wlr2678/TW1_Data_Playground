import glob, os
from lxml import etree

# Define path to generic label file
path_generic_2BL = r'C:\Users\Darren Wu\Desktop\SpaceInfos\2023\TW1Cont\MSCam\MSCam Unused Files'
os.chdir(path_generic_2BL)

et = etree.parse('generic.2BL')
root = et.getroot()
root[0][0].text = 'generic.2B'
print(root[0][0].text)
root[1][0][0].text = 'generic_A.2A'
print(root[1][0][0].text)
root[2][5][0].text = 'generic_A.2B'
print(root[2][5][0].text)
root[3][0][0].text = 'generic_A.2B'
print(root[3][0][0].text)

et.write('generic_M.2BL', encoding="utf-8", xml_declaration=True, pretty_print=True)