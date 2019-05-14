import os  
from xml.etree.ElementTree import parse, Element

def main():
    path = '/Users/sytu/Desktop/data_project/hand_recognition/data/voc2012/Annotations'
    newPathDir = '/Users/sytu/Desktop/data_project/hand_recognition/data/voc2012/JPEGImages/'
    files = os.listdir(path)

    for xmlFile in files:
        # if os.path.isdir(xmlFile): 
            # print(xmlFile + ' dir?')
            # pass
        xmlPath = os.path.join(path, xmlFile)  
        dom = parse(xmlPath)
        root = dom.getroot()
        xmlFileName = xmlFile[0:-4]
        xmlFileFullName = xmlFileName + '.jpg'
        newPath = newPathDir + xmlFileFullName

        root.find('path').text = newPath
        root.find('filename').text = xmlFileFullName
        root.find('folder').text = "JPEGImages"

        dom.write(xmlPath, xml_declaration=True)


if __name__=='__main__':
    main()