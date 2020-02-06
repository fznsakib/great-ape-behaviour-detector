import os
import subprocess
from xml.dom.minidom import parseString
from xml.dom.ext import PrettyPrint
from StringIO import StringIO

os.chdir('../dataset/annotations/')

# get all directories in current directory
dirs = [ f.path for f in os.scandir('.') if f.is_dir() ]

for folder in dirs:
    
    os.chdir(folder)
    
    # get all files in directory
    files = [ f.path for f in os.scandir('.') if f.is_file() ]
    
    # prettify each xml
    for filename in files:
        print(filename)
        with open(filename) as xmldata:
            xml = parseString(xmldata.read())
            
            tmpStream = StringIO()
            PrettyPrint(xml, stream=tmpStream, encoding='utf-8')
            print(tmpStream.getvalue())
            
            # soup = BeautifulSoup(xml.toprettyxml())
            
            # print(soup.prettify())
            exit()
            
            # with open(filename, 'w') as filetowrite:
            #     filetowrite.write(xml_pretty_str)
            
    os.chdir("..")
    
exit()

from xml.dom.ext import PrettyPrint
from StringIO import StringIO

def toprettyxml_fixed (node, encoding='utf-8'):
    tmpStream = StringIO()
    PrettyPrint(node, stream=tmpStream, encoding=encoding)
    return tmpStream.getvalue()