import os
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_data(path):
    temp_list = []
    tree = ET.parse(path)
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                    temp = {}
                    temp['label'] = name
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                            temp['x1'] = xmin
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                            temp['y1'] = ymin
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                            temp['x2'] = xmax
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                            temp['y2'] = ymax
            temp['img'] = path.split('/')[-1].split('.')[0]
            temp_list.append(temp)
    return temp_list


path = "BCCD/Annotations"
df = pd.DataFrame([])
for i in os.listdir(path):
    df = pd.concat([df, pd.DataFrame(xml_to_data(f"{path}/{i}"))])

df.to_csv('train.csv', index=False)
