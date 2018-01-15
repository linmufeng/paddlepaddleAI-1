import cPickle
import sys
import json
import pandas as pd
import numpy as np
import math

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def pre_processing(id):
    json_data= load_json("/home/kesci/work/Broad/INFO/meta.json")
    database=json_data['database']
    len_image=0
    for video_name in database.keys():
        if (video_name == id):
            video_info=database[video_name]
            video_subset=video_info["subset"]
            try:
                with open("/mnt/BROAD-datasets/video/"+video_subset+"/"+str(video_name)+".pkl",'rb') as f:
                    img_fea=cPickle.load(f)
                    len_image=len(img_fea)
            except:
                print "Not found!"
                return
            
            # array init
            labelArr = np.zeros([len_image])
            segment_info=video_info['annotations']
            
            # add seg flag
            for seg_info in segment_info:
                begin = float(seg_info["segment"][0])
                begin = math.ceil(begin)
                begin = int(begin)
                
                end = float(seg_info["segment"][1])
                end = math.floor(end)
                end = int(end)
                labelArr[begin:end+1] = 1
    return labelArr

def getLabelArray(id):
    labelArray = pre_processing(id) 
    # np.set_printoptions(threshold='nan')
    # print labelArray
