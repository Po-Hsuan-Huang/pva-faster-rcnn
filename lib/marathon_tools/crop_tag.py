#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:29:02 2017

@author: pohsuan

Crop the marathon tags from raw images according to bndbox defined in .xml files if 

ground truth labels are given. Or, crop the marathon tags from raw images if

detection result 'detections.pkl' from faster-rcnn is given.

Also, the position of cropped images is pickled in 'tag_pos.p' for later uses.

** Note: tags smaller than a certain sizes is discarded.
 
Parameter : 
    
    GT : Boolean 
        Switch expressing how the bounding box was created. 


"""
from fast_rcnn.config import cfg, get_output_dir
import xml.etree.ElementTree as ET
from PIL import Image
import glob
import os
import numpy as np

import cPickle as pickle

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def xml_maker(alist, img, filename):
    
    width, height =  img.size
    depth = 3
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "Wan-Jin-Shi_Marathon"
    ET.SubElement(root, "filename").text = filename+".jpg"
    
    node1 = ET.SubElement(root, "source")
    ET.SubElement(node1, "database").text = "The WJS Database"
    ET.SubElement(node1, "annotation").text = "PASCAL VOC2007"
    ET.SubElement(node1, "image").text = "Acer"
    ET.SubElement(node1, "flickrid").text = "no"
#    ET.SubElement(node1, "font_type").text = font_type
    
    node2 = ET.SubElement(root, "owner")
    ET.SubElement(node2, "flickrid").text = "Acer"
    ET.SubElement(node2, "name").text = "KR7800"

    # size of the the screen
    node3 = ET.SubElement(root, "size")
    ET.SubElement(node3, "width").text = str(height)
    ET.SubElement(node3, "height").text = str(width)
    ET.SubElement(node3, "depth").text = str(depth)
    
    ET.SubElement(root, "segmented").text = "0"
    
        
    for i, obj in enumerate(alist) :
        if i == 0:
            Xmin, Ymin, Xmax, Ymax = obj['bbox']
        else:  
            node4 = ET.SubElement(root, "object")
            ET.SubElement(node4, "name").text = obj['name']
            ET.SubElement(node4, "pose").text = "unspecified"
            ET.SubElement(node4, "truncated").text = "0"
            ET.SubElement(node4, "difficult").text = "0"
    
            # Bounding box of the target numbers 
            xmin, ymin, xmax, ymax = obj['bbox']
            xmin = int(round(xmin - Xmin))
            ymin = int(round(ymin - Ymin))
            xmax = int(round(xmax - Xmin))
            ymax = int(round(ymax - Ymin))

            ET.SubElement(node4, "annotation_id").text = "bndbox_img"
            node6 = ET.SubElement(node4, "bndbox")
            ET.SubElement(node6, "xmin").text = str(xmin)
            ET.SubElement(node6, "ymin").text = str(ymin)
            ET.SubElement(node6, "xmax").text = str(xmax)
            ET.SubElement(node6, "ymax").text = str(ymax)
#            print (width, height, obj['name'] ,str(xmin ),str(ymin ),str(xmax ),str(ymax ))

#            print (width, height, obj['name'] ,str(xmin - Xmin),str(ymin - Ymin),str(xmax - Xmin),str(ymax - Ymin))
                
    return ET.ElementTree(root)

def namer(path, i ,src, dst):
     
   filename = path.split(src)[1]
   filename = filename.split('.xml')[0] 
   if not os.path.isdir(dst):
        os.mkdir(dst)

   cachefile = os.path.join(dst, '{:s}.jpg').format(filename + str(i))
    
   return cachefile

    
def crop_tag(imdb, net, GT = False):
    '''
    crop the tags from backgrond images using the detection resulting from test.py

    path to detections.pkl : os.path.join(output_dir, detections.pkl) 

    '/home/pohsuan.huang/pva-faster-rcnn/output/Marathon_step1/marathon_2017_test'

    path to images : imdb.image_path_at(i) 
                     imdb.image_path_from_index(index)

    '''
    # store crop tag images in the same folder storing detections.pkl
    output_dir = get_output_dir(imdb, net)
    det_file = os.path.join(output_dir, 'detections.pkl')

#%% Crop images with ground truth xml files 
    if GT :
        annots = sorted(glob.glob( os.path.join(imdb._data_path, 'Annotatoins', '*.xml')))
        images = sorted(glob.glob( os.path.join(imdb._data_path, 'JPEGImages', '*.jpg')))       
        for ind , annot in enumerate(annots):# loop over images
            objs = parse_rec(annot)
            filename = annot.split(src + src_annot) [1]
            filename = filename.split('.xml')[0] 
            img = Image.open(os.path.join(src,src_img,'{:s}.jpg').format(filename))        
            img_crops = [img.crop(obj['bbox']) for obj in objs if obj['name'] == 'tag']
            tag_inds = [ idx for idx, obj in enumerate(objs) if obj['name']=='tag']  
            text_bndboxs = [ [obj for obj in objs if  all(np.sign(np.array(obj['bbox']) /
                                  - np.array(objs[i]['bbox'] )) == [1, 1, -1, -1]) ] for i in tag_inds]
            
            img_bndboxs = [ [tag]+txtls for tag, txtls in zip( [objs[k] for k in tag_inds], text_bndboxs)]            
            for i, img in enumerate(img_crops) : # loop over tags
                cachefile = namer(annot, i ,src, dst_folder)
                if np.any([ m < 100 for m in img.size ]) :
                    pass
                    # image too small 
                else :
                    img.save(cachefile)
                    tree = xml_maker(img_bndboxs[i], img, cachefile)
                    tree.write( cachefile.split('.jpg')[0] + '.xml')   
#%%                    
# crop tags from detections.                    
    else :  
                         
        with open( det_file, 'rb') as detectoins:
            if not os.path.isdir( os.path.join(imdb._data_path,'crop_pics')):
                os.mkdir(os.path.join(imdb._data_path,'crop_pics'))

            text_path = os.path.join(imdb._data_path,'crop_pics','ImageSets','Main')
            image_path = os.path.join(imdb._data_path,'crop_pics','JPEGImages')

            assert os.path.exists(text_path), 'Path does not exist: {}'.format(text_path)
            assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)

            with open( os.path.join(text_path, 'test2.txt'), 'w') as g:

                imgbox = pickle.load(detectoins)
            
            	images = [  imdb.image_path_from_index(ind) for ind in imdb._image_index]
            	tag_pos =[]
            	for i, image in enumerate(images):                    
                    k = 1 # only extract 'tag' class
                    thres = 0.4 # confince threshold
                    objs_bbx =[ bx[0:-1] for bx in imgbox[k][i] if bx[-1] > thres ]
                    objs = [ {'name': 'tag', 'bbox': bb} for bb in objs_bbx]    
                    img = Image.open(image)
                    img_crops = [img.crop(obj['bbox']) for obj in objs if obj['name'] == 'tag']
                    img_crop_pos = [ obj['bbox'] for obj in objs if obj['name'] == 'tag']
                    for j, img in enumerate(img_crops) : # loop over tags
                        if np.all([ img.size[1] > 90 ]) :# image too small 
                            tag_pos.append(img_crop_pos[j] )
                            filename = os.path.basename(image)
                            filename = os.path.splitext(filename)[0]    
                            cachefile = os.path.join( image_path, '{:s}.jpg').format(filename + str(j))
                            img.save(cachefile)
                            g.write(filename + str(j) + '\n') 
                
            with open( os.path.join(output_dir, 'tag_pos.p'), 'wb') as fp:
                pickle.dump(tag_pos, fp)
                           
                        
                    
