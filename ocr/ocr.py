import os

from PIL import Image
import time
import pandas as pd 
import numpy as np 
import re 
import cv2
import boto3
from trp import Document
from paddleocr import PaddleOCR,draw_ocr



class OCR(object):

    def __init__(self, ocr_type = 'textract'):
        if ocr_type == 'textract':
            self.textract = boto3.client('textract')
        else:
            self.reader = PaddleOCR(use_angle_cls=True, lang='en')
        self.detection_dict = dict()
        
        
    def textract_ocr(self, img_path):
        with open(img_path, "rb") as document:
            response = self.textract.analyze_document(
                Document={
                    'Bytes': document.read(),
                },
                FeatureTypes=["FORMS"])
        doc = Document(response)
        results_dict = {str(a.key):str(a.value) for a in doc.pages[0].form.fields}

        return results_dict 
    
    def paddle_ocr(self, img_path):
        try:
            result = ocr.readtext(image)
            #print(pd.DataFrame([result]).T )
            cnic  =  OCR.extract_cnic_number(image,result)
            name =  OCR.extract_name(image,result)
            result_processed = []
            result_name = []
            for line in result:
                result_processed.append(line[1][0])
            for line in result:
                result_name.append(line/100)
            #result_processed.append(dates)
            dates =  OCR.extract_dates(image,result_processed)
            cnic =  OCR.extract_cnic_number(image,result_processed)
            name =  OCR.extract_name(image,result_name)
            results_dict = {'Name':name, 'Identity Number':cnic, 'dates':dates}
            return dates,cnic,name

        except Exception as e :
            return {"[ERROR] Unable to process file: {0}".format(e)}
    @staticmethod
    def normalize(img,result):
        w,h = img.shape[:-1]
        normalize_bbx = []
        detected_labels = []
        for (bbox, text, prob) in result:
            (tl, tr, br, bl) = bbox
            tl[0],tl[1] = round(tl[0] / h,3),round(tl[1] / w,3)
            tr[0],tr[1] = round(tr[0] / h,3),round(tr[1] / w,3)
            br[0],br[1] = round(br[0] / h,3),round(br[1] / w,3)
            bl[0],bl[1] = round(bl[0] / h,3),round(bl[1] / w,3)
            normalize_bbx.append([tl,tr,br,bl])
            detected_labels.append(text)
        return normalize_bbx,detected_labels
    @staticmethod
    def calculate_distance(key,bbx):
        euc_sum = 0
        for val1,val2 in zip(key,bbx):
            euc_sum = euc_sum + distance.euclidean(val1,val2)
            return euc_sum
    @staticmethod    
    def get_value(key,normalize_output):
        distances = {}
        for bbx,text in normalize_output:
            distances[text] = OCR.calculate_distance(key,bbx)
            return distances  
    @staticmethod
    def pre_result (image , result):
        image = image
        #new_result  = pd.DataFrame([result]).T
        norm_boxes,labels = OCR.normalize(image,result)

        # %%
        normalize_output = list(zip(norm_boxes,labels)) 
        #print("normalize_output" , normalize_output)

        # %%
        """
        ## Measuring Distance 
        """
        ## Defining our Static Card Template Boxes  


        # %%
        # name_key = [[0.272, 0.233], [0.323, 0.233], [0.323, 0.27], [0.272, 0.27]]
        name_value = [[0.283, 0.271], [0.415, 0.271], [0.415, 0.325], [0.283, 0.325]]
        # father_key = [[0.285, 0.42], [0.388, 0.42], [0.388, 0.457], [0.285, 0.457]]
        father_value = [[0.29, 0.456], [0.494, 0.456], [0.494, 0.514], [0.29, 0.514]]
        # dob_key = [[0.519, 0.713], [0.631, 0.713], [0.631, 0.756], [0.519, 0.756]]
        dob_value = [[0.529, 0.751], [0.648, 0.751], [0.648, 0.803], [0.529, 0.803]]
        # doi_key = [[0.274, 0.821], [0.384, 0.821], [0.384, 0.858], [0.274, 0.858]]
        doi_value = [[0.285, 0.857], [0.404, 0.857], [0.404, 0.908], [0.285, 0.908]]
        # doe_key = [[0.519, 0.821], [0.647, 0.821], [0.647, 0.866], [0.519, 0.866]]
        doe_value = [[0.531, 0.859], [0.65, 0.859], [0.65, 0.911], [0.531, 0.911]]

        # %%
        # %%
        dict_data = {}
        output_dict = {}
        output_dict['Name'] = name_value
        #output_dict['Father Name']  = father_value
        #output_dict['Date of Birth'] = dob_value
        #output_dict['Date of Issue'] = doi_value
        #output_dict['Date of Expiry'] = doe_value

        # %%
        for key,value in output_dict.items():
            output_dict = get_value(value,normalize_output)
            answer = list(min(output_dict.items(), key=lambda x: x[1]))[0]
            dict_data[key] = answer

        return (output_dict)

    @staticmethod
    def extract_dates(image,result):
        #_, _,output_dictionary = pre_result (image, result)
        result =pd.DataFrame([result]).T

        result.columns = ['index']
        temp =result[['index']]
        #temp =result.reset_index()[['index']]
        temp['index'] = temp['index'].str.replace(",","-",regex=False)
        temp['index'] = temp['index'].str.replace("/","-",regex=False)
        temp['index'] = temp['index'].str.replace("\\","-",regex=False)
        temp['index'] = temp['index'].str.replace(".","-",regex=False)

        temp['index'] = temp['index'].str.replace("+","-",regex=False)
        temp['index'] = temp['index'].str.replace(" ","-",regex=False)
        dates = []
        for stringg in temp['index']: 
        #print(stringg)
        #print()


            x = re.findall("\d{2}-\d{2}-\d{4}"  , str(stringg))
            if x :
              #output.append(x)
                temp.loc[temp['index']==stringg,'label'] ='DATE'

                dates.append(x)


        print("\nFound {} dates for this {}\n")      

        return dates if len(dates)>0 else "not_found"
    @staticmethod
    def extract_cnic_number(image , result):
        result =pd.DataFrame([result]).T

        result.columns = ['index']
        temp =result[['index']]
        temp['index'] = temp['index'].str.replace(",","-",regex=False)
        temp['index'] = temp['index'].str.replace("/","-",regex=False)
        temp['index'] = temp['index'].str.replace("\\","-",regex=False)
        temp['index'] = temp['index'].str.replace(".","-",regex=False)

        temp['index'] = temp['index'].str.replace("+","-",regex=False)
        temp['index'] = temp['index'].str.replace(" ","-",regex=False)
        cnic_number = None
        for stringg in temp['index']: 
        #print(stringg)
        #print()
            #x= x.replace("-","")  
            x = re.findall("\d{5}-\d{7}-\d{1}"  , str(stringg))
            stringg2= stringg.replace("-","")  
            x2 = re.findall("\d{13}"  , str(stringg2))

            if x or x2 :
              #output.append(x)
              #print('x',x)
                temp.loc[temp['index']==stringg,'label'] ='CNIC_NUMBER'

                cnic_number = x if x else x2
                break
              #cnic_number = x2

        return cnic_number if cnic_number else "not_found"
    @staticmethod
    def extract_name(image , result):

        output_dictionary = pre_result(image, result)
        print("output dictionary ", output_dictionary)
        x=0
        output_names = pd.DataFrame(output_dictionary.items(),columns=['text','distance']).sort_values(by='distance')  
        for output_name in output_names['text'].values[:2] :
            if ( ((output_name.lower() != 'name') & (output_name.lower() != '') & (len(output_name)>3))  and  x==0 ):
                x=1
                x = output_name 
                #temp.loc[temp['index']==stringg,'label'] ='NAME'
                found_output_name = output_name
                print ( "\nNAME is :" ,  output_name)
                name = output_name

                break
        return name  if name else "not_found"

    
    @staticmethod
    def get_name(results_dict):
        try:
            if 'Name' in results_dict.keys():
                return results_dict['Name']
        except:
            print("No name found")
            return "Please enter your name"
    
    @staticmethod
    def get_cnic(results_dict):
        try:
            if 'Identity Number' in results_dict.keys():
                return results_dict['Identity Number']
        except:
            print("No CNIC number found")
            return "Please enter your CNIC number"
    @staticmethod    
    def get_dob(results_dict):
        try:
            if 'Date of Birth' in results_dict.keys():
                return results_dict['Date of Birth']
        except:
            print("No date of birth number found")
            return "Please enter your date of birth"
        
    @staticmethod
    def get_doe(results_dict):
        try:
            if 'Date of Expiry' in results_dict.keys():
                return results_dict['Date of Expiry']
        except:
            print("No date of expiry found")
            return "Please enter your CNIC date of expiry"
    @staticmethod
    def get_date_paddle(results_dict):
        try:
            if 'dates' in results_dict.keys():
                return results_dict['dates']
        except:
            print("No dates found")
            return "Please enter dates"

    def get_results_textract(img_path):
        ocr = OCR()
        results = ocr.textract_ocr(img_path)
        name,cnic,dob,doe = OCR.get_name(results), OCR.get_cnic(results), OCR.get_dob(results), OCR.get_doe(results)
        
        return (name, cnic, dob, doe)
    
    def get_results_paddle(img_path):
        ocr = OCR('paddle')
        results = ocr.paddle_ocr(img_path)
        name,cnic,dates = OCR.get_name(results), OCR.get_cnic(results), OCR.get_dates(results)
        
        return (name, cnic, dates)
        
    
        
