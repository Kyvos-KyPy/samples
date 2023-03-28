"""
# Copyright 2023 by Kyvos Insights
# Created By: Eugene Asahara
# Created Date: February 2023
# version ='1.0'
# 
Azure Language Services Interface for Kyvos.

"""
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

import os
import pandas as pd
import datetime
import json

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori # Used to create itemsets.

from dotenv import load_dotenv
load_dotenv() # can't get .env to save as just .env, not .env.txt

class ObjectGroups():

    DEFAULT_IMAGE_PATH = os.getenv("COMPUTER_VISION_IMAGE_PATH")

    # Table metadata.
    tblmta = {
        "item_basket_fact":{"columns":["basket", "item", "confidence"]},
        "item_dimension":{"columns":["items"]},
        "basket_dimension":{"columns":["basket","create_date","description"]},
        "basket_group_fact":{"columns":["basket","group"]},
        "item_group_fact":{"columns":["item","group"]},
        "group_dimension":{"columns":["group"]}
    }

    def __init__(self):
        self._reset()

    def _reset(self):
        self.frequent_itemsets = pd.DataFrame()
        print("before")
        self.table_info = {
            "item_basket_fact":{"func":self.item_basket_fact,"data":pd.DataFrame()},
            "item_dimension":{"func":self.item_dimension,"data":pd.DataFrame()},
            "basket_dimension":{"func":self.basket_dimension,"data":pd.DataFrame()},
            "basket_group_fact":{"func":self.basket_group_fact,"data":pd.DataFrame()},
            "item_group_fact":{"func":self.item_group_fact,"data":pd.DataFrame()},
            "group_dimension":{"func":self.group_dimension,"data":pd.DataFrame()}                          
        }
        print("adter")
        self.save_file_name = f"{os.getenv('COMPUTER_VISION_SAVE_DATA_PATH')}computer_vision.json"
        self.frequent_itemsets_filename = f"{os.getenv('COMPUTER_VISION_SAVE_DATA_PATH')}computer_vision_frequent_itemsets.csv"


    def CreateGroup(self, 
        local_image_path:str=DEFAULT_IMAGE_PATH, 
        min_support:float=0.3, 
        max_len:int=5,
        get_description:bool=True,
        recognized_tags:list=None,
        min_confidence:float=0.6
    ):
        """
        USAGE
        =====
        recognized_tags - A list of objects that will be recognized. Other tags from Computer Vision will be ignored. None=Accept all tags.
        """
        computervision_client = ComputerVisionClient(os.getenv('COMPUTER_VISION_ENDPOINT'), CognitiveServicesCredentials(os.getenv('COMPUTER_VISION_KEY')))

        self.tags = {} 

        for filename in os.listdir(local_image_path):
            if filename.lower().endswith(".jpg"):
                print(f"Working on {filename}")
                file_path = os.path.join(local_image_path, filename)
                local_image = open(file_path, "rb")
                tags_result_local = computervision_client.tag_image_in_stream(local_image)
                _description = None
                if get_description:
                    _image = open(file_path,"rb")
                    _description_result = computervision_client.describe_image_in_stream(_image)
                    _description = [d.text for d in _description_result.captions] if len(_description_result.captions)>0 else None

                if len(tags_result_local.tags) == 0:
                    print("No tags detected.")
                else:
                    self.tags.update(
                        {
                            filename:                          
                            {
                                "tags":[tag.name for tag in tags_result_local.tags if not recognized_tags or (tag.name in recognized_tags and tag.confidence>=min_confidence)],
                                "tags_complete":[{"name":tag.name, "confidence":tag.confidence} for tag in tags_result_local.tags],
                                "description":_description,
                                "create_date":datetime.datetime.fromtimestamp(os.path.getctime(file_path))

                            }
                        }
                    )
        self._create_frequent_itemsets(max_len, min_support)

    def _create_frequent_itemsets(self, max_len:int, min_support:float):
        te = TransactionEncoder()

        _listlist = [v["tags"] for k, v in self.tags.items()] # TransformationEncoder.fit requires paramter:list[list].
        te_ary = te.fit(_listlist).transform(_listlist)

        _df_flattened:pd.DataFrame = pd.DataFrame(te_ary, columns = te.columns_)


        self.frequent_itemsets = apriori(_df_flattened, min_support=min_support, use_colnames=True)
        self.frequent_itemsets["group"] = self.frequent_itemsets["itemsets"].apply(lambda x: sorted([v for v in x])) # convert to list.
        # self.frequent_itemsets.drop(['itemsets'], axis=1, inplace=True) # don't need this frozenset column anymore.
        self.frequent_itemsets["len"] = self.frequent_itemsets["group"].apply(lambda x: len(x))
        self.frequent_itemsets = self.frequent_itemsets[self.frequent_itemsets["len"]<=max_len]
        return self.frequent_itemsets

    def read(self, max_len:int=5, min_support:float=0.3, save_file_name:str=None ):
        """ Read a saved computer_vision.json file.

        save_file_name - We can override the default name of the json file. This should be a fully-qualified name.
        """
        if not save_file_name:
            self.save_file_name = save_file_name

        self._reset()
        print(f"reading {self.save_file_name}")
        with open(self.save_file_name, 'r') as json_file:
            self.tags = json.load(json_file)
        self._create_frequent_itemsets(max_len, min_support)



    def save(self):
        """Save the raw json file along with the itemsets discovered through the apriori function.
        
        The json file name is specified in the SAVE_FILENAME key of the .env file.
        The frequent item sets csv file is specified in the FREQUENT_ITEM_SETS item of the .env file.
        
        """
        if self._problem_create_table:
            return

        with open(self.save_file_name, 'w') as json_file:
            json_file.write(json.dumps(self.tags, default=str, indent=2))
        self.frequent_itemsets.to_csv (self.frequent_itemsets_filename, index = False, header=True)
        for k, v in self.table_info.items():
            print(f"saving {k}")
            if not v["data"].empty:
                v["data"].to_csv(f"{os.getenv('COMPUTER_VISION_SAVE_DATA_PATH')}{k}.csv")

    @property
    def basket_dimension(self)->pd.DataFrame:
        tkey = "basket_dimension"

        if self._problem_create_table:
            return None

        if self.table_info[tkey]["data"].empty:
            self.table_info[tkey]["data"] = self._set_df_index(pd.DataFrame([ [k,v["create_date"],v["description"]] for k,v in self.tags.items()], columns=self._table_columns(tkey)), "basket")
        return self.table_info[tkey]["data"]

    def _set_df_index(self, df:pd.DataFrame, id_prefix:str)->pd.DataFrame:
        """ format an index column name.
        """
        # df.reset_index(inplace=True)
        df.index.name = f'{id_prefix}_id'
        return df

    def get_tag_confidence(self, item:str, basket:str)->float:
        if basket in self.tags:
            _tag = [x for x in self.tags[basket]["tags_complete"] if x['name'] == item]
            if _tag:
                return _tag[0]["confidence"]
        return None

    @property
    def item_basket_fact(self)->pd.DataFrame:
        """Returns a list of ALL tags and the basket it's in.
        """
        tkey = "item_basket_fact"
        if self._problem_create_table:
            return None

        _items = [
            {"basket":k, "item":t["name"], "confidence":t["confidence"]}
            for k,v in self.tags.items()
                for t in v["tags_complete"]
        ]
        self.table_info[tkey]["data"] = pd.DataFrame(_items, columns = self._table_columns(tkey))
        return self.table_info[tkey]["data"]

    @property
    def all_items(self):
        """ Return a DataFrame of all tags recognized in the images.
            This is as opposed to only the tags we selected through the recognized_tags parameter of CreateGroup.
        """
        if self._problem_create_table:
            return None

        return self.table_info["item_basket_fact"]["data"].groupby(['item']).size().reset_index(name="basket_count")

    @property
    def _problem_create_table(self)->bool:
        """ Common logic determining if we can build tables derived from the apriori itemssets.
        """
        if self.frequent_itemsets.empty:
            print("Images must be processed before performing operations.")
            return True
        return False

    def _table_columns(self, table_name:str):
        if table_name in ObjectGroups.tblmta:
            return ObjectGroups.tblmta[table_name]["columns"]
        return None

    @property
    def item_dimension(self)->pd.DataFrame:
        
        tkey = "item_dimension"
        if self._problem_create_table:
            return None

        if self.table_info[tkey]["data"].empty:
            _items = [
                v
                for _, r in self.frequent_itemsets.iterrows()
                for v in r["itemsets"]
            ]
            self.table_info[tkey]["data"] = self._set_df_index(pd.DataFrame(set(_items),columns = self._table_columns(tkey)), "item")
        return self.table_info[tkey]["data"]
    

    @property
    def basket_group_fact(self)->pd.DataFrame:

        tkey = "basket_group_fact"

        if self._problem_create_table:
            return None

        if self.table_info[tkey]["data"].empty:
            _group = []
            for i,r in self.group_dimension.iterrows():
                group = set(r["group"].split(","))
                for k,v in self.tags.items():
                    file_tags = set(v["tags"])
                    if group.issubset(file_tags):
                        _group.append([k, r["group"]])
    
            self.table_info[tkey]["data"] = self._set_df_index(pd.DataFrame(_group, columns = self._table_columns(tkey)), "basket_group")
        return self.table_info[tkey]["data"]

    @property
    def item_group_fact(self):

        tkey = "item_group_fact"

        if self._problem_create_table:
            return None

        if self.table_info[tkey]["data"].empty:
            f = [
                    [b, r["group"]] 
                    for i,r in self.group_dimension.iterrows() 
                    for b in r["group"].split(",") 
                ]
            self.table_info[tkey]["data"] = self._set_df_index(pd.DataFrame(f, columns = self._table_columns(tkey)), "item_group")
        return self.table_info[tkey]["data"]

    @property
    def group_dimension(self):

        tkey = "group_dimension"

        if self._problem_create_table:
            return None

        if self.table_info[tkey]["data"].empty:
            _items = [
                ",".join([v for v in sorted(list(r["itemsets"]))])
                for _, r in self.frequent_itemsets.iterrows()
            ]
            _items = list(set(_items)) # Distinct the items.
            self.table_info[tkey]["data"] = self._set_df_index(pd.DataFrame(_items, columns = self._table_columns(tkey)), "group")
            self.table_info[tkey]["data"]["len"] = self.table_info[tkey]["data"]["group"].apply(lambda x: len(x.split(",")))
        return self.table_info[tkey]["data"]

    def get_baskets(self, items:list, item_in:bool=True)->pd.DataFrame:
        """ gets baskets (image, video files) that are tagged with of a list of items speficied in items.

        Usage
        -----

        items - a list of items baskets include or in comprising a group.
        items_in - True means that an basket includes and of the items. False means items specifies must contain all.
        """

        if item_in:
            _df_item = self.item_group_fact[self.item_group_fact.item.isin(items)]        
            _df = pd.merge(_df_item, self.basket_group_fact, on=["group"], how='inner')
        else:
            _group = ",".join(sorted(items))
            _df = self.basket_group_fact[self.basket_group_fact["group"] == _group]        
        
        _df = pd.DataFrame(_df["basket"].unique(),columns=["basket"])
        _df = pd.merge(_df,self.basket_dimension, on=["basket"], how='left')
 
        return _df[["basket","create_date"]]
    