import os
import shutil
import random
import yaml
class DataSplitter:
    """
    Split data into train, val and test set.
    When completed dataset folder should contain:
    
    1. train/ Folder containing images (jpg) and labels (txt), images and corresponding label has same name.
    2. val/ - || - Val set.
    3. test/ - || - Test set.
    4. data.yaml. Yaml file containing: names: - list name of classes, 
                                        path: - path to dataset folder.
                                        train: - relative path to train folder, 
                                        val: - relative path to train folder, 
                                        test: - relative path to test folder.
                                        nc: - number of classes.
    Args: 
        input_folder: Folder containing images and labels.
        output_folder: Folder to save train, val and test set.
        train: Percentage of data to use for training.
        val: Percentage of data to use for validation.
        test: Percentage of data to use for testing.
    """
    def __init__(self,input_folder:str, output_folder:str, classes:list,train:float,val:float,test:float) -> None:
        assert train+val+test == 1.0, "Train, val and test must add up to 1.0"
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.train = train
        self.val = val
        self.test = test
        self.train_folder = os.path.join(self.output_folder,"train")
        self.val_folder = os.path.join(self.output_folder,"val")
        self.test_folder = os.path.join(self.output_folder,"test")
        self.data_yaml = os.path.join(self.output_folder,"data.yaml")
        self.data_paths = []
        self.classes = classes
        self.nc = len(self.classes)
        self.create_folders()
        self.get_paths()
        self.shuffle_data()
        self.create_yaml()
        # self.split_data()
    def create_folders(self,):
        """
        Create train, val and test folders.
        """
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(self.train_folder):
            os.makedirs(self.train_folder)
        if not os.path.exists(self.val_folder):
            os.makedirs(self.val_folder)
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)
    def get_paths(self):
        """
        Get paths to images and labels. Store them in tuple.
        """
        for file in os.listdir(self.input_folder):
            print(file)
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(self.input_folder,file)
                label_path = os.path.join(self.input_folder,file.split(".")[0]+".txt")
                self.data_paths.append((img_path,label_path))
    def move_data(self,paths,folder):
        import time
        """
        Copy data to train, val or test folder.
        """
        print("Copying data to: {}".format(folder))
        for img_path, label_path in paths:
            print(f"Moving {img_path} and {label_path}")
            dest_path = os.path.join(folder,img_path.split("/")[-1].split(".")[0])
            shutil.move(img_path, dest_path+".jpg")
            shutil.move(label_path, dest_path+".txt")
    def reformat_data(self,paths):
        """
        Labels constist of cls,x,y,w,h in txt files. 
        """
        raise NotImplementedError
    def create_yaml(self):
        """
        Create yaml file containing dataset information.
        """
        data = {"names":self.classes,
                "path":self.output_folder,
                "train":os.path.join(self.output_folder,"train"),
                "val":os.path.join(self.output_folder,"val"),
                "test":os.path.join(self.output_folder,"test"),
                "nc":self.nc}
        with open(self.data_yaml, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    def shuffle_data(self):
        """
        Shuffle data.
        """
        random.shuffle(self.data_paths)
    def split_data(self):
        """
        Split data into train, val and test set.
        """
        print(f"Splitting {len(self.data_paths)} images into train, val and test set...")
        train_len = int(len(self.data_paths)*self.train)
        val_len = int(len(self.data_paths)*self.val)
        test_len = int(len(self.data_paths)*self.test)
        train_paths = self.data_paths[:train_len]
        val_paths = self.data_paths[train_len:train_len+val_len]
        test_paths = self.data_paths[train_len+val_len:]
        print("Train: {}, Val: {}, Test: {}".format(len(train_paths),len(val_paths),len(test_paths)))
        self.move_data(train_paths,self.train_folder)
        self.move_data(val_paths,self.val_folder)
        self.move_data(test_paths,self.test_folder)
        # self.create_yaml()
        print("Done splitting data!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="Path to folder containing images and labels.")
    parser.add_argument("--output_folder", type=str, help="Path to folder to save train, val and test set.")
    parser.add_argument("--train", type=float, help="Percentage of data to use for training.")
    parser.add_argument("--val", type=float, help="Percentage of data to use for validation.")
    parser.add_argument("--test", type=float, help="Percentage of data to use for testing.")
    parser.add_argument("--classes", type=list, nargs="+", help="List of classes. Make sure that the order of the classes are the same as in the labels.")
    args = parser.parse_args()
    splitter = DataSplitter(args.input_folder, args.output_folder, args.train, args.val, args.test)
    splitter.split_data()