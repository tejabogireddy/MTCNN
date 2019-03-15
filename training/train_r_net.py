import os
from utils.data_loader import DataLoader
from training.training_combined import rnet_trainer

def train_network(combined_data_file_name):
    dataloader = DataLoader(combined_data_file_name)
    data = dataloader.load_data()
    rnet_trainer('./model_store', data, 32, 0.001)

if __name__ == '__main__':
    train_network('./prepare_data/Combined_Data/RNet/train_rnet_combined_new.txt')
