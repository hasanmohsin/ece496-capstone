import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import sys


from dataset import YouCookII
from dataset import YouCookIICollate
from torch.utils.data import DataLoader
from loss import *
from accuracy import *
from transformers import get_linear_schedule_with_warmup
from model import Model



from eval_fi import eval_all_dataset

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def train(model, num_actions_train=4, num_actions_valid = 6, batch_size=4, epochs=25, lr=0.001, MAX_DETECTIONS=20, ckpt_every = 50, save_path =  "/h/sagar/ece496-capstone/weights/t3", 
train_set_type= "pseudo", valid_set_type = "fi"):    
    
    # Validation set defaults to test set for now for diagnosing.
    step_lens = [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 27]
    
    if train_set_type == "pseudo":
        print("Train set: Pseudo vids of length: {}".format(num_actions_train))
        train_datasets = [YouCookII(num_actions_train, "/h/sagar/ece496-capstone/datasets/ycii_{}".format(num_actions_train))]
    elif train_set_type == "reg":
        print("Train set: YC2 vids of regular length: {}".format(num_actions_train))
        train_datasets = [YouCookII(num_action, "/h/sagar/ece496-capstone/datasets/ycii") for num_action in num_actions_train]
    #for debug only
    elif train_set_type == "fi":
        print("Train set: Finding It eval set of length: {}".format(step_lens))
        train_datasets = [YouCookII(num_action, "/h/sagar/ece496-capstone/datasets/fi") for num_action in step_lens]

    if valid_set_type == "fi":
        print("Validation set: Finding It eval set of length: {}".format(step_lens))
        valid_datasets = [YouCookII(num_action, "/h/sagar/ece496-capstone/datasets/fi") for num_action in step_lens]
    elif valid_set_type == "reg":
        print("Validation set: YC2 vids of regular length: {}".format(num_actions_train))
        valid_datasets = [YouCookII(num_actions_valid, "/h/sagar/ece496-capstone/datasets/ycii")]

    train_size = sum([len(train_dataset) for train_dataset in train_datasets])
    valid_size = sum([len(valid_dataset) for valid_dataset in valid_datasets])
    
    print("Training Dataset Size: {}, Validation Dataset Size: {}".format(train_size, valid_size))
    print("Effective Batch Size: {} * {} = {}".format(num_actions_train, batch_size, num_actions_train * batch_size))
    print("Learning Rate: {}, Epochs: {}".format(lr, epochs))
    
    collate = YouCookIICollate(MAX_DETECTIONS=MAX_DETECTIONS)
    
    train_dataloaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True, worker_init_fn=seed_worker)
                         for train_dataset in train_datasets]
    valid_dataloaders = [DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False, worker_init_fn=seed_worker)
                         for valid_dataset in valid_datasets]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.2*epochs), epochs)

    train_loss = np.zeros(epochs)
    valid_loss = np.zeros(epochs)
    
    train_accuracy = np.zeros(epochs)
    valid_accuracy = np.zeros(epochs)
    
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0.
        datapoints = 0
        
        for train_dataloader in train_dataloaders:
            for input_data in train_dataloader:
                _, bboxes, features, actions, steps, entities, entity_count, _ = input_data
                
                # Zero out any gradients.
                optimizer.zero_grad()

                # Run inference (forward pass).
                loss_data, VG, RR = model(steps, features, bboxes, entities, entity_count)

                # Loss from alignment.
                loss_ = compute_loss_batched(loss_data)

                # Backpropagation (backward pass).
                loss_.backward()

                # Update parameters.
                optimizer.step()

                epoch_loss += loss_
                datapoints += len(steps) * len(actions[0])
                            
        # Scheduler update.
        scheduler.step()
        epoch_loss = epoch_loss / datapoints
        
        # Save loss and accuracy at each epoch and plot.
        train_loss[epoch] = float(epoch_loss)
        train_accuracy[epoch] = get_alignment_accuracy(model, train_dataloaders) 
        
        valid_loss[epoch] = get_alignment_loss(model, valid_dataloaders)
        valid_accuracy[epoch] = get_alignment_accuracy(model, valid_dataloaders)
        
        print("Epoch {} - Train Loss: {:.2f}, Validation Loss: {:.2f}, Train Accuracy: {:.2f}, Validation Accuracy: {:.2f}"
              .format(epoch + 1, train_loss[epoch], valid_loss[epoch], train_accuracy[epoch], valid_accuracy[epoch]))

        #save model weights every few epochs, or at last epoch
        if (epoch+1) % ckpt_every == 0 or (epoch +1 == epochs):
            print("--Checkpointing at epoch {}---".format(epoch+1))
            
            torch.save(model.state_dict(), save_path+"/checkpoints/weights_epoch_{}".format(epoch+1))
    
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(valid_accuracy, label='valid accuracy')
    plt.legend()
        
    plt.savefig("{}/training_curves.png".format(save_path))
        
    return train_loss, valid_loss, train_accuracy, valid_accuracy, VG, loss_data, input_data


def main(args):

     #create default dir name based on specified options
    if args.no_train:
        train_type= "Pretrain"
    else:
        train_type= "Trained"
    desc_str = "{} (lr = {}, epochs = {}, bs = {}, num_actions_valid={})".format(train_type, args.lr, args.epochs, args.bs, args.num_actions_valid)
   
    args.save_dir = args.save_dir+desc_str+"/"
    

    makedirs(args.save_dir)
    makedirs(args.save_dir + "checkpoints/")
    
    #redirect output to file
    if args.print_file:
        orig_stdout = sys.stdout
        f = open("{}/print_output.txt".format(args.save_dir), 'w')
        sys.stdout = f
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)



    model = Model(device)
    
    if args.train_set_type== "reg":         
        num_actions_train = [3, 4, 5, 7, 8, 9, 10]     
    elif args.train_set_type=="pseudo":         
        num_actions_train = args.num_actions_train     
    else:         
        num_actions_train = [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 27] 
    
    num_actions_valid = args.num_actions_valid
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr

    if args.load_model:
        assert(args.load_dir != "None")
        print("Loading Model from {}".format(args.load_dir))
        model.load_state_dict(torch.load(args.load_dir))


    if not args.no_train:

        print("\n---------Training Start.-------------\n")

        train_loss, valid_loss, train_accuracy, valid_accuracy, VG, loss_data, data = train(
            model, 
            num_actions_train=num_actions_train,
            num_actions_valid=num_actions_valid,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            ckpt_every = args.ckpt_every,
            save_path = args.save_dir,
            train_set_type = args.train_set_type,
            valid_set_type=  args.valid_set_type
        )

        print("\n-------------Training Done!----------\n")

    if not args.no_eval:
        print("\n----------------Eval Start.----------------\n")
    
        eval_all_dataset(model, path="/h/sagar/ece496-capstone/datasets/fi")
        print("\n----------------Eval Done!----------------\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
   

    parser.add_argument('--seed', type=int, default=12)
  
    parser.add_argument('--plot_every', type=int, default=1)

    #training settings
    parser.add_argument('--no_train', action="store_true")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_actions_train', type = int, default = 4)
    parser.add_argument('--num_actions_valid', type = int, default = 6)
    parser.add_argument('--epochs', type = int, default=200)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--train_set_type', type= str, default="pseudo")
    parser.add_argument('--valid_set_type', type=str, default="fi")

    #eval settings
    parser.add_argument('--no_eval', action="store_true")    
    
    parser.add_argument('--ckpt_every', type= int, default = 50)
   
    #save dir for checkpoints + results of eval
    parser.add_argument('--save_dir', type=str, default = "/h/mhasan/ece496-capstone/train/ScriptResults/")

    #for loading previous models
    parser.add_argument('--load_dir', type=str, default="None")
    parser.add_argument('--load_model', action="store_true")

    parser.add_argument('--print_file', action="store_true")

    args = parser.parse_args()
    

    main(args)

