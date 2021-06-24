from dataset import YouCookII
from dataset import YouCookIICollate
from torch.utils.data import DataLoader
from loss import *
from accuracy import *
from transformers import get_linear_schedule_with_warmup
from model import Model

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from eval_fi import eval_all_dataset

torch.manual_seed(0)

random.seed(0)

np.random.seed(0)

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

def train(model, num_actions_train=[8], num_actions_valid=6, batch_size=4, epochs=25, lr=0.001, MAX_DETECTIONS=20, ckpt_every = 50, save_path =  "/h/sagar/ece496-capstone/weights/t3"):    
    train_datasets = [YouCookII(num_action, "/h/sagar/ece496-capstone/datasets/ycii") for num_action in num_actions_train]
    valid_dataset = YouCookII(num_actions_valid, "/h/sagar/ece496-capstone/datasets/ycii")
    
    train_size = sum([len(train_dataset) for train_dataset in train_datasets])
    valid_size = len(valid_dataset)
    
    print("Training Dataset Size: {}, Validation Dataset Size: {}".format(train_size, valid_size))
        
    collate = YouCookIICollate(MAX_DETECTIONS=MAX_DETECTIONS)
    
    train_dataloaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True, worker_init_fn=seed_worker)
                         for train_dataset in train_datasets]
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True, worker_init_fn=seed_worker)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.2*epochs), epochs)

    train_loss = np.zeros(epochs)
    valid_loss = np.zeros(epochs)
    
    train_accuracy = np.zeros(epochs)
    valid_accuracy = np.zeros(epochs)
    
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0.
        num_batches = 0
        
        total = 0
        correct = 0
        
        for train_dataloader in train_dataloaders:
            for data in train_dataloader:
                _, bboxes, features, steps, entities, entity_count, _, _ = data

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
                num_batches += 1
            
        # Scheduler update.
        scheduler.step()
        epoch_loss = epoch_loss / (num_batches * batch_size)
        
        # Save loss and accuracy at each epoch and plot.
        train_loss[epoch] = float(epoch_loss)
        train_accuracy[epoch] = get_alignment_accuracy(model, train_dataloader, batch_size) 
        
        valid_loss[epoch] = get_alignment_loss(model, valid_dataloader, batch_size)
        valid_accuracy[epoch] = get_alignment_accuracy(model, valid_dataloader, batch_size)

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
    
    #plt.show()
    
    plt.savefig("{}/training_curves.png".format(save_path))

    return train_loss, valid_loss, train_accuracy, valid_accuracy, VG, loss_data, data


def main(args):
    makedirs(args.save_dir)
    makedirs(args.save_dir + "checkpoints/")
    model = Model(device)
    
    num_actions_train = [3, 4, 5, 7, 8, 9, 10]
    num_actions_valid = args.num_actions_valid
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.lr

    if args.load_model:
        assert(args.load_dir != "None")
        print("Loading Model from {}".format(args.load_dir))
        model.load_state_dict(torch.load(args.load_dir))


    if not args.no_train:

        print("Training Start.")

        train_loss, valid_loss, train_accuracy, valid_accuracy, VG, loss_data, data = train(
            model, 
            num_actions_train=num_actions_train,
            num_actions_valid=num_actions_valid,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            ckpt_save_per = args.ckpt_per,
            save_path = args.save_dir
        )

        print("Training Done!")

    if not args.no_eval:
        print("Eval Start.")
    
        eval_all_dataset(model, path="/h/sagar/ece496-capstone/datasets/fi")
        print("Eval Done!")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
   

    parser.add_argument('--seed', type=int, default=12)
  
    parser.add_argument('--plot_every', type=int, default=1)

    #training settings
    parser.add_argument('--no_train', action="store_true")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_actions_valid', type = int, default = 6)
    parser.add_argument('--epochs', type = int, default=200)
    parser.add_argument('--bs', type=int, default=16)
    
    #eval settings
    parser.add_argument('--no_eval', action="store_true")    
    
    parser.add_argument('--ckpt_every', type= int, default = 50)
   
    #save dir for checkpoints + results of eval
    parser.add_argument('--save_dir', type=str, default = "/h/mhasan/ece496-capstone/train/Results/")

    #for loading previous models
    parser.add_argument('--load_dir', type=str, default="None")
    parser.add_argument('--load_model', action="store_true")

    args = parser.parse_args()
    
    #create default dir name based on specified di
    if args.no_train:
        train_type= "Pretrain"
    else:
        train_type= "Trained"
    desc_str = "{} (lr = {}, epochs = {}, bs = {}, num_actions_valid={})".format(train_type, args.lr, args.epochs, args.bs, args.num_actions_valid)
   
    args.save_dir = args.save_dir+desc_str+"/"
    
    main(args)

