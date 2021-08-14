import torch
from torch import nn
import numpy as np

from tools import *



def train(model, train_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update, 
          nu=0.0, eta=0.0, backward=0,steps=1, steps_back=1, gradclip=1):

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                         

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []
                            
    for epoch in range(num_epochs):
        for batch_idx, data_list in enumerate(train_loader):
            model.train()

            loss_fwd = torch.tensor(0.0)
            loss_bwd = torch.tensor(0.0)
            loss_identity = torch.tensor(0.0)
            loss_identity_y = torch.tensor(0.0)
            loss_consist = torch.tensor(0.0)

            #loss_control = criterion(y_dim, model.encoder(Xfirst.float().to(device)) )*step

            #loss_forward
            y = model.encoder(data_list[0].to(device))
            for k in range(steps):
                y = model.dynamics(y)
                loss_fwd += criterion(model.decoder(y), data_list[k+1].to(device))
                loss_identity_y += criterion(y, model.encoder(data_list[k+1].to(device)))

            #loss_backward
            y = model.encoder(data_list[-1].to(device))
            for k in range(steps_back):
                y = model.backdynamics(y)                  
                loss_bwd += criterion(model.decoder(y), data_list[::-1][k+1].to(device))
                loss_identity_y += criterion(y, model.encoder(data_list[::-1][k+1].to(device)))

            #loss_identity loss_identity_y
            for k in range(steps):
                y = model.encoder(data_list[k].to(device))
                x = model.decoder(y)
                loss_identity += criterion(x, data_list[k].to(device))

            #loss_consist
            for k in range(steps):
                y = model.encoder(data_list[k].to(device))

                x = model.decoder(model.backdynamics(model.dynamics(y)))
                loss_consist += criterion(x, data_list[k].to(device))

                x = model.decoder(model.dynamics(model.backdynamics(y)))
                loss_consist += criterion(x, data_list[k].to(device))
                

            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist * 0.5 + loss_identity_y*0.5

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % 20 == 0:

                print('********** Epoche %s **********' %(epoch+1))
                print("loss identity: ", loss_identity.item())
                print("loss backward: ", loss_bwd.item())
                print("loss consistent: ", loss_consist.item())
                print("loss forward: ", loss_fwd.item())
                print("loss identity_y: ", loss_identity_y.item())
                print("loss sum: ", loss.item())

                epoch_hist.append(epoch+1)

                if hasattr(model.dynamics, 'dynamics'):
                    w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
                    print(np.abs(w))


    return model, optimizer, [epoch_hist, loss_fwd.item(), loss_consist.item()]
