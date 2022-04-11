import torch
import os
def pr(a, b, p=2):
        return round(100 * a/b, p) if a > 0 else -1
        
def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf, exit_model, num_exit, exit_idx, logger=None, fine_tuning=False, item_limit=None):
    """Train one epoch by traditional training.
    """
    item_num = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.train_device)
        labels = labels.to(conf.train_device)
        if item_limit and item_num + labels.shape[0] > item_limit:
            logger.info("Shrinking the last batch")
            images = images[:item_limit-item_num]
            labels = labels[:item_limit-item_num]
        if labels.shape[0] == 0:
            break
        
        item_num+=labels.shape[0]
        labels = labels.squeeze()

        outputs, results = model.forward(images, conf, return_vectors=True, logger= logger, training=True)
        vectors = results["vectors"]
        # print(num_exit, exit_idx, len(vectors))
        early_pred = exit_model(vectors[num_exit])
        if fine_tuning:
            # _, early_pred = torch.max(early_pred, 1)
            loss = criterion(early_pred, outputs)
        else:
            loss = criterion(early_pred, outputs) 
            # loss = criterion(early_pred, outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Exit %d, Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (exit_idx, cur_epoch, batch_idx, len(data_loader) if not item_limit else min(len(data_loader), item_limit), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        # if (batch_idx + 1) % conf.save_freq == 0:
        #     saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
        #     state = {
        #         'state_dict': model.module.state_dict(),
        #         'epoch': cur_epoch,
        #         'batch_id': batch_idx
        #     }
        #     torch.save(state, os.path.join(conf.out_dir, saved_name))
        #     logger.info('Save checkpoint %s to disk.' % saved_name)
    if not item_limit:
        saved_name = 'Exit_%d.pt' % (exit_idx) #'Exit_%d_epoch_%d.pt' % (exit_idx, cur_epoch)
        state = {'state_dict': exit_model.state_dict(), 
                'epoch': cur_epoch, 'batch_id': batch_idx}
        torch.save(state, os.path.join(f"{conf.out_dir}/exits/{conf.exit_type}", saved_name))
        logger.info('Save checkpoint %s to disk...' % saved_name)


