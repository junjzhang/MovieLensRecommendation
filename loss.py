def glocal_loss(pred_p, reg_loss, train_m, train_r):
    diff = train_m * (train_r - pred_p)
    mse = (diff**2).mean()
    loss_p = mse + reg_loss
    return loss_p
