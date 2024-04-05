from torch.optim.lr_scheduler import OneCycleLR

from torch_lr_finder import LRFinder



# Initialize the network
model =  model.to(device)



# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Find learning rate
lr_finder = LRFinder(model, optimizer, criterion, device=device)
#lr_finder.range_test(train, end_lr=100, num_iter=100)
lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state

# Get the best loss and its corresponding learning rate
best_loss = lr_finder.best_loss
best_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(best_loss)]
print("Best Loss: %s\nBest Learning Rate: %s" % (best_loss, best_lr))

# Assuming best_lr is the suitable learning rate, we can use it as LR_MAX
LR_MAX = best_lr
LR_MIN = LR_MAX / 10

# Define the One Cycle Policy
scheduler = OneCycleLR(optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader), epochs=24, pct_start=5/24, anneal_strategy='linear', div_factor=LR_MAX/LR_MIN, final_div_factor=1.0)
