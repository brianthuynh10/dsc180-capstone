from src import clean, Trainer

def main(): 
     # -- Pull Data -- 
    print('Beginning Data Cleaning')
    train, val, test = clean() # XRayDataset objects

    # -- Train Model (using papers setup) -- 
    print('Model created & training will start now')
    # Hyperparameters to experiement with:
    batch_size = [8, 16, 32, 64, 128]
    
    for bs in batch_size:
        print(f'Training with batch size: {bs}')
        trainer = Trainer(epochs=20, lr=1e-5, batch_size=bs, train_dataset=train, val_dataset=val, test_dataset=test)
        trainer.train()
        # -- Evaluate Model --
        print(trainer.evaluate())


if __name__ == "__main__":
    main()

