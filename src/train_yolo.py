from yolov8.ultralytics.models.yolo.detect import DetectionTrainer
import torch
import yaml
from pathlib import Path
import argparse
import os
import shutil

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', metavar='N', type=str, nargs='+', help='list of classes to train')
    parser.add_argument('--epochs', help='epochs', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--model', help='model')
    #data paramsPp
    parser.add_argument('--train-data', help='train data folder')
    parser.add_argument('--val-data', help='val data folder if available', default='')
    parser.add_argument('--expname', help='tag for the training/validation job', default='')
    args = parser.parse_args()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = os.environ['CUDA_VISIBLE_DEVICES'] if torch.cuda.is_available() else 'cpu'

    # ensure output dir exists regardless
    Path('models/'+args.expname).mkdir(exist_ok=True, parents=True)
    Path('runs/train/'+args.expname).mkdir(exist_ok=True, parents=True)
    
    #if there is something in models/yolov8_custom_1 make a copy

    classnames = args.classes
    train_data = Path(args.train_data).resolve()
    val_data = Path(args.val_data).resolve() if args.val_data != '' else train_data
    
    config = {
            "train": str(train_data),
            "val": str(val_data),
            "nc": len(classnames),
            "names": classnames,
            "basemodel": args.model
        }
    # update with current config
    with open(f"models/{args.expname}/dataset_latest.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Load a model       
    model = f'best/best_latest.pt' if os.path.exists(f'best/best_latest.pt') else  args.model
    # Train the model
    train_arg=dict(model=model,
                   data="models/"+args.expname+"/dataset_latest.yaml",
                   batch=args.batch_size,
                   epochs=args.epochs,
                   exist_ok=True,
                   device=device,
                   project='runs/train',
                   name=args.expname
                   )
        
    trainer = DetectionTrainer(overrides=train_arg)
    trainer.train()
    if local_rank==0:
            t = torch.load('runs/train/'+args.expname+'/weights/best.pt')
            torch.save(t, 'models/'+args.expname+'/best_latest.pt')
            
    # Define the source and destination paths
    source_path = 'models/' + args.expname + '/best_latest.pt'
    dest_dir = 'best/' + args.expname
    dest_path = os.path.join(dest_dir, 'best_latest.pt')

    # Ensure the destination directory exists, create it if it doesn't
    os.makedirs(dest_dir, exist_ok=True)

    # Copy the file to the new location, overwriting it if it already exists
    shutil.copy2(source_path, dest_path)
            
if __name__ == '__main__':
    main()