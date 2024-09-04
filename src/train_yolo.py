# from yolov5 import train
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import argparse
import os
# from ai_pipeline_utils import get_device_ids, is_mlflow_tracked, plot_from_csv

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
    
    # ensure output dir exists regardless
    Path('models/'+args.expname).mkdir(exist_ok=True, parents=True)
    Path('runs/train/'+args.expname).mkdir(exist_ok=True, parents=True)

    # train_data =Path("data/train/images").resolve()
    # print("train_data path= ",train_data)
    # val_data =Path("data/valid/images").resolve()
    # classnames = ['Car','Different-Traffic-Sign','Red-Traffic-Light','Pedestrian','Warning-Sign','Pedestrian-Crossing','Green-Traffic-Light','Prohibition-Sign','Truck','Speed-Limit-Sign','Motorcycle']
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
    weights = f'runs/train/{args.expname}/weights/best.pt' if os.path.exists(f'runs/train/{args.expname}/weights/best.pt') else  args.model
    model = YOLO(weights)
        
    # Train the model
    model.train(data="models/{args.expname}/dataset_latest.yaml",
                          batch=2,
                          epochs=1,
                          exist_ok=True,
                          project='runs/train',
                          name=args.expname,
                          )

if __name__ == '__main__':
    main()