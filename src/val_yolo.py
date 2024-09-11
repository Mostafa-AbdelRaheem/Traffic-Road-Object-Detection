from yolov8.ultralytics.models.yolo.detect import DetectionValidator
import torch
import yaml
from pathlib import Path
import argparse
import os
import shutil

def copy_files(expname):
    source_dir = os.path.abspath('runs/train/'+expname)
    dest_dir = os.path.abspath('best/'+expname)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy files
    for filename in os.listdir(source_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(dest_dir, filename)

            # Copy and replace
            shutil.copy2(src_file, dst_file)
            print(f'Copied {filename} to {dest_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', metavar='N', type=str, nargs='+', help='list of classes to train')
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--conf_thres', type=float)
    parser.add_argument('--iou_thres', type=float)
    parser.add_argument('--model', default = "")
    parser.add_argument('--validate_on', metavar='N', type=str, nargs='+', default=["test"], help='val on the ff. dataset-splits: ["test" and/or "val" and/or "train"] (in case of data-checks) - result from the last in list is published to tracking-url')
    parser.add_argument('--expname', help='tag for the training/validation job', default='')

    args = parser.parse_args()
    
    model_dataset_config = f'models/{args.expname}/dataset_latest.yaml'
    Path(f'validations/{args.expname}').mkdir(exist_ok=True, parents=True)
    model_config_file = f'models/{args.expname}/dataset_latest.yaml'
    
    # check that the loaded model will be validated with the currently set valiidation data
    model_data_opts = yaml.safe_load(open( Path(model_config_file).resolve() ))
    datafolders = {"train": "train", "val": "valid", "test": "test"}
    with open(model_config_file, "w") as f:
        for task in args.validate_on:
            model_data_opts[task] = str(Path(f'data/{datafolders[task]}/images/').resolve())
        yaml.dump(model_data_opts, f)    
    
    model =  f"best/{args.expname}/best_latest.pt" if os.path.exists(f"best/{args.expname}/best_latest.pt") else  args.model
    print("model",model)
    for task in args.validate_on:  

        val_args = dict(model=model,
                        conf=args.conf_thres,
                        iou=args.iou_thres,
                        batch=args.batch_size,
                        data=model_dataset_config,
                        project=f'runs/val/{args.expname}',
                        name=task,
                        exist_ok=True
                    )
        validator = DetectionValidator(args=val_args)
        validator()
        
    # Define source and destination directories

    copy_files(args.expname)

if __name__ == '__main__':
    main()