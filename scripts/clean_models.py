# filename: scripts/clean_models.py
import os
import shutil

def clean_models(base_dir='models'):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
    
    for subdir in os.listdir(base_dir):
        path = os.path.join(base_dir, subdir)
        if os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Emptied {path}")

if __name__ == "__main__":
    clean_models()