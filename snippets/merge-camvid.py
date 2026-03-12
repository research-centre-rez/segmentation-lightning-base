import numpy as np
from pathlib import Path
import os
import shutil
from tqdm.cli import tqdm

def _parse_valid_pairs(
    filepath:os.PathLike
)-> tuple[os.PathLike|None,os.PathLike|None]:
    filepath = Path(filepath)
    with open(filepath) as fp:
        lines = fp.readlines()

    dataset_dir = filepath.parent
    all_pairs = [ [dataset_dir/p.lstrip('/') for p in l.split(maxsplit =2 )] for l in lines]

    return np.array([
        p
        for p in all_pairs
        if len(p) ==2 
    ])
        
def _detect_duplicities(path_list):
    ddict = {}
    for p in path_list:
        arr = ddict.get(p.stem,[])
        arr.append(p)
        ddict[p.stem] = arr
    return [ v for k,v in ddict.items() if len(v) ==2]

            

def merge_camvid_datasets(
    dataset_1_txt:os.PathLike,
    dataset_2_txt:os.PathLike, 
    destination_dir:os.PathLike,
    exist_ok = False,
    error_on_conflict = None
):
    destination_dir = Path(destination_dir) 
    if not exist_ok and destination_dir.exists():
        raise Exception(f"Folder {destination_dir} exists")
    
    dataset_1_txt = Path(dataset_1_txt)
    ds1_pairs_paths = _parse_valid_pairs(dataset_1_txt)
    
    dataset_2_txt = Path(dataset_2_txt)
    ds2_pairs_paths = _parse_valid_pairs(dataset_2_txt)

    ds_pairs = np.vstack([ds1_pairs_paths, ds2_pairs_paths])
    
    if error_on_conflict:
        dupl_imgs = _detect_duplicities(ds_pairs[:,0])
        dupl_annot = _detect_duplicities(ds_pairs[:,1])
        
        if dupl_imgs or dupl_annot:
            raise Exception(
                f"Conflicts in files: IMGS:{dupl_imgs} ANNOT:{dupl_annot}"
            )

    destination_dir.mkdir(exist_ok=True,parents=True)
    with open(Path(destination_dir)/'default.txt','w') as fp:
        fp.writelines([
            f"/{a.parent.name}/{a.name} {b.parent.name}/{b.name} \n" 
            for a,b in ds_pairs
        ])

    # TODO check label_colors.txt
    shutil.copy(
        dataset_1_txt.parent/'label_colors.txt',
        destination_dir/'label_colors.txt'
    )

    for img_path, annot_path in ds_pairs:
        img_dir = destination_dir/'default'
        img_dir.mkdir(exist_ok=True)
        shutil.copy(
            img_path,
            img_dir/img_path.name,
        )
        annot_dir= destination_dir/'defaultannot'
        annot_dir.mkdir(exist_ok=True)
        shutil.copy(
            annot_path,
            annot_dir/annot_path.name,
        )
        

merge_camvid_datasets(
    '/<dataset_1>/default.txt',
    '/<dataset_2>/default.txt',
    '/<dataset_merged>',
    exist_ok=True,
    error_on_conflict=True
)
