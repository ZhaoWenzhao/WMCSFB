from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_panc(panc_base_dir: str, nnunet_dataset_id: int = 600):
    task_name = "3Dirc"#"KiTS2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    #cases = subdirs(kits_base_dir, prefix='case_', join=False)
    count = 0
    for tr in range(20):#cases:
        count=count+1
        print(tr)
        print(count)
        if count<211:
            tr1="{0:0=4d}".format(tr+1)
            shutil.copy(join(panc_base_dir, tr1+'.nii.gz'), join(imagestr, f'{tr1}_0000.nii.gz'))
            shutil.copy(join(panc_base_dir, tr1+'vessel01.nii.gz'), join(labelstr, f'{tr1}.nii.gz'))
            
        

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "vessel": 1
                          },
                          #regions_class_order=(1, 3, 2),
                          num_training_cases=20,#len(cases), 
                          file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='released',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="3Dirc")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('input_folder', type=str,
    #                    help="The downloaded and extracted 3Dircadb1 dataset (must have case_XXXXX subfolders)")
    parser.add_argument('-d', required=False, type=int, default=600, help='nnU-Net Dataset ID, default: 600')
    args = parser.parse_args()
    amos_base = '/home/bbb/nnunet/data_3Dircadb1'#args.input_folder
    convert_panc(amos_base, args.d)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset

