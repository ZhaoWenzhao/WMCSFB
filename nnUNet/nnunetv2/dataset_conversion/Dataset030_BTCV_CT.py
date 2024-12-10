from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_amos_task1(amos_base_dir: str, nnunet_dataset_id: int = 30):
    """
    AMOS doesn't say anything about how the validation set is supposed to be used. So we just incorporate that into
    the train set. Having a 5-fold cross-validation is superior to a single train:val split
    """
    task_name = "BTCV"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    #dataset_json_source = load_json(join(amos_base_dir, 'dataset.json'))
    niis_img = subfiles(join(amos_base_dir, 'Training/img'), suffix='.nii.gz', join=False)

    training_identifiers = [i[len("img"):-7] for i in niis_img]#dataset_json_source['training']]
    tr_ctr = 0
    for tr in training_identifiers:
        #if int(tr.split("_")[-1]) <= 410: # these are the CT images
        tr_ctr += 1
        shutil.copy(join(amos_base_dir, 'Training/img', "img"+tr + '.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
        shutil.copy(join(amos_base_dir, 'Training/label', "label"+tr + '.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    niis_img_test = subfiles(join(amos_base_dir, 'Testing/img'), suffix='.nii.gz', join=False)
    test_identifiers = [i[len("img"):-7] for i in niis_img_test]
    for ts in test_identifiers:
        #if int(ts.split("_")[-1]) <= 500: # these are the CT images
        shutil.copy(join(amos_base_dir, 'Testing/img', "img"+ts + '.nii.gz'), join(imagests, f'{ts}_0000.nii.gz'))

    '''
    val_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['validation']]
    for vl in val_identifiers:
        if int(vl.split("_")[-1]) <= 409: # these are the CT images
            tr_ctr += 1
            shutil.copy(join(amos_base_dir, 'imagesVa', vl + '.nii.gz'), join(imagestr, f'{vl}_0000.nii.gz'))
            shutil.copy(join(amos_base_dir, 'labelsVa', vl + '.nii.gz'), join(labelstr, f'{vl}.nii.gz'))
                            labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
    '''
    generate_dataset_json(out_base, {0: "CT"}, 
                          labels={
                            "background": 0,
                            "spleen":1,
                            "right kidney":2,
                            "left kidney":3,
                            "gallbladder":4,
                            "esophagus":5,
                            "liver":6,
                            "stomach":7,
                            "aorta":8,
                            "inferior vena cava":9,
                            "portal vein and splenic vein":10,
                            "pancreas":11,
                            "right adrenal gland":12,
                            "left adrenal gland":13},#{v: int(k) for k,v in dataset_json_source['labels'].items()},
                            num_training_cases=tr_ctr, file_ending='.nii.gz',
                            dataset_name=task_name, reference='https://www.synapse.org/#!Synapse:syn3193805/wiki/217789',
                            release='https://www.synapse.org/#!Synapse:syn3193805/wiki/217789',
                            overwrite_image_reader_writer='NibabelIOWithReorient',
                            description="Under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. "
                            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted BTCV (https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) data. ")
    parser.add_argument('-d', required=False, type=int, default=30, help='nnU-Net Dataset ID, default: 30')
    args = parser.parse_args()
    amos_base = '/home/bbb/nnunet/BTCVRawData'#args.input_folder
    convert_amos_task1(amos_base, args.d)


