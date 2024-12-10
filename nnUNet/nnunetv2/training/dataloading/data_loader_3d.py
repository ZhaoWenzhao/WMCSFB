import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

###############################################################################
## 24 By Wenzhao Zhao
################################################################################
import torch
import torch.nn.functional as F
def rand_affine_grid3d_pair(N=1,upper=2.0,lower=1.0,shear_range=1.0,rot_range=1.0):
    ## random transformation parameters
    angle_vecx = (torch.rand(N,1,1)*2-1)*np.pi*rot_range #
    angle_vecz = (torch.rand(N,1,1)*2-1)*np.pi*rot_range #
    shear_vec01 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    shear_vec02 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    shear_vec12 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    shear_vec10 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    shear_vec20 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    shear_vec21 = (torch.rand(N,1,1)*2-1)*np.pi*0.5 * shear_range#
    scale00 = (torch.rand(N,1,1)*(upper-lower))+lower #
    scale11 = scale00#
    scale22 = scale00#

    ## building simple transformaton matrices
    rot3 = torch.cat((
    torch.cos(angle_vecx),-1*torch.sin(angle_vecx),torch.zeros(N,1,1),
    torch.sin(angle_vecx),torch.cos(angle_vecx),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.zeros(N,1,1),torch.ones(N,1,1)
    ), dim=2)
    rot3 = rot3.view(N,3,3)
    
    rot1 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.cos(angle_vecz),-1*torch.sin(angle_vecz),
    torch.zeros(N,1,1),torch.sin(angle_vecz),torch.cos(angle_vecz)
    ), dim=2)
    rot1 = rot1.view(N,3,3)

    s01 = torch.tan(shear_vec01)
    shear01 = torch.cat((
    torch.ones(N,1,1),s01,torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.ones(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.zeros(N,1,1),torch.ones(N,1,1),
    ),dim=2)
    shear01 = shear01.view(N,3,3)

    s02 = torch.tan(shear_vec02)
    shear02 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),s02,
    torch.zeros(N,1,1),torch.ones(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.zeros(N,1,1),torch.ones(N,1,1),
    ),dim=2)
    shear02 = shear02.view(N,3,3)

    s12 = torch.tan(shear_vec12)
    shear12 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.ones(N,1,1),s12,    
    torch.zeros(N,1,1),torch.zeros(N,1,1),torch.ones(N,1,1),
    ),dim=2)
    shear12 = shear12.view(N,3,3)
    
    s10 = torch.tan(shear_vec10)
    shear10 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),torch.zeros(N,1,1),
    s10,torch.ones(N,1,1),torch.zeros(N,1,1),    
    torch.zeros(N,1,1),torch.zeros(N,1,1),torch.ones(N,1,1),
    ),dim=2)
    shear10 = shear10.view(N,3,3)
    
    s20 = torch.tan(shear_vec20)
    shear20 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.ones(N,1,1),torch.zeros(N,1,1),    
    s20,torch.zeros(N,1,1),torch.ones(N,1,1),
    ),dim=2)
    shear20 = shear20.view(N,3,3) 
    
    s21 = torch.tan(shear_vec21)
    shear21 = torch.cat((
    torch.ones(N,1,1),torch.zeros(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.ones(N,1,1),torch.zeros(N,1,1),    
    torch.zeros(N,1,1),s21,torch.ones(N,1,1),
    ),dim=2)
    shear21 = shear21.view(N,3,3) 
    
    scale = torch.cat((
    scale00,torch.zeros(N,1,1),torch.zeros(N,1,1),
    torch.zeros(N,1,1),scale11,torch.zeros(N,1,1),
    torch.zeros(N,1,1),torch.zeros(N,1,1),scale22,
    ),dim=2)
    scale = scale.view(N,3,3)

    ## compositing the final transformation matrices
    theta0 = torch.bmm(rot1,torch.bmm(rot3,torch.bmm(scale,torch.bmm(shear20,torch.bmm(shear10,torch.bmm(shear21,torch.bmm(shear01,torch.bmm(shear12,shear02))))))))

    theta0i = torch.linalg.inv(theta0)# inverse transformation matrices

    theta = torch.zeros(N,3,4)
    theta[:,:,0:3] = torch.Tensor(theta0)#
    
    theta_inv = torch.zeros(N,3,4)
    theta_inv[:,:,0:3] = torch.Tensor(theta0i)#
    # grid is of size NxHxWx2

    return theta,theta_inv#

################################################################################


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            
            ##24#########################################
            ## By Wenzhao Zhao
            ## Additional affine augmentation for equivariance tests
            if self.full_aug:#

                theta,theta_inv = rand_affine_grid3d_pair(N=1,upper=2.0,lower=1.0,shear_range=1.0,rot_range=1.0) # full aug

                data = data[None,:,:,:,:]
                seg = seg[None,:,:,:,:]
                grid = torch.affine_grid_generator(theta, data.shape, align_corners=False)
                data = F.grid_sample(torch.tensor(data,dtype=torch.float).contiguous(), grid)
                seg = F.grid_sample(torch.tensor(seg,dtype=torch.float).contiguous(), grid)
                data = data[0,:,:,:,:].numpy()
                seg = seg[0,:,:,:,:].numpy()

            #############################################
            
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
