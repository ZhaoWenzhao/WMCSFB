import scipy.special as special
import numpy as np
import torch
import os

def cartesian_to_polar_coordinates3D(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi_y = np.arctan2(y, x)
    theta_z = np.arccos(z/rho)
    theta_z[np.isnan(theta_z)]=1e-12#
    return phi_y, theta_z, rho

def Jn(n, r):
  return special.spherical_jn(n,r)#

def calculate_FB_3Dbases_shear(L1, alpha, rot_theta, rot_z, shear_xz, shear_yz, shear_xy, shear_zy, shear_yx, shear_zx, sph_bessel):
    '''
    s = 2^alpha is the scale
    alpha <= 0
    maxK is the maximum num of bases you need
    '''
    maxK = (2 * L1 + 1)**3-1#

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 2.5

    if L1 < 2:
        truncate_freq_factor = 2.5

    xx, yy, zz = np.meshgrid(range(-L, L+1), range(-L, L+1), range(-L, L+1))
    #xx = xx+shear_xy*yy#################################
    #zz = zz+shear_zy*yy#################################
    xx = xx+shear_xz*zz#################################
    yy = yy+shear_yz*zz#################################
    
    xx = xx+shear_xy*yy#################################
    zz = zz+shear_zy*yy#################################
    
    yy = yy+shear_yx*xx#################################
    zz = zz+shear_zx*xx################################# 
    
    xx = alpha*xx/(R)
    yy = alpha*yy/(R)
    zz = alpha*zz/(R)

    ugrid = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)], 1)
    # angleGrid, lengthGrid
    tgrid, zgrid, rgrid = cartesian_to_polar_coordinates3D(ugrid[:,0], ugrid[:,1], ugrid[:,2])
    
    ########################################
    tgrid = tgrid+rot_theta ################
    zgrid = zgrid+rot_z #############

    num_grid_points = ugrid.shape[0]

    maxAngFreq = 15

    B = sph_bessel[(sph_bessel[:,0] <= maxAngFreq) & (sph_bessel[:,3]<= np.pi*R*truncate_freq_factor)]
    # print("B.shape")
    # print(B.shape)

    idxB = np.argsort(B[:,2])

    mu_ns = B[idxB, 2]**2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]
    
    z=np.cos(zgrid)#

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns = np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases = 0
    Pmn_list = []

    max_ki = max(ang_freqs[:])
    for i in range(B.shape[0]):
        ki = np.int64(ang_freqs[i])
        qi = rad_freqs[i]
        rkqi = R_ns[i]
        
        r0grid = rgrid*R_ns[i]
        #print(r0grid.shape)
        F = special.spherical_jn(ki, np.array(r0grid.copy()))#special.jv(ki, r0grid)
        
        #Legendre
        for li in range(np.int64(max_ki+1-ki)):
            n = li+ki#

            Phi = 1./np.abs(special.spherical_jn(np.int32(ki)+1, R_ns[i]))*F#

            Phi[rgrid >=1] = 0

            Phi_ns[:, i] = Phi

            if ki == 0:#
                Pmn_z = special.lpmv(ki, n, z)
                
                Psi.append(Phi*Pmn_z)
                kq_Psi.append([ki,qi,rkqi,n])
                num_bases = num_bases+1
                
                Pmn_list.append(Pmn_z)##########

            else:
                Pmn_z = special.lpmv(ki, n, z)
                Psi.append(Phi*Pmn_z*np.cos(ki*tgrid)*np.sqrt(2))
                Psi.append(Phi*Pmn_z*np.sin(ki*tgrid)*np.sqrt(2))
                kq_Psi.append([ki,qi,rkqi,n])
                kq_Psi.append([ki,qi,rkqi,n])
                num_bases = num_bases+2
                
                Pmn_list.append(Pmn_z)##########
                Pmn_list.append(Pmn_z)##########
                ######################################################
                        
    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)
    #print(Psi.shape)
    num_bases = Psi.shape[1]

    Pmn_list = np.array(Pmn_list)##########

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
        
        Pmn_list = Pmn_list[:maxK]
        
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2*L+1, 2*L+1, 2*L+1).transpose(1,2,3,0)
    psi = p[1:-1, 1:-1, 1:-1, :]
    # print(psi.shape)
    psi = psi.reshape((2*L1+1)**3, num_bases)    
        
    # normalize
    # using the sum of psi_0 to normalize.
    c = np.sum(psi[:,0])
    
    psi = psi/c

    # Add zero frequency basis ################
    psi = np.concatenate((psi,1.0/(2*L1+1)/(2*L1+1)/(2*L1+1)*np.ones(((2*L1+1)**3, 1))),axis=1).reshape(((2*L1+1)**3, num_bases+1))

    return psi, c, kq_Psi,Pmn_list
#################################################################################
def tensor_fourier_bessel_affine3D(c_o,c_in,size, rot_theta_limit, rot_z_limit, 
    shear_xz_limit, shear_yz_limit, shear_xy_limit, shear_zy_limit, shear_yx_limit, shear_zx_limit, scale_up, scale_low, bessel_folder, num_funcs=None):

    print('sfb 3d applied')
    # change path based on environment
    sph_file = bessel_folder + 'spherical_bessel.npy'
    sph_bessel = np.load(sph_file)
      
    base_rot_theta = (np.random.rand(c_o,c_in)*2-1)*rot_theta_limit*np.pi
    base_rot_z = (np.random.rand(c_o,c_in)*2-1)*rot_z_limit*np.pi
    
    base_scale = np.random.rand(c_o,c_in)*(scale_up-scale_low)+scale_low
    base_scale = 2**(base_scale)########################################

    base_shear_xy = (np.random.rand(c_o,c_in)*2-1)*shear_xy_limit
    base_shear_xy = np.tan(np.pi*base_shear_xy)
    
    base_shear_zy = (np.random.rand(c_o,c_in)*2-1)*shear_zy_limit
    base_shear_zy = np.tan(np.pi*base_shear_zy)
    
    base_shear_xz = (np.random.rand(c_o,c_in)*2-1)*shear_xz_limit
    base_shear_xz = np.tan(np.pi*base_shear_xz)
    
    base_shear_yz = (np.random.rand(c_o,c_in)*2-1)*shear_yz_limit
    base_shear_yz = np.tan(np.pi*base_shear_yz)
    
    base_shear_yx = (np.random.rand(c_o,c_in)*2-1)*shear_yx_limit
    base_shear_yx = np.tan(np.pi*base_shear_yx)
    
    base_shear_zx = (np.random.rand(c_o,c_in)*2-1)*shear_zx_limit
    base_shear_zx = np.tan(np.pi*base_shear_zx)
    #print(base_rotation)    
    
    t_x = np.random.uniform(low=0, high=size, size=(c_o,c_in)).astype(int)
    t_y = np.random.uniform(low=0, high=size, size=(c_o,c_in)).astype(int)
    t_z = np.random.uniform(low=0, high=size, size=(c_o,c_in)).astype(int)
    
    max_order = size-1
    
    num_funcs = num_funcs or size ** 3

    basis_xy = []

    bxy = []
    
    for i in range(c_o):
        for j in range(c_in):

            psi, c, kq_Psi, _ = calculate_FB_3Dbases_shear(size//2, base_scale[i,j], base_rot_theta[i,j], base_rot_z[i,j],
                            base_shear_xz[i,j], base_shear_yz[i,j],base_shear_xy[i,j], base_shear_zy[i,j],base_shear_yx[i,j], base_shear_zx[i,j],sph_bessel)
            
            #print('psi.shape')
            #print(psi.shape)
            psi = psi.transpose((1,0))
            base_n_m = (psi).reshape((-1,size,size,size))
            
            ##########################################################################
            base_n_m = np.roll(base_n_m,(t_x[i,j],t_y[i,j],t_z[i,j]),axis=(1,2,3))
            ##########################################################################
            
            #print('base_n_m',base_n_m.shape)
            # print(base_n_m.shape)                
            bxy.append(base_n_m)
    #print('np.array(bxy).shape',np.array(bxy).shape)
    basis_xy.extend(bxy)

    basis = torch.Tensor(np.stack(basis_xy))#[:num_funcs]
    #print(basis.shape)
    basis = basis.reshape(c_o, c_in, num_funcs, size, size, size).permute((2,0,1,3,4,5)).contiguous()

    return basis,base_scale,base_rot_theta,base_rot_z,base_shear_xz,base_shear_yz,base_shear_xy,base_shear_zy,base_shear_yx,base_shear_zx,t_x,t_y,t_z

########################################################################################################
from typing import Union, List, Tuple
import torch.nn as nn
class Conv3d_sfb(nn.Module):
    """
    Convolution with weighted spherical Fourier Bessel filters, 2024, by Wenzhao Zhao
    """
    '''
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    '''
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],#: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',
        numbasis: int = -1,
        filter_path: str = "/home/bbb/24SFB3D/nnunet/nnUNet/weightsFB23/",
        ):#49):#  
        # TODO: refine this type             
        #         height, width, patch_x, patch_y, channel,ks=1, stride=1, c_feature=24, c_in=3):#):
        #height, width, patch_x,patch_y, channel, ks=1, stride=2, c_feature=64, c_in=3):
        super(Conv3d_sfb, self).__init__()

        if not np.isscalar(kernel_size):
            self.kernel_size = kernel_size[0]#
        else:
            self.kernel_size = kernel_size#
        self.patch_x = self.kernel_size#
        self.patch_y = self.kernel_size#
        self.patch_z = self.kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding#[x+1 for x in padding]#
        self.dilation = dilation

        if numbasis == -1:
            #if not np.isreal(kernel_size):
            if not np.isscalar(kernel_size):
                self.numbasis = (kernel_size[0]-2)**3#
            else:
                self.numbasis = (kernel_size-2)**3#
        else:
            self.numbasis = numbasis
        self.groups = groups
        '''
        if in_channels<4:
            self.groups = 1 #in_channels#1#groups
        else:
            self.groups = 4 #24 #groups//4
        '''
        if bias is False:
            self.bias = None #False#
        else:
            #self.bias = bias
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            torch.nn.init.zeros_(self.bias)
            
        self.padding_mode = padding_mode

        save_path = filter_path
        save_file = save_path+"basisFB_ic"+str(in_channels//self.groups)+"_oc"+str(self.out_channels)+"_k"+str(self.kernel_size)+"_nb"+str(self.numbasis)+".pt"
        if not os.path.exists(save_file):
            wt,base_scale,base_rot_theta,base_rot_z,base_shear_xz,base_shear_yz,base_shear_xy,base_shear_zy,base_shear_yx,base_shear_zx,t_x,t_y,t_z=self.get_str_fb_filter_tensor(self.numbasis,
                self.in_channels//self.groups,self.out_channels,self.patch_x,self.patch_y,self.patch_z, filter_path)
            
            torch.save(wt, save_file)
        
        wt = torch.load(save_file)            
            
        self.register_buffer('wt_filter', wt)
        
        self.weight = torch.nn.Parameter(torch.Tensor(self.numbasis,self.out_channels,self.in_channels//self.groups))#(c_feature*c_in,3))
        torch.nn.init.xavier_uniform_(
        self.weight,
        gain=torch.nn.init.calculate_gain("linear"))        
        
    def forward(self, x):
        assert len(x.shape) == 5, 'x must been 5 dimensions, but got ' + str(len(x.shape))
        b,c,h,w,d = x.shape
        ##b, t, h, w = x.shape
        filter = torch.einsum('nct,nctijk->nctijk', self.weight, self.wt_filter).contiguous().mean(dim=0).contiguous()

        if isinstance(self.stride, tuple) or isinstance(self.stride, list):
            stride1 = self.stride[0]
        elif isinstance(self.stride,int):
            stride1 = self.stride

        padding1 = self.kernel_size//2
        
        result = torch.nn.functional.conv3d(x, filter, 
            stride = stride1, padding = padding1, dilation = self.dilation, groups = self.groups, bias = self.bias)

        return result #

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:#(0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != 1:#(1,) * len(self.dilation):
            s += ', dilation={dilation}'
        #if self.output_padding != 0:#(0,) * len(self.output_padding):
        #    s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.numbasis != -1:
            s += ', numbasis={numbasis}'
        #if self.affine_grid is True:
        #    s += ', affine_grid=True'
        return s.format(**self.__dict__)

    def get_str_fb_filter_tensor(self, numbasis, c_in, channel, patch_x, patch_y, patch_z, bessel_folder, seeds=20):    

        c_o = channel#64
        c_in = c_in#3
        size = patch_x#9
        
        rot_theta_limit = 1.0
        rot_z_limit = 0.5

        shear_xz_limit = 0.25#0.4#0.125#0
        shear_yz_limit = 0.25#0.4#0.125#0
        shear_xy_limit = 0.25#0.4#0.125#0
        shear_zy_limit = 0.25#0.4#0.125#0
        shear_yx_limit = 0.25#0.4#0.125#0
        shear_zx_limit = 0.25#0.4#0.125#0
        scale_up = 1.0     # 2.0
        scale_low = 0.0    # 1.0

        np.random.seed(seeds)

        wt_filter,base_scale,base_rot_theta,base_rot_z,base_shear_xz,base_shear_yz,base_shear_xy,base_shear_zy,base_shear_yx,base_shear_zx,t_x,t_y,t_z = tensor_fourier_bessel_affine3D(c_o,c_in,size, 
            rot_theta_limit, rot_z_limit, shear_xz_limit, shear_yz_limit, shear_xy_limit, shear_zy_limit, shear_yx_limit, shear_zx_limit, scale_up, scale_low, bessel_folder)#, num_funcs=num_funcs)

        wt_filter = wt_filter[0:numbasis,:,:,:,:,:]######################
        
        wt_filter = torch.tensor(wt_filter,dtype=torch.float)
        base_scale = torch.tensor(base_scale)
        base_rot_theta = torch.tensor(base_rot_theta)
        base_rot_z = torch.tensor(base_rot_z)
        base_shear_xz = torch.tensor(base_shear_xz)
        base_shear_yz = torch.tensor(base_shear_yz)
        base_shear_xy = torch.tensor(base_shear_xy)
        base_shear_zy = torch.tensor(base_shear_zy)
        base_shear_yx = torch.tensor(base_shear_yx)
        base_shear_zx = torch.tensor(base_shear_zx)
        t_x = torch.tensor(t_x)
        t_y = torch.tensor(t_y)
        t_z = torch.tensor(t_z)     

        return wt_filter,base_scale,base_rot_theta,base_rot_z,base_shear_xz,base_shear_yz,base_shear_xy,base_shear_zy,base_shear_yx,base_shear_zx,t_x,t_y,t_z
      

###########################
if __name__ == '__main__':
    data = torch.rand((2, 4, 32, 32, 32))
    conv1 = Conv3d_fb(4,16,(5,5,5),1,padding=(2,2,2))
    print(data.shape)
    output = conv1(data)
    print('output.shape')
    print(output.shape)