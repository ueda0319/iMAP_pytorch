import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import time
from visualizer import Visualizer
#import mcubes
@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))
class IMAP(nn.Module):
    def __init__(self):
        super(IMAP, self).__init__()
        self.B = nn.Linear(3,93, bias=False).cuda()
        nn.init.normal_(self.B.weight, 0.0, 25.0)
        self.fc1 = nn.Linear(93,256).cuda()
        self.fc2 = nn.Linear(256,256).cuda()
        self.fc3 = nn.Linear(256+93,256).cuda()
        self.fc4 = nn.Linear(256,256).cuda()
        self.fc5 = nn.Linear(256,4, bias=False).cuda()
        self.fc5.weight.data[3,:]*=0.1
    def forward(self, pos):
        # Position embedding
        gamma =torch.sin(self.B(pos))
        # NeRF model
        h1 = F.relu(self.fc1(gamma))
        h2 = torch.cat([F.relu(self.fc2(h1)), gamma],dim=1)
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        out = self.fc5(h4)
        return out

class Camera():
    def __init__(self, rgb, depth, px,py,pz,rx,ry,rz,a=0.0, b=0.0,fx=525.0, fy=525.0, cx=319.5, cy=239.5):
        self.params = torch.tensor([rx,ry,rz,px,py,pz,a,b]).detach().cuda().requires_grad_(True)
        # Camera Calibrations
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = torch.from_numpy(np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
            ]).astype(np.float32)).cuda().requires_grad_(False)
        self.Ki = torch.from_numpy(np.array([
            [1.0/fx, 0.0, -cx/fx],
            [0.0, 1.0/fy, -cy/fy],
            [0.0, 0.0, 1.0],
            ]).astype(np.float32)).cuda().requires_grad_(False)
        # For convert depth from 16bit color
        self.factor = 1 / 50000.0
        # Images cache
        self.rgb = torch.from_numpy((rgb).astype(np.float32)).cuda()*(1.0/256)
        self.depth = torch.from_numpy(depth.astype(np.float32)).cuda()*self.factor
        # Lighting parameter
        self.exp_a = torch.cuda.FloatTensor(1)
        self.R = torch.cuda.FloatTensor(3,3).fill_(0)
        self.T = torch.cuda.FloatTensor(3,3).fill_(0)
        self.Li = torch.cuda.FloatTensor(64).fill_(1.0/64)
        self.size = depth.shape
        self.update_transform()
        self.optimizer = optim.Adam([self.params], lr=0.005)
    def setImages(self, rgb, depth):
        self.rgb = torch.from_numpy((rgb).astype(np.float32)).cuda()*(1.0/256)
        self.depth = torch.from_numpy(depth.astype(np.float32)).cuda()*self.factor
    # Calc Transform from camera parameters
    def update_transform(self):
        i = torch.cuda.FloatTensor(3,3).fill_(0)
        i[0,0] = 1
        i[1,1] = 1
        i[2,2] = 1
        w1 = torch.cuda.FloatTensor(3,3).fill_(0)
        w1[1, 2] = -1
        w1[2, 1] = 1
        w2 = torch.cuda.FloatTensor(3,3).fill_(0)
        w2[2, 0] = -1
        w2[0, 2] = 1
        w3 = torch.cuda.FloatTensor(3,3).fill_(0)
        w3[0, 1] = -1
        w3[1, 0] = 1

        th = torch.norm(self.params[0:3])
        thi = 1.0/(th+1e-12)
        n = thi * self.params[0:3]
        c = torch.cos(th)
        s = torch.sin(th)
        w = n[0]*w1 + n[1]*w2 + n[2]*w3
        ww = torch.matmul(w,w)
        R1 = i + s * w 
        self.R = R1 + (1.0-c)*ww
        self.T = self.params[3:6]
        self.exp_a = torch.exp(self.params[6])

    def rays_batch(self, u, v):
        batch_size = u.shape[0]
        p = torch.cuda.FloatTensor(batch_size, 3,1).fill_(1)
        p[:, 0, 0] = u
        p[:, 1, 0] = v
        ray = torch.matmul(self.R, torch.matmul(self.Ki, p))[:,:,0]
        # Normalize Ray Vector
        with torch.no_grad():
            ray_li = (1.0/torch.norm(ray, dim=1).reshape(batch_size,1).expand(batch_size,3))
        rayn = ray * ray_li
        return rayn, self.T.reshape(1,3).expand(batch_size,3)
    
# Hierarchical sampling 
def sample_pdf(bins, weights, N_samples):
    # Get pdf
    weights = weights + 1e-5
    pdf = weights * torch.reciprocal(torch.sum(weights, -1, keepdim=True))
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    samples_cat,_ = torch.sort(torch.cat([samples, bins], -1), dim=-1)
    return samples_cat


class Mapper():
    def __init__(self):
        self.model = IMAP().cuda()
        self.model_tracking = IMAP().cuda()
        self.cameras = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.render_id=0
    def updateModelForTracking(self):
        self.model_tracking.load_state_dict(copy.deepcopy(self.model.state_dict()))
    def addCamera(self, rgb_filename, depth_filename, px,py,pz,rx,ry,rz, a, b):
        rgb = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        camera = Camera(rgb, depth, px,py,pz,rx,ry,rz,a,b)
        self.cameras.append(camera)
    '''
    def render_marching_cube(self, voxel_size=64, threshold=30.0):
        with torch.no_grad():
            vs = 2.4/voxel_size
            t = np.linspace(-1.2, 1.2, voxel_size+1)
            query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
            sampling = torch.from_numpy(query_pts.reshape([-1,3])).cuda()
            print(sampling.cpu())
            out = self.model(sampling)
            sigma=out[:,3].detach().cpu().numpy().reshape(voxel_size+1,voxel_size+1,voxel_size+1)
            print(np.min(sigma), np.max(sigma))
            vertices, triangles = mcubes.marching_cubes(sigma, threshold)
            if vertices.shape[0]==0:
                return np.zeros((3,3), np.float32), np.zeros((3,3), np.float32)
            print(vertices.shape)
            color_sampling = torch.from_numpy(np.stack(vertices, -1).astype(np.float32).reshape([-1,3]))*vs-1.2

            out = self.model(color_sampling.cuda())
            colors = out[:,:3].detach().cpu().numpy()

            vt=vertices[triangles.flatten()]*vs
            cl = colors[triangles.flatten()]
            return vt,cl
    '''
        
    def volume_render(self, dists, sigmas):
        max_dist = 1.5
        batch_size = dists.shape[0]
        step = dists.shape[1]
        deltas = dists[:, 1:] - dists[:, :-1]
        o = 1-torch.exp(-sigmas[:, :-1,3]*deltas)
        wo = torch.cumprod((1-o), 1)
        w = torch.cuda.FloatTensor(batch_size, step-1).fill_(0)
        w[:, 1:] = o[:, 1:] * wo[:, :-1]
        #print(w[0,:])
        dh = w * dists[:, :-1]
        ih = w.reshape(batch_size, -1,1).expand(batch_size, -1,3) * sigmas[:, :-1, :3]
        d = torch.sum(dh,dim=1)
        i = torch.sum(ih, dim=1)
        dv = torch.sum(w*torch.square(dists[:, :-1]-d.reshape(batch_size,1).expand(batch_size, step-1)), dim=1)

        # BlackBack
        d += wo[:,-1] *max_dist
        #i += wo[:,-1].reshape(batch_size,1).expand(batch_size,3)
        # dv *= torch.reciprocal(torch.sum(w,axis=1))
        return d,i,dv

    def render_rays(self, u, v, camera, n_coarse=32, n_fine=12, model_freeze=False):
        if model_freeze:
            model=self.model_tracking
        else:
            model=self.model
        batch_size = u.shape[0]
        ray, orig= camera.rays_batch(u,v)

        with torch.no_grad():
            ds = torch.linspace(0.0001,1.2,n_coarse).cuda().reshape(1, n_coarse).expand(batch_size, n_coarse)
            rays = orig.reshape(batch_size, 1,3).expand(batch_size, n_coarse,3) + ray.reshape(batch_size, 1,3).expand(batch_size, n_coarse,3)* ds.reshape(batch_size, n_coarse,1).expand(batch_size, n_coarse,3)
            sigmas = model(rays.reshape(-1,3)).reshape(batch_size,n_coarse, 4)
        
            delta = ds[0,1]-ds[0,0]
            o = 1-torch.exp(-sigmas[:, :,3]*delta)[:,1:]
            t = 1-torch.exp(-torch.cumsum(sigmas[:, :,3]*delta, 1)[:,:-1])
            w = o*t
            ds_fine = sample_pdf(ds, w, n_fine)
        rays_fine = orig.reshape(batch_size, 1,3).expand(batch_size, n_coarse+n_fine,3) + ray.reshape(batch_size, 1,3).expand(batch_size, n_coarse+n_fine,3)* ds_fine.reshape(batch_size, n_coarse+n_fine,1).expand(batch_size, n_coarse+n_fine,3)
        
        sigmas_fine = model(rays_fine.reshape(-1,3)).reshape(batch_size,n_coarse+n_fine, 4)
        d_f,i_f,dv_f = self.volume_render(ds_fine, sigmas_fine)
        return d_f,camera.exp_a*i_f+camera.params[7],dv_f
    # render Full image
    def render(self, camera):
        with torch.no_grad():
            h = camera.size[0]
            w = camera.size[1]
            depth = torch.cuda.FloatTensor(h,w).fill_(0)
            rgb = torch.cuda.FloatTensor(h,w,3).fill_(0)
            vstep=40
            for v in range(0,h,vstep):
                vs = factor * torch.arange(v,v+vstep).reshape(vstep,1).expand(vstep,w).reshape(-1).cuda()
                us = factor * torch.arange(w).reshape(1,w).expand(vstep,w).reshape(-1).cuda()
                d,r, dv, d_f,i_f,dv_f = self.render_rays(us, vs, camera)
                depth[v:v+vstep,:] = d_f.reshape(-1,w)
                rgb[v:v+vstep,:,:] = i_f.reshape(-1,w,3)
            
            rgb_cv = torch.clamp(rgb*255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_cv = torch.clamp(depth*50000/256, 0, 255).detach().cpu().numpy().astype(np.uint8)
            return rgb_cv, depth_cv
    # render preview image
    def render_small(self, camera, label):
        with torch.no_grad():
            camera.update_transform()
            factor=5
            h = int(camera.size[0]/factor)
            w = int(camera.size[1]/factor)
            depth = torch.cuda.FloatTensor(h,w).fill_(0)
            rgb = torch.cuda.FloatTensor(h,w,3).fill_(0)
            vs = factor * torch.arange(h).reshape(h,1).expand(h,w).reshape(-1).cuda()
            us = factor * torch.arange(w).reshape(1,w).expand(h,w).reshape(-1).cuda()
            d_f,i_f,dv_f = self.render_rays(us, vs, camera)
            depth = d_f.reshape(-1,w)
            rgb = i_f.reshape(-1,w,3)
            rgb_cv = torch.clamp(rgb*255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_cv = torch.clamp(depth*50000/256, 0, 255).detach().cpu().numpy().astype(np.uint8)

            rgb_gt = torch.clamp(camera.rgb*255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_gt = torch.clamp(camera.depth*50000/256, 0, 255).detach().cpu().numpy().astype(np.uint8)
            prev_rgb = cv2.hconcat([cv2.resize(rgb_cv, (camera.size[1], camera.size[0])), rgb_gt])
            prev_depth = cv2.cvtColor(cv2.hconcat([cv2.resize(depth_cv, (camera.size[1], camera.size[0])), depth_gt]), cv2.COLOR_GRAY2RGB)
            prev = cv2.vconcat([prev_rgb, prev_depth])
            cv2.imwrite("render/{}_{:04}.png".format(label,self.render_id), prev)
            self.render_id+=1
            cv2.imshow("{}_rgb".format(label), prev)
            cv2.waitKey(1)
    # Mapping step
    def mapping(self, batch_size = 200, activeSampling=True):
        if len(self.cameras)<5:
            camera_ids = np.arange(len(self.cameras))
        else:
            camera_ids = np.random.randint(0,len(self.cameras)-2,5)
            camera_ids[3] = len(self.cameras)-1
            camera_ids[4] = len(self.cameras)-2
        
        for camera_id in camera_ids:
            self.optimizer.zero_grad()
            self.cameras[camera_id].optimizer.zero_grad()
            self.cameras[camera_id].update_transform()


            h = self.cameras[camera_id].size[0]
            w = self.cameras[camera_id].size[1]
            # ActiveSampling
            if activeSampling:
                with torch.no_grad():
                    sh = int(h/8)
                    sw = int(w/8)
                    ul=[]
                    vl=[]
                    ri = torch.cuda.IntTensor(64).fill_(0)
                    for i in range(64):
                        ni = int(batch_size*self.cameras[camera_id].Li[i])
                        if ni<1:
                            ni=1
                        ri[i] = ni
                        ul.append((torch.rand(ni)*(sw-1)).to(torch.int16).cuda() + (i%8)*sw)
                        vl.append((torch.rand(ni)*(sh-1)).to(torch.int16).cuda() + int(i/8)*sh)
                    us = torch.cat(ul)#(torch.rand(batch_size)*(w-1)).to(torch.int16).cuda()
                    vs = torch.cat(vl)#(torch.rand(batch_size)*(h-1)).to(torch.int16).cuda()
            else:
                us = (torch.rand(batch_size)*(w-1)).to(torch.int16).cuda()
                vs = (torch.rand(batch_size)*(h-1)).to(torch.int16).cuda()
            depth, rgb, depth_var = self.render_rays(us, vs, self.cameras[camera_id])
            rgb_gt = torch.cat([self.cameras[camera_id].rgb[v, u, :].unsqueeze(0) for u, v in zip(us, vs)])
            depth_gt = torch.cat([self.cameras[camera_id].depth[v, u].unsqueeze(0) for u, v in zip(us, vs)])
            depth[depth_gt==0]=0
            with torch.no_grad():
                ivar = torch.reciprocal(torch.sqrt(depth_var))
                ivar[ivar.isinf()]=1
                ivar[ivar.isnan()]=1
            lg = torch.mean(torch.abs(depth-depth_gt)*ivar)
            lp = 5*torch.mean(torch.abs(rgb-rgb_gt))

            
            loss = lg+lp
            loss.backward()#retain_graph=True)
            self.optimizer.step()
            #print(lg.detach().cpu(), lp.detach().cpu())
            if camera_id > 0:
                self.cameras[camera_id].optimizer.step()

            # Update ActiveSampling
            if activeSampling:
                with torch.no_grad():
                    e = torch.abs(depth-depth_gt) + torch.sum(torch.abs(rgb -rgb_gt), 1)
                    ris = torch.cumsum(ri, 0)
                    Li = torch.cuda.FloatTensor(64).fill_(0)
                    Li[0] = torch.mean(e[:ris[0]])
                    for i in range(1,64):
                        Li[i] = torch.mean(e[ris[i-1]:ris[i]])
                    LiS = 1.0 / torch.sum(Li)
                    self.cameras[camera_id].Li = LiS * Li
    # Tracking step
    def track(self, camera, batch_size = 200):
        self.updateModelForTracking()
        for iter in range(20):
            camera.optimizer.zero_grad()
            camera.update_transform()
            h = camera.size[0]
            w = camera.size[1]
            us = (torch.rand(batch_size)*(w-1)).to(torch.int16).cuda()
            vs = (torch.rand(batch_size)*(h-1)).to(torch.int16).cuda()
            depth, rgb, depth_var = self.render_rays(us, vs, camera, model_freeze=True)
            rgb_gt = torch.cat([camera.rgb[v, u, :].unsqueeze(0) for u, v in zip(us, vs)])
            depth_gt = torch.cat([camera.depth[v, u].unsqueeze(0) for u, v in zip(us, vs)])
            depth[depth_gt==0]=0
            with torch.no_grad():
                ivar = torch.reciprocal(torch.sqrt(depth_var))
                ivar[ivar.isinf()]=1
                ivar[ivar.isnan()]=1
            lg = torch.mean(torch.abs(depth-depth_gt)*ivar)
            lp = 5*torch.mean(torch.abs(rgb-rgb_gt))
            loss = lg+lp
            loss.backward()
            camera.optimizer.step()
            p = float(torch.sum(((torch.abs(depth - depth_gt) * torch.reciprocal(depth_gt+1e-12)) < 0.1).int()).cpu().item()) /batch_size
            if p>0.8:
                break
        print("Tracking: P=", p)
        #del camera.optimizer
        return p