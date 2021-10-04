import torch
import sys
import glob
import os
import csv
import cv2
import threading
import pyrealsense2 as rs
import numpy as np
from model import Camera, Mapper

# Thread for mapping loop
def mappingThread(mapper):
    while True:
        mapper.mapping()
        time.sleep(0.1)
        print("update map")
def main():
    mapper = Mapper()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Create Align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Wait for first frame
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())

    mapper.cameras.append(Camera(
        cv2.imread(color_img, 
                    depth_img, 
                    0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0
    )))

    fixed_camera = Camera(color_img, 
                            depth_img, 
                            0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0)
    tracking_camera = Camera(color_img, 
                            depth_img, 
                            0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0)
    # Initialize Map
    for i in range(200):
        mapper.mapping(batch_size=200, activeSampling=False)

    # For calc kinematics for camera motion
    last_pose = tracking_camera.params
    camera_vel = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0]).detach().cuda().requires_grad_(True)
    
    last_kf=0

    mapping_thread = threading.Thread(target=mappingThread, args=(mapper,))
    mapping_thread.start()
    # Start tracking
    try:
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            tracking_camera.params.data += camera_vel
            tracking_camera.setImages(color_img, 
                                    depth_img)
            pe = mapper.track(tracking_camera)
            camera_vel = 0.2 * camera_vel + 0.8*(tracking_camera.params-last_pose)
            last_pose = tracking_camera.params
            if pe < 0.65 and frame-last_kf>5:
                p = tracking_camera.params
                mapper.addCamera(color_img, 
                                depth_img,
                                last_pose[3],last_pose[4],last_pose[5],
                                last_pose[0],last_pose[1],last_pose[2],
                                last_pose[6],last_pose[7])
                print("Add keyframe")
                print(last_pose.cpu())
                last_kf=frame
            # Render from tracking camera
            mapper.render_small(tracking_camera, "view")

    finally:
        # Stop streaming
        pipeline.stop()

    mapping_thread.join()

main()