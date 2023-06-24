import pyrealsense2 as rs



class realSense:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.cfg = self.pipeline.start(self.config)

        self.depth_sensor = self.cfg.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.profile_depth = self.cfg.get_stream(rs.stream.depth)
        self.profile_color = self.cfg.get_stream(rs.stream.color)

        self.extrinsics = self.profile_color.get_extrinsics_to(self.profile_depth)

        self.align = rs.align(rs.stream.color)

    def get_frame(self):

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        self.camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        return aligned_depth_frame, color_frame