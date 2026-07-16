from ultralytics import solutions
import cv2



def load_heatmap_model(model_yolo):
    heatmap_model = solutions.Heatmap(
        show=False,  # display the output
        model=model_yolo,  # path to the YOLO26 model file
        colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
        # region=region_points,  # object counting with heatmaps, you can pass region_points
        # classes=[0, 2],  # generate heatmap for specific classes, e.g., person and car.
    )
    return heatmap_model

def load_counting_model(model_yolo,region_points=None):

    if region_points is None:
        region_points = [[605, 221], [894, 211], [1243, 389], [799, 427]]  # rectangular region

    counting_model = solutions.ObjectCounter(
        show=False,  # display the output
        region=region_points,  # pass region points
        model=model_yolo,  # model="yolo26n-obb.pt" for object counting with OBB model.
        classes=[1, 2, 3, 4, 5],
        tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
    )
    return counting_model

def load_counting_region_model(model_yolo,region_points=None):

    if region_points is None:
        region_points = {
            "region-01": [[1212, 384], [1137, 432], [764, 197], [817, 178]],
            "region-02": [[1137, 431], [991, 447], [689, 217], [753, 204]],
            "region-03": [[592, 227], [688, 221], [981, 447], [826, 447]],
            "region-04": [[368, 258], [482, 247], [622, 498], [418, 509]],
            "region-05": [[248, 260], [360, 256], [416, 509], [218, 507]],
            "region-06": [[111, 253], [233, 266], [195, 503], [8, 447]],
            "region-07": [[25, 460], [233, 526], [639, 513], [832, 509], [1045, 522], [1043, 700], [81, 706]],
        }

    counting_region_model = solutions.RegionCounter(
        show=False,  # display the frame
        region=region_points,  # pass region points
        model=model_yolo,  # model for counting in regions, e.g., yolo26s.pt
    )

    return counting_region_model


def load_speed_model(model_yolo,fps=None):

    if fps is None:
        fps = 25


    speed_model = solutions.SpeedEstimator(
        show=False,  # display the output
        model=model_yolo,  # path to the YOLO26 model file.
        fps=fps,  # adjust speed based on frame per second
        # max_speed=120,  # cap speed to a max value (km/h) to avoid outliers
        # max_hist=5,  # minimum frames object tracked before computing speed
        meter_per_pixel=0.10,  # highly depends on the camera configuration
        # classes=[0, 2],  # estimate speed of specific classes.
        # line_width=2,  # adjust the line width for bounding boxes
    )

    return speed_model