from utils import *
from darknet import Darknet


def process(img_bytes):
    # Set the location and name of the cfg file
    cfg_file = './cfg/yolov3.cfg'

    # Set the location and name of the pre-trained weights file
    weight_file = './weights/yolov3_tline.weights'

    # Set the location and name of the COCO object classes file
    namesfile = 'data/class.names'

    # Load the network architecture
    m = Darknet(cfg_file)

    # Load the pre-trained weights
    m.load_weights(weight_file)

    # Load the COCO object classes
    class_names = load_class_names(namesfile)

    # Load the image
    img = bytes_to_numpy(img_bytes)

    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # We resize the image to the input width and height of the first layer of the network.
    resized_image = cv2.resize(original_image, (m.width, m.height))

    # Set the IOU threshold. Default value is 0.4
    iou_thresh = 0.3

    # Set the NMS threshold. Default value is 0.6
    nms_thresh = 0.3

    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

    # Print and save the objects found and their confidence levels
    objects_count, objects_confidence = print_objects(boxes, class_names)

    if not objects_count:
        return original_image, 0

    # Plot the image with bounding boxes and corresponding object class labels
    crop_image = crop_boxes(original_image, boxes)
    rotate_image = cv2.rotate(crop_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotate_image, objects_count
