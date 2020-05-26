import glob
import xml.etree.ElementTree as ET


# Get XML root of annotation
def get_root(annotations_dir, video, frame_no):
    annotation_path = f"{annotations_dir}/{video}/{video}_frame_{frame_no}.xml"
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    return root


# Get ape XML element from a frame by its ape id
def get_ape_by_id(annotations_dir, video, frame_no, ape_id):
    root = get_root(annotations_dir, video, frame_no)
    ape = None

    for ape_element in root.findall("object"):
        current_ape_id = int(ape_element.find("id").text)
        if current_ape_id == ape_id:
            ape = ape_element

    return ape


# Get activity of ape from ape XML element
def get_activity(ape):
    activity = ape.find("activity").text
    return activity


# Get species of ape from ape XML element
def get_species(ape):
    species = ape.find("name").text
    return species


# Get number of apes present across a given video.
def get_no_of_apes(annotations_dir, video):
    no_of_frames = len(glob.glob(f"{annotations_dir}/{video}/*.xml"))
    no_of_apes = 0

    for frame_no in range(1, no_of_frames + 1):
        root = get_root(annotations_dir, video, frame_no)

        for ape_element in root.findall("object"):
            ape_id = int(ape_element.find("id").text)
            if ape_id > no_of_apes:
                no_of_apes = ape_id

    return no_of_apes


# Get size (height, width) of a given video
def get_image_size(annotations_dir, video):
    root = get_root(annotations_dir, video, 1)
    size = root.find("size")
    height = int(size[0].text)
    width = int(size[1].text)
    return height, width


# Get coordinates from ape XML element
def get_ape_coordinates(ape):
    coordinates = []

    for coordinate in ape.find("bndbox"):
        coordinates.append(float(coordinate.text))

    return coordinates


# Get the ith sample from the dataset
def find_sample(samples, index):
    current_index = 0

    for key in samples.keys():
        for i, value in enumerate(samples[key]):
            if current_index == index:
                return (
                    key,
                    samples[key][i]["ape_id"],
                    samples[key][i]["start_frame"],
                    samples[key][i]["activity"],
                )
            current_index += 1
