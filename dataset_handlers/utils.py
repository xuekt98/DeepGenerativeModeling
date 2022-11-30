import os


def get_image_paths_from_dir(dir):
    image_list = os.listdir(dir)
    image_list.sort()
    image_paths = []
    for i in range(0, len(image_list)):
        image_paths.append(os.path.join(dir, image_list[i]))
    return image_paths
