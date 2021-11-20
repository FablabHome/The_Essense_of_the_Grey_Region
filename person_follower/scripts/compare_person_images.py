#!/usr/bin/env python3

import argparse
import sys

import cv2 as cv

from core.Detection import PersonReidentification


def main(bin_path, xml_path, initial_image, images):
    person_feat_extractor = PersonReidentification(bin_path, xml_path)
    init_image = cv.imread(initial_image)
    init_desc = person_feat_extractor.extract_descriptor(init_image)
    print(f'Result comparing to {initial_image}')
    for image_name in images:
        image = cv.imread(image_name)
        desc = person_feat_extractor.extract_descriptor(image)
        desc = desc.reshape(256, 1)
        similarity = person_feat_extractor.compare_descriptors(init_desc, desc)
        print(f'{image_name}         ->       {similarity}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bin-path', type=str, required=True,
                        help="Bin path of the person re-identification model")
    parser.add_argument('-x', '--xml-path', type=str, required=True,
                        help="XML path of the person re-identification model")
    parser.add_argument('-i', '--initial-image', type=str, required=True,
                        help="Other images inputed will be compared to this image")
    parser.add_argument('-s', '--images', type=str, nargs='+', required=True,
                        help="Images to compare to the initialize one")

    args = vars(parser.parse_args())
    try:
        main(**args)
    except Exception as e:
        print(f"Program terminated due to {e}")
        sys.exit(-1)
