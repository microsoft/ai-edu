# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import requests
from collections import OrderedDict

subscription_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"    #should add your key here



def call_cv_api(image_data):

    # You must use the same region in your REST call as you used to get your
    # subscription keys. For example, if you got your subscription keys from
    # westus, replace "westcentralus" in the URI below with "westus".
    #
    # Free trial subscription keys are generated in the "westus" region.
    # If you use a free trial subscription key, you shouldn't need to change
    # this region.
    vision_base_url = "https://xxxxxxxxx.api.cognitive.microsoft.com/vision/v2.0/"    # should complete the url here

    analyze_url = vision_base_url + "analyze"

    # Read the image into a byte array
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Description,Tags,Categories,Color'}
    response = requests.post(
         analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    # The 'analysis' object contains various fields that describe the image. The most
    # relevant caption for the image is obtained from the 'description' property.
    analysis = response.json()

    return analysis


def analysis_to_tags(analysis):
    tags = [tag['name'] for tag in analysis['tags']] + analysis['description']['tags'] \
           + [c.lower() for c in analysis['color']['dominantColors']]

    # remove duplicates
    tags = list(OrderedDict.fromkeys(tags))

    return tags


if __name__ == '__main__':
    with open('imgs/download.jpg', 'rb') as f:
        print(analysis_to_tags(call_cv_api(f.read())))
