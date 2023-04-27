import requests
import json
import urllib.request

from datetime import datetime
from typing import Dict, List
from pathlib import Path
from PIL import Image

import numpy as np


class DataExtractor:

    def __init__(self, newspaper, url, offset_list=[0, 2000, 4000, 6000, 8000, 10000], num_records_by_year=10) -> None:
        self.newspaper = newspaper
        self.url = url
        self.offset_list = offset_list
        self.num_records_by_year = num_records_by_year

    def __call__(self, ):

        results = self._make_endpoint_request()
        timestamps = [val["tstamp"] for val in results]
        grouped_timestamps = self.group_by_year(timestamps)
        selected_timestamps = self._select_timestamps(grouped_timestamps)
        selected_items = self._filter_selected_timestamps(results, selected_timestamps)
        return selected_items


    def _make_endpoint_request(self, ):

        results = []
        for offset in self.offset_list:
            url = f"{self.url}&offset={offset}"
            response = requests.get(url)
            response_text = json.loads(response.text)

            try:
                if response_text["response_items"]:
                    results.append(response_text)
            except Exception as e:
                print(e)
                break

        try:
            results = [item for sublist in results for item in sublist["response_items"]]
        except Exception as e:
            print(e)
            results = []

        return results


    @staticmethod
    def group_by_year(timestamps: List[str]):
        result = {}
        for timestamp in timestamps:
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            year = dt.year
            if year in result:
                result[year].append(timestamp)
            else:
                result[year] = [timestamp]
        return result

    @staticmethod
    def flat_list(list: List):
        return [item for sublist in list for item in sublist]


    def _select_timestamps(self, data: Dict):
        timestamps_to_collect = []
        for year in data.keys():
            if len(data[year]) < self.num_records_by_year:
                timestamps_to_collect.append(data[year])
            else:
                timestamps_to_collect.append(np.random.choice(data[year], self.num_records_by_year, replace=False))

        timestamps_to_collect = self.flat_list(timestamps_to_collect)
        return timestamps_to_collect


    def _filter_selected_timestamps(self, data, timestamps):
        data_to_extract = [val for val in data if val['tstamp'] in timestamps]
        return data_to_extract


    def download_picture_from_arquivo(self, file) -> None:
        p = Path("data/raw")
        url = file["linkToScreenshot"]
        source_file = Path(p, self.newspaper)
        source_file.mkdir(parents=True, exist_ok=True)
        file_name = Path(source_file, f"{self.newspaper}-{file['tstamp']}.png")
        try:
            urllib.request.urlretrieve(url, file_name)
            #time.sleep(1)
        except Exception as e:
            print(e)
            return None


def crop_image(path, shape=(2000, 2000)):

    # Open the big image file
    big_image = Image.open(path)
    width, height = shape
    #num_rows = big_image.height // height

    # Loop through each row and column and crop the big image into small images
    for row in range(1):
        # Calculate the coordinates for cropping the small image
        x0 = 0
        y0 = row * height
        x1 = x0 + width
        y1 = y0 + height

        # Crop the small image from the big image
        small_image = big_image.crop((x0, y0, x1, y1))

        # Save the small image with a unique filename
        filename = path.replace("raw", "crop")
        Path(filename).parent.mkdir(exist_ok=True)
        small_image.save(filename)