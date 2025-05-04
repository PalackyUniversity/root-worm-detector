from scipy.stats import skew, kurtosis
import pandas as pd
import numpy as np
import cv2

class ExportLogic:
    @classmethod
    def export_data(cls, file_path, selections, image_data):
        export_list = [cls.build_row(d, selections) for d in image_data]
        df = pd.DataFrame(export_list)
        if file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

    @classmethod
    def build_row(cls, data, selections):
        row = {"Image File": data["path"]}
        areas = cls.get_contour_areas(data)
        if selections.get("count"):
            row["Contour Count"] = len(areas)
        if selections.get("total"):
            row["Total Contour Area"] = np.sum(areas) if areas else 0
        if areas:
            arr = np.array(areas)
            if selections.get("avg"):
                cls.add_avg_metrics(row, arr)
            if selections.get("median"):
                cls.add_median_metrics(row, arr)
            if selections.get("desc"):
                cls.add_desc_metrics(row, arr)
        else:
            cls.add_empty_metrics(row)
        return row

    @staticmethod
    def get_contour_areas(data):
        return [cv2.contourArea(cnt) for cnt in data["contours"]]

    @staticmethod
    def add_avg_metrics(row, arr):
        avg = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0
        stderr = std / np.sqrt(len(arr))
        row["Average Contour Area"] = avg
        row["Avg Std"] = std
        row["Avg StdErr"] = stderr
        row["Avg Variance"] = std ** 2

    @staticmethod
    def add_median_metrics(row, arr):
        row["Median Contour Area"] = np.median(arr)
        row["Q05"] = np.percentile(arr, 5)
        row["Q10"] = np.percentile(arr, 10)
        row["Q25"] = np.percentile(arr, 25)
        row["Q75"] = np.percentile(arr, 75)
        row["Q90"] = np.percentile(arr, 90)
        row["Q95"] = np.percentile(arr, 95)

    @staticmethod
    def add_desc_metrics(row, arr):
        row["Min Area"] = np.min(arr)
        row["Max Area"] = np.max(arr)
        row["Range Area"] = np.max(arr) - np.min(arr)
        row["Skewness"] = skew(arr)
        row["Kurtosis"] = kurtosis(arr)

    @staticmethod
    def add_empty_metrics(row):
        for key in [
            "Average Contour Area", "Avg Std", "Avg StdErr", "Avg Variance",
            "Median Contour Area", "Q05", "Q10", "Q25", "Q75", "Q90", "Q95",
            "Min Area", "Max Area", "Range Area", "Skewness", "Kurtosis"
        ]:
            row[key] = 0
