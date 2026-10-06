from scipy.stats import skew, kurtosis
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

class ExportLogic:
    @classmethod
    def export_data(cls, file_path, selections, image_data):
        export_list = [cls.build_row(d, selections) for d in image_data]
        df = pd.DataFrame(export_list)
        females = pd.DataFrame([row for data in image_data for row in cls.female_rows(data)])
        if file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
            if selections.get("individual"):
                path = Path(file_path)
                females.to_csv(path.with_name(path.stem+"_females.csv"), index=False)
        else:
            with pd.ExcelWriter(file_path) as writer:
                df.to_excel(writer, sheet_name="Images", index=False)
                if selections.get("individual"):
                    females.to_excel(writer, sheet_name="Females", index=False)

    @staticmethod
    def female_rows(data):
        rows = []
        for record in data.get("measurements", []):
            rows.append({"Image File": data["path"], "Detection Index": record.get("det_index"),
                         "DPI": data.get("dpi"), "Status": record.get("status"),
                         "Detector Confidence": record.get("detector_score"), "Nice": record.get("nice"),
                         "Nice Probability": record.get("nice_probability"),
                         "Model Nice": record.get("model_nice", record.get("nice")),
                         "Manual Nice Override": record.get("nice_override"),
                         "Nice Threshold": data.get("nice_threshold", .3), "Area (mm²)": record.get("area_mm2"),
                         "X": record.get("x"), "Y": record.get("y"), "Pipeline": data.get("pipeline"),
                         **record.get("features", {})})
        return rows

    @classmethod
    def build_row(cls, data, selections):
        row = {"Image File": data["path"], "Pipeline": data.get("pipeline"),
               "Detector Model": data.get("model_id"), "Inference Device": data.get("provenance", {}).get("device")}
        if selections.get("nice", True) and "measurements" in data:
            records = data["measurements"]
            nice_areas = [record["area_mm2"] for record in records
                          if record.get("nice") is True and record.get("area_mm2") is not None]
            row.update({"DPI": data.get("dpi"), "Nice Count": sum(record.get("nice") is True for record in records),
                        "Nice Area Count": len(nice_areas),
                        "Classified Count": sum(record.get("nice") is not None for record in records),
                        "Model-Classified Count": sum(record.get("status") == "classified"
                                                      or record.get("nice_probability") is not None
                                                      for record in records),
                        "Unclassified Count": sum(record.get("nice") is None for record in records),
                        "Total Nice Area (mm²)": float(np.sum(nice_areas)),
                        "Mean Nice Area (mm²)": float(np.mean(nice_areas)) if nice_areas else None,
                        "Median Nice Area (mm²)": float(np.median(nice_areas)) if nice_areas else None})
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
