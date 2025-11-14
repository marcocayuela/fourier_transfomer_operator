import os
import csv

class MetricLogger:
    """
    Handles logging metrics to a CSV file.
    Creates the file if it doesn't exist, appends if it does.
    """
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames

        # Check if file exists
        file_exists = os.path.isfile(filepath)
        # Open file in append mode
        self.csvfile = open(filepath, 'a', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)

        # If the file is new, write header
        if not file_exists:
            self.writer.writeheader()

    def log(self, row_dict):
        """
        Write one row of metrics to CSV.
        row_dict: dictionary with keys matching fieldnames
        """
        self.writer.writerow(row_dict)
        self.csvfile.flush()  # ensure data is written immediately

    def close(self):
        """Close the CSV file."""
        self.csvfile.close()