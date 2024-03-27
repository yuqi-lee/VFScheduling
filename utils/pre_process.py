import csv
import os

def process_csv(input_files, output_file):
    for input_file in input_files:
        with open(input_file, 'r', newline='') as infile:
            reader = csv.reader(infile, delimiter=';')
            next(reader)
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                for row in reader:
                    processed_row = [row[4], row[6], row[10]]
                    writer.writerow(processed_row)

if __name__ == "__main__":
    input_files = ["2013-7/500.csv", "2013-8/500.csv", "2013-9/500.csv"]
    output_file = "VM500.csv"  
    process_csv(input_files, output_file)
