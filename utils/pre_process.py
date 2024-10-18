import csv
import os

id = 500

file1 = f"/users/YuqiLi/rnd/2013-7/"+str(id)+".csv"
file2 = f"/users/YuqiLi/rnd/2013-8/"+str(id)+".csv"
file3 = f"/users/YuqiLi/rnd/2013-9/"+str(id)+".csv"

def process_csv(input_files, output_file):
    for input_file in input_files:
        with open(input_file, 'r', newline='') as infile:
            reader = csv.reader(infile, delimiter=';')
            next(reader)
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                for row in reader:
                    processed_row = [row[0], row[10]]
                    writer.writerow(processed_row)

if __name__ == "__main__":
    input_files = [file1, file2, file3]
    output_file = "VM.csv"  
    process_csv(input_files, output_file)
