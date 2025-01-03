#collect unique values of each feature to determine coding of feautres
import os
import json
import csv
from collections import Counter

#some celebrity pages were absent of all data so replace with default details
default_details = {
    "First Name": "NA",
    "Middle Name": "NA",
    "Last Name": "NA",
    "Maiden Name": "NA",
    "Full Name at Birth": "NA",
    "Alternative Name": "NA",
    "Birthday": "NA",
    "Birthplace": "NA",
    "Height": "NA",
    "Weight": "NA",
    "Zodiac Sign": "NA",
    "Sexuality": "NA",
    "Religion": "NA",
    "Ethnicity": "NA",
    "Nationality": "NA",
    "Occupation": "NA"
}

csv_header = list(default_details.keys())

#collect data and check for empty details
def process_json_files(csv_file):
    all_data = []
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, 'info/')
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                
                #check if 'details' exists
                details = data.get('details', {})
                if not details or details == {}:
                    details = default_details  
                
                #collect the details data
                person_data = {key: details.get(key, "NA") for key in csv_header}
                all_data.append(person_data)
    
    write_to_csv(all_data, csv_file)

def write_to_csv(data, csv_file):

    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=csv_header)
        writer.writeheader() 
        
        for row in data:
            writer.writerow(row)


def write_unique_values(input_file, output_file):
    
    column_unique_values = {}
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)
        
        column_unique_values = {header: set() for header in headers}
        
        for row in reader:
            for i, value in enumerate(row):
                if value == '':  
                    value = 'NA'
                column_unique_values[headers[i]].add(value)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        max_unique_values = max(len(values) for values in column_unique_values.values())
        
        for i in range(max_unique_values):
            row = []
            for header in headers:
                values_list = list(column_unique_values[header])
                row.append(values_list[i] if i < len(values_list) else '')
            writer.writerow(row)


def main():

    input_file = 'all_details.csv'
    output_file = 'unique_values.csv'
    write_unique_values(input_file, output_file)

if __name__ == "__main__":
    main()

'''File name: all_details.csv
Number of entries: 31906
Headers of CSV
--------------
First Name,Middle Name,Last Name,Maiden Name,Full Name at Birth,Alternative Name,Birthday,Birthplace,Height,Weight,Zodiac Sign,Sexuality,Religion,Ethnicity,Nationality,Occupation
----------------
Most common value for each column
----------------
Col 'First Name' - 8057 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '3072'
        Value: 'John' - Count: '286'
        Value: 'Michael' - Count: '228'
------------------------------
Col 'Middle Name' - 3619 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '23205'
        Value: 'Marie' - Count: '177'
        Value: 'Ann' - Count: '149'
------------------------------
Col 'Last Name' - 15744 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '4431'
        Value: 'Smith' - Count: '127'
        Value: 'Williams' - Count: '104'
------------------------------
Col 'Maiden Name' - 2562 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '28772'
        Value: 'Smith' - Count: '21'
        Value: 'Jones' - Count: '21'
------------------------------
Col 'Full Name at Birth' - 19268 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '12602'
        Value: 'Daniel Smith' - Count: '3'
        Value: 'Szorcsik Viktória' - Count: '2'
------------------------------
Col 'Alternative Name' - 13650 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '18077'
        Value: 'Jimmy' - Count: '10'
        Value: 'Matt' - Count: '10'
------------------------------
Col 'Birthday' - 15440 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '8251'
        Value: '30th November, 1986' - Count: '41'
        Value: '30th November, 1985' - Count: '35'
------------------------------
Col 'Birthplace' - 9531 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '9794'
        Value: 'Los Angeles, California, USA' - Count: '436'
        Value: 'New York City, New York, USA' - Count: '325'
------------------------------
Col 'Height' - 110 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '16809'
        Value: '5' 7" (170 cm)' - Count: '1284'
        Value: '5' 10" (178 cm)' - Count: '1159'
------------------------------
Col 'Weight' - 222 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '25641'
        Value: '121lbs (55 kg)' - Count: '373'
        Value: '110lbs (50 kg)' - Count: '303'
------------------------------
Col 'Zodiac Sign' - 13 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '9462'
        Value: 'Cancer' - Count: '2008'
        Value: 'Leo' - Count: '1977'
------------------------------
Col 'Sexuality' - 6 unique value(s):
--------most common(s)--------
        Value: 'Straight' - Count: '19037'
        Value: 'NA' - Count: '11077'
        Value: 'Bisexual' - Count: '1045'
------------------------------
Col 'Religion' - 38 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '25296'
        Value: 'Christian' - Count: '1869'
        Value: 'Roman Catholic' - Count: '1566'
------------------------------
Col 'Ethnicity' - 9 unique value(s):
--------most common(s)--------
        Value: 'White' - Count: '18214'
        Value: 'NA' - Count: '8436'
        Value: 'Multiracial' - Count: '2729'
------------------------------
Col 'Nationality' - 142 unique value(s):
--------most common(s)--------
        Value: 'American' - Count: '12696'
        Value: 'NA' - Count: '6085'
        Value: 'British' - Count: '2399'
------------------------------
Col 'Occupation' - 213 unique value(s):
--------most common(s)--------
        Value: 'NA' - Count: '6018'
        Value: 'Actress' - Count: '5367'
        Value: 'Actor' - Count: '3848'
------------------------------'''