#this file codes data in the celebrity files so they can be used as node features
#DO NOT run another time it will mess up files once they are already coded
import os
import json
import csv
import re

nationalities_coded_path = 'feature_codings/nationalities_coded.csv'
occupations_coded_path = 'feature_codings/occupations_coded.csv'
ethnicities_coded_path = 'feature_codings/ethnicities_coded.csv'
religions_coded_path = 'feature_codings/religions_coded.csv'
zodiac_coded_path = 'feature_codings/zodiac_coded.csv'

#directory containing JSON files (if first time would replace with info)
json_dir = 'test_subjects/'

#load csv files into dictionaries
def load_coding_file(csv_path):
    coding_dict = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            value, code = row[0], row[1]
            coding_dict[value] = code
    return coding_dict

#extract only birthday from birthyear
def extract_year(birthday):
    if birthday and birthday not in ["NA", "-"]:
        return birthday.split(',')[-1].strip()
    return birthday

#extract only country from birth year
def extract_country(birthplace):
    if birthplace and birthplace not in ["NA", "-"]:
        return birthplace.split(',')[-1].strip()
    return birthplace

#convert height to cm always
def extract_height(height):
    if height and height not in ["NA", "-"]:
        match = re.search(r'\((\d+)\s*cm\)', height)
        if match:
            return match.group(1)
    return height

#apply all transformations to celebrity features
def apply_codings(json_data, coding_dicts):
    details = json_data.get("details", {})

    #do all the coded features using coding dict
    for field, coding_dict in coding_dicts.items():
        if field in details and details[field] in coding_dict:
            details[field] = coding_dict[details[field]]
    
    #for birthday, birthplace and height transform accordingly
    if "Birthday" in details:
        details["Birthday"] = extract_year(details["Birthday"])
    if "Birthplace" in details:
        details["Birthplace"] = extract_country(details["Birthplace"])
    if "Height" in details:
        details["Height"] = extract_height(details["Height"])

    json_data["details"] = details
    return json_data

def main():
    
    nationality_codes = load_coding_file(nationalities_coded_path)
    occupation_codes = load_coding_file(occupations_coded_path)
    ethnicity_codes = load_coding_file(ethnicities_coded_path)
    religion_codes = load_coding_file(religions_coded_path)
    zodiac_codes = load_coding_file(zodiac_coded_path)

    field_to_coding_dict = {
        "Nationality": nationality_codes,
        "Occupation": occupation_codes,
        "Ethnicity": ethnicity_codes,
        "Religion": religion_codes,
        "Zodiac Sign": zodiac_codes
    }
    
    #iterate through all celebrities
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            updated_data = apply_codings(data, field_to_coding_dict)
            
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(updated_data, json_file, indent=4)

if __name__ == "__main__":
    main()


