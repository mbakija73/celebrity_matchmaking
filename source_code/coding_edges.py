#this file converts relationship information into data that can be used for edge attributes
import os
import json
import re
from datetime import datetime

#directory containing JSON files
info_dir = 'info'

#relationship type mapping
relationship_type_map = {
    "": 0,
    "Unknown": 1,
    "Relationship": 2,
    "Encounter": 3,
    "Married": 4,
    "On-Screen": 5
}

#reference date months from January 1900
reference_date = datetime(1900, 1, 1)

#function to calculate months since January 1900
def calculate_months_since_jan_1900(date_str):
    if not date_str or date_str in ["", "NA", None]:
        return -1

    try:
        #for dates containing month ex Sep 2013
        match = re.match(r'(\d{1,2}[a-z]{2})?\s*([A-Za-z]+)\s*(\d{4})', date_str)
        if match:
            year = int(match.group(3))
            if year == 0:
                return -1
            month = datetime.strptime(match.group(2), "%b").month
            date = datetime(year, month, 1)
            delta = (date.year - reference_date.year) * 12 + (date.month - reference_date.month)
            return delta

        #for dates with year only
        year_only_match = re.match(r'^\d{4}$', date_str)
        if year_only_match:
            year = int(year_only_match.group(0))
            if year == 0: 
                return -1
            date = datetime(year, 1, 1)
            delta = (date.year - reference_date.year) * 12
            return delta

    except Exception as e:
        print(f"Error parsing date: {date_str} ({e})")
        return -1

    return -1 #default if no year



#function to calculate relationship length and convert to month legth
def process_length(length_str):
    if not length_str or length_str in ["", "-", "NA", None]:
        return -1

    if "< 1 month" in length_str:
        return 0

    #for relationship length of a year or month
    match = re.match(r'(\d+)\s*(year|month)', length_str)
    if match:
        value = int(match.group(1))
        if "year" in match.group(2):
            months = value * 12  
            return months if months <= 1200 else -1  
        return value

    return -1  

#process all edge features
def process_relationships(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            for relationship in data.get("relationships", []):
                #process relationship type
                relationship["relationship_type"] = relationship_type_map.get(
                    relationship.get("relationship_type", ""), 0
                )
                #process start and end dates
                relationship["start_months_since_1900"] = calculate_months_since_jan_1900(
                    relationship.get("start_date", "")
                )
                relationship["end_months_since_1900"] = calculate_months_since_jan_1900(
                    relationship.get("end_date", "")
                )
                #process the relationship length
                relationship["length_in_months"] = process_length(relationship.get("length", ""))
            
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)

if __name__ == "__main__":
    process_relationships(info_dir)
