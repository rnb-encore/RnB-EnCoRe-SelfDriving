#!/usr/bin/env python3
"""
Simple script to find all unique perception field types in the JSON data.
Looks for patterns like "\n{field}:" in the perception values.
"""

import json
import re


def find_fields(json_file='data.json', key='perception'):
    """
    Find all unique field types in the perception values.
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Set of unique field names found
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Set to store unique field names
    # unique_fields = set()
    field_count = {}
    
    # Regular expression to find lines ending with ":"
    # This will match any line that has text followed by a colon
    pattern = re.compile(r'(?:^|\n)\s*([A-Za-z][A-Za-z0-9\s\-_]*?):\s*(.+?)(?=\n|$)', re.MULTILINE | re.DOTALL)
    
    # Iterate through all items
    for item in data:
        if key in item:
            field_text = item[key]
            
            # Find all matches
            matches = pattern.findall(field_text)
            
            # Add to our set (automatically handles duplicates)
            for match_name, match_content in matches:
                field_name = match_name.strip()
                if field_name:  # Only add non-empty fields
                    # unique_fields.add(field_name)
                    field_count[field_name] = field_count.get(field_name, 0) + 1
    
    return field_count


def main():
    """Main function."""
    # Find all unique perception fields
    data_path = "/path/to/Agent-Driver/data/finetune/data_samples_train.json"
    # key = 'reasoning'
    # key = 'perception'
    key = 'chain_of_thoughts'
    fields = find_fields(json_file=data_path, key=key)
    
    # Display results

    print(f"Found the following {key} field types:")
    print("-" * 50)
    
    for i, field in enumerate(sorted(fields), 1):
        # print(f"{i}. {field}")
        print(f"{i}. {field} ({fields[field]})")
    
    print("-" * 50)
    print(f"Total unique fields: {len(fields)}")


if __name__ == "__main__":
    main()
