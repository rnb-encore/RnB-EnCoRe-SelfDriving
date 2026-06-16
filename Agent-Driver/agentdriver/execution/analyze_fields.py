#!/usr/bin/env python3
"""
Corrected script to find ALL perception-style field types in JSON data,
including fields that have content on the same line after the colon.
"""

import json
import re


def find_all_field_patterns(json_file='data.json'):
    """
    Find all unique field types that follow the pattern "FieldName: content"
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Set of unique field names found
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Set to store unique field names
    unique_fields = set()
    
    # Pattern explanation:
    # ^ - Start of line (with MULTILINE flag)
    # \s* - Optional whitespace at the beginning
    # ([^:\n*]+?) - Capture group: one or more chars that aren't ':', '\n', or '*' (non-greedy)
    # : - Literal colon
    # The pattern doesn't require end-of-line, so it catches "Field: value" patterns
    pattern = re.compile(r'^\s*([^:\n*]+?):', re.MULTILINE)
    
    # Alternative pattern that's even more permissive:
    # This one looks for any "word/phrase: " pattern
    alternative_pattern = re.compile(r'\n\s*([^:\n]+?):', re.MULTILINE)
    
    # Iterate through all items
    for item in data:
        # Check all fields, not just perception
        for key, value in item.items():
            if isinstance(value, str):
                # Find all matches using the main pattern
                matches = pattern.findall(value)
                
                # Also try the alternative pattern
                alt_matches = alternative_pattern.findall(value)
                matches.extend(alt_matches)
                
                # Add to our set (automatically handles duplicates)
                for match in matches:
                    field_name = match.strip()
                    # Filter out certain patterns
                    if field_name and not field_name.startswith('*'):
                        # Also filter out very long strings (likely not field names)
                        if len(field_name) < 100:
                            unique_fields.add(field_name)
    
    return unique_fields


def find_fields_in_specific_keys(json_file='data.json', target_keys=None):
    """
    Find field patterns in specific keys of the JSON data.
    
    Args:
        json_file: Path to the JSON file
        target_keys: List of keys to search in (default: all string values)
        
    Returns:
        Dict mapping keys to their unique field names
    """
    if target_keys is None:
        target_keys = ['perception', 'reasoning', 'chain_of_thoughts', 'ego', 'commonsense']
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store fields by key
    fields_by_key = {key: set() for key in target_keys}
    
    # More flexible pattern that catches "Field Name: anything"
    # Including multi-word field names with spaces
    pattern = re.compile(r'(?:^|\n)\s*([A-Za-z][A-Za-z0-9\s\-_]*?):\s*(.+?)(?=\n|$)', re.MULTILINE | re.DOTALL)
    
    # Iterate through all items
    for item in data:
        for key in target_keys:
            if key in item and isinstance(item[key], str):
                matches = pattern.findall(item[key])
                
                for field_name, field_content in matches:
                    cleaned_name = field_name.strip()
                    # Basic filtering
                    if (cleaned_name and 
                        not cleaned_name.startswith('*') and 
                        len(cleaned_name) < 100 and
                        not all(c == '*' for c in cleaned_name)):
                        fields_by_key[key].add(cleaned_name)
    
    return fields_by_key


def main():
    """Main function."""
    print("=" * 60)
    print("FIELD PATTERN ANALYSIS")
    print("=" * 60)
    
    # Method 1: Find all fields across entire dataset
    print("\n1. All unique field patterns found in the data:")
    print("-" * 50)
    data_path = "/path/to/Agent-Driver/data/finetune/data_samples_train.json"
    all_fields = find_all_field_patterns(data_path)
    for i, field in enumerate(sorted(all_fields), 1):
        print(f"{i:3}. {field}")
    
    print(f"\nTotal unique fields: {len(all_fields)}")
    
    # Method 2: Find fields organized by JSON key
    print("\n2. Fields organized by JSON key:")
    print("-" * 50)
    
    fields_by_key = find_fields_in_specific_keys(data_path)
    
    for key, fields in fields_by_key.items():
        if fields:
            print(f"\n📁 {key.upper()}:")
            for field in sorted(fields):
                print(f"    • {field}")
    
    # Check if "Driving Plan" was found
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("-" * 50)
    
    driving_plan_found = any("Driving Plan" in fields for fields in fields_by_key.values())
    if driving_plan_found:
        print("✅ 'Driving Plan' was successfully detected!")
    else:
        print("❌ 'Driving Plan' was not detected - may need to adjust the pattern")
    
    # Also check in all_fields
    if "Driving Plan" in all_fields:
        print("✅ 'Driving Plan' found in comprehensive search")
        

if __name__ == "__main__":
    main()
