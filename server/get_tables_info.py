# get_tables_info.py
import json

def get_tables_info():
   # Load database schema from JSON file
    with open('tables_info.json', 'r') as f:
        DB_SCHEMA = json.load(f)

    table_info = ""
    for table in DB_SCHEMA["tables"]:
        table_info += f"\nTable: {table['name']}\n"
        table_info += f"Description: {table['description']}\n"
        table_info += "Columns:\n"
        for col in table['columns']:
            table_info += f"- {col['name']} ({col['type']}): {col['description']}\n"
        table_info += "\n"
    return table_info