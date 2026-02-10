import json
import os

def get_tables_info(tables_info_path: str | None = None) -> str:
    """
    Load a human-readable database schema description from `tables_info.json`.

    By default this reads `server/text_to_sql/tables_info.json` relative to this file,
    so callers don't depend on the current working directory.
    """
    if tables_info_path is None:
        tables_info_path = os.path.join(
            os.path.dirname(__file__),
            "text_to_sql",
            "tables_info.json",
        )

    with open(tables_info_path, "r", encoding="utf-8") as f:
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
