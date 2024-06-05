import sqlite3
import os
from quer import output_text
def query_database(db_path, query):
    """
    Connects to the SQLite database, executes the given query, and fetches the results.

    Parameters:
    db_path (str): The absolute path to the database file.
    query (str): The SQL query to execute.

    Returns:
    list: A list of tuples containing the query results.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch all the results
        results = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        return results
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Absolute path to the SQLite database file
    db_path = r'ims\db.sqlite3'
    
    # SQL query to execute
    query = output_text[0]  # Replace 'your_table' with the actual table name
    
    results = query_database(db_path, query)
    for row in results:
        print(row)
