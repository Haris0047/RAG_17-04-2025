import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH"))
import json
from datetime import datetime

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


class SQLTool:
    def __init__(self):
        self.db_conventional_string = os.getenv("DATABASE_URL")
        self.conn = self.make_connection(self.db_conventional_string)

    def make_connection(self, connection_string):
        try:
            return psycopg2.connect(connection_string)
        except Exception as e:
            print(e)

    def fetch_sec_files(self, symbol, start_date, end_date, type):
        try:
            # q = f"""SELECT *
            #     FROM sec_filling_record
            #     WHERE filling_date BETWEEN '{start_date}' AND '{end_date}' and symbol = '{symbol}' and type ilike '{type}';"""
            type_array = []
            for fil_type in type:
                type_str = f"{fil_type}"
                type_array.append(type_str)

            q = f"""SELECT *
                FROM sec_filling_record
                WHERE filling_date BETWEEN '{start_date}' AND '{end_date}' and symbol = '{symbol}'

                AND type ILIKE ANY(ARRAY{type_array});"""

            print(q)
            insert_query = sql.SQL(q)
            cursor = self.conn.cursor()
            cursor.execute(insert_query)
            file_records = [
                {
                    "id": 1,
                    "symbol": "AAPL",
                    "type": "10-Q",
                    "filling_date": "2024-08-02",
                    "final_link": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm",
                    "filling_name": None,
                    "created_at": "",
                }
            ]
            columns = [
                "id",
                "symbol",
                "type",
                "filling_date",
                "final_link",
                "filling_name",
                "created_at",
                "pdf_filling_url",
                "html_filling_url",
            ]
            rows = cursor.fetchall()
            records = [dict(zip(columns, row)) for row in rows]
            for rec in records:
                rec["filling_date"] = rec["filling_date"].isoformat()

        except Exception as e:
            print(e)
            self.conn.close()
            records = None
        return records

    def insert(self, table_name, df=None, columns=None, values=None, return_id=False):
        try:
            cursor = self.conn.cursor()
            if df:
                values = [tuple(x) for x in df.to_numpy()]
                columns = list(df.columns)
            if columns and values:
                query = "INSERT INTO {} ({}) VALUES %s".format(
                    table_name, ",".join(columns)
                )
                execute_values(cursor, query, values)
                self.conn.commit()
                return None
            else:
                print(e)
        except Exception as e:
            # error_logger.log(e)
            print(e)
            self.conn.rollback()
            raise Exception(e)

    def update_filling_name(self, filling_name, html_filling_url, id):
        try:
            q = """
                    UPDATE sec_filling_record
                    SET filling_name = %s, html_filling_url = %s
                    WHERE id = %s;
                """
            fetch_query = sql.SQL(q)
            data = (filling_name, html_filling_url, id)
            cursor = self.conn.cursor()
            cursor.execute(fetch_query, data)
            self.conn.commit()
        except Exception as e:
            print(e)

    def read_data_testing(self, user_id, symbol):
        try:
            q = """select read_context_v1_testing(%(symbol)s,%(user_id)s,5)"""

            insert_query = sql.SQL(q)
            cursor = self.conn.cursor()
            cursor.execute(insert_query, {"user_id": user_id, "symbol": symbol})
            rows = cursor.fetchall()
            rows = rows[0]
        except Exception as e:
            print(e)
            self.conn.close()
            rows = None
        return rows

    def update_read_data_testing(self, user_id, symbol):

        cursor = self.conn.cursor()

        try:
            cursor.callproc(
                "public.updated_read_context_v1_testing", (symbol, user_id, 5)
            )
            self.conn.commit()
            rows = cursor.fetchone()
        except Exception as e:
            self.conn.rollback()
            print(f"An error occurred: {e}")

        return rows

    def update_filling_urls(self, url_with_id, filling_type):
        try:
            q = """
                UPDATE sec_filling_record
                SET {column_name} = %s
                WHERE id = %s;
            """
            cursor = self.conn.cursor()
            if filling_type:
                column_name = "pdf_filling_url"
            else:
                column_name = "html_filling_url"
            # Loop through the list of IDs and URLs
            for id, filling_url in url_with_id:
                data = (filling_url, id)
                cursor.execute(q.format(column_name=column_name), data)
            # Commit all updates at once
            self.conn.commit()
        except Exception as e:
            print(f"Error updating filings: {e}")
