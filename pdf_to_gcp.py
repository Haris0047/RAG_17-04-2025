import datetime
import json
import os
import subprocess

import requests
import sec_downloader as sd
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from sec_downloader import Downloader

# GCP credentials and path
gcp_credential = os.getenv("GCP_FILE_CRED")
from db_work import SQLTool

credentials_dict = json.loads(gcp_credential)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
client = storage.Client(project="stock_data", credentials=credentials)
bucket = client.bucket("eod_stock_data")
GCP_PDF_FOLDER_PATH = "raw/filings/pdf"
GCP_HTML_FOLDER_PATH = "raw/filings/html"

load_dotenv()

main_path = os.getcwd() + "/data/"


class HtmlToPdfGcpUploader:
    # def __init__(self):
    # self.dl = Downloader("Traderware", "x.tan@traderverse.io")
    # self.db = SQLTool()

    def SaveFiling(self, file_name, pdf_filing_type):

        # Loop through the list and save only the specified files
        # for file_name in filig_name_list:

        if pdf_filing_type:
            filing_path = main_path + "pdf"
            os.makedirs(filing_path, exist_ok=True)
            source_path = os.path.join(filing_path, file_name)
            if os.path.exists(source_path):
                # Upload the PDF file to GCP
                blob = bucket.blob(f"{GCP_PDF_FOLDER_PATH}/{file_name}")
                blob.upload_from_filename(source_path)
                print(f"Uploaded {file_name} to GCP bucket {bucket}")
            else:
                print(f"{file_name} does not exist in the source folder.")

        else:
            filing_path = main_path + "html"
            os.makedirs(filing_path, exist_ok=True)
            source_path = os.path.join(filing_path, file_name)
            if os.path.exists(source_path):
                # Upload the PDF file to GCP
                blob = bucket.blob(f"{GCP_HTML_FOLDER_PATH}/{file_name}")
                blob.upload_from_filename(source_path)
                print(f"Uploaded {file_name} to GCP bucket {bucket}")

            else:
                print(f"{file_name} does not exist in the source folder.")

            print(filing_path)
            print(file_name)
            # Check if the file is publicly accessible
            if blob.acl.get_entities():
                blob.make_public()
                print(f"Public URL for {file_name}: {blob.public_url}")
            else:
                print(f"{file_name} is not publicly accessible.")

        print("PDF file upload to GCP completed.")

    def download_using_request(self, link):
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

        # Making the request without the If-Modified-Since header
        response = requests.get(link, headers=headers)
        html_content = response.text
        return html_content

    def convert_html_to_pdf(self, input_html, output_pdf):
        try:
            # Define the path to save the PDF in the 'pdf' folder
            pdf_folder = main_path + "pdf"
            os.makedirs(
                pdf_folder, exist_ok=True
            )  # Create the 'pdf' folder if it doesn't exist
            output_pdf_path = os.path.join(pdf_folder, output_pdf)
            try:
                command = ["wkhtmltopdf", input_html, output_pdf_path]
                result = subprocess.run(command, capture_output=True, text=True)
            except:
                # Construct the wkhtmltopdf command
                wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
                # Construct the command
                command = [wkhtmltopdf_path, input_html, output_pdf_path]
                result = subprocess.run(command, capture_output=True, text=True)
            # Check if the command was successful
            if result.returncode == 0:
                print(f"PDF successfully created: {output_pdf_path}")
            else:
                print(f"Error in conversion: {result.stderr}")
            return pdf_folder
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return pdf_folder

    def delete_files(self, file_name, pdf_filing_type):
        if pdf_filing_type:
            folder_path = main_path + "pdf"
        else:
            folder_path = main_path + "html"

        # for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    def HtmlToPDF(self, files):
        pdf_file_names = []
        url_with_id = []
        # converting html to pdf
        for file in files:
            output_pdf = (
                file["symbol"]
                + "_"
                + file["type"]
                + "_"
                + file["filling_date"]
                + ".pdf"
            )

            self.convert_html_to_pdf(file["final_link"], output_pdf)
            url = f"https://storage.googleapis.com/eod_stock_data/raw/filings/pdf/{output_pdf}"
            id_and_url = (file["id"], url)
            url_with_id.append(id_and_url)

            pdf_file_names.append(output_pdf)

        return pdf_file_names, url_with_id

    def upload_html(self, files):
        html_file_names = []
        url_with_id = []

        # Define the path to save the PDF in the 'pdf' folder
        html_folder = main_path + "html"
        os.makedirs(
            html_folder, exist_ok=True
        )  # Create the 'pdf' folder if it doesn't exist

        for file in files:
            output_html = (
                file["symbol"]
                + "_"
                + file["type"]
                + "_"
                + file["filling_date"]
                + ".html"
            )
            html_file_path = os.path.join(html_folder, output_html)
            try:
                # print("trying")
                html_content = self.dl.download_filing(url=file["final_link"]).decode()
            except:
                html_content = self.download_using_request(file["final_link"])

                # Save the HTML content to a file
            with open(html_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            html_file_names.append(output_html)

            url = f"https://storage.googleapis.com/eod_stock_data/raw/filings/html/{output_html}"
            id_and_url = (file["id"], url)
            url_with_id.append(id_and_url)

        return html_file_names, url_with_id

    def main(self, files, filing_type=True):
        if filing_type:
            files, url_with_id = self.HtmlToPDF(files=files)
        else:
            files, url_with_id = self.upload_html(files=files)

        self.SaveFiling(filig_name_list=files, pdf_filing_type=filing_type)
        self.delete_files(files_to_delete=files, pdf_filing_type=filing_type)
        print(url_with_id)
        self.db.update_filling_urls(url_with_id=url_with_id, filling_type=filing_type)

    def fetched_html(self, file_names):
        print(file_names)
        try:
            html_dict = {}
            for file_name in file_names:
                html_url = f"https://storage.googleapis.com/eod_stock_data/raw/filings/html/{file_name}.html"
                response = requests.get(html_url)
                if response.status_code == 200:
                    html_content = response.content.decode(
                        "utf-8"
                    )  # Decode the HTML content to a string
                    print("HTML file fetched successfully.")
                    html_dict[file_name] = html_content

                else:
                    print(
                        f"Failed to fetch the file. Status code: {response.status_code}"
                    )
                    return None
            return html_dict
        except Exception as e:
            print(f"Failed to download the file due to {e}.")
            return None


url = "AAPL_10-K_2006-12-15.html"
dd = [
    {
        "id": 277,
        "symbol": "AAPL",
        "type": "10-K",
        "filling_date": "2006-12-29",
        "final_link": "https://www.sec.gov/Archives/edgar/data/320193/000110465906084288/a06-25759_210k.htm",
        "filling_name": "AAPL_10-K_2006-12-29.pdf",
        "created_at": datetime.datetime(
            2024, 8, 21, 14, 12, 58, 360996, tzinfo=datetime.timezone.utc
        ),
        "pdf_filling_url": "https://storage.googleapis.com/eod_stock_data/raw/filings/pdf/AAPL_10-K_2006-12-29.pdf",
        "html_filling_url": "https://storage.googleapis.com/eod_stock_data/raw/filings/html/AAPL_10-K_2006-12-29.html",
    },
    {
        "id": 276,
        "symbol": "AAPL",
        "type": "10-K",
        "filling_date": "2007-11-15",
        "final_link": "https://www.sec.gov/Archives/edgar/data/320193/000104746907009340/a2181030z10-k.htm",
        "filling_name": "AAPL_10-K_2007-11-15.pdf",
        "created_at": datetime.datetime(
            2024, 8, 21, 14, 12, 58, 360996, tzinfo=datetime.timezone.utc
        ),
        "filling_url": "https://storage.googleapis.com/eod_stock_data/raw/filings/AAPL_10-K_2007-11-15.pdf",
    },
    {
        "id": 278,
        "symbol": "AAPL",
        "type": "10-K",
        "filling_date": "2006-12-15",
        "final_link": "https://www.sec.gov/Archives/edgar/data/320193/000110465906081617/a06-25759_1nt10k.htm",
        "filling_name": "AAPL_10-K_2006-12-15.pdf",
        "created_at": datetime.datetime(
            2024, 8, 21, 14, 12, 58, 360996, tzinfo=datetime.timezone.utc
        ),
        "filling_url": "https://storage.googleapis.com/eod_stock_data/raw/filings/AAPL_10-K_2006-12-15.pdf",
    },
]

# import time

# a = time.time()
# hpg = HtmlToPdfGcpUploader()
# hpg.main(files=dd)
# b = time.time()
# print(b - a)
