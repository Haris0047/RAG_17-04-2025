from __future__ import annotations

import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import os
import random

# chunking
import re
import subprocess
from collections import defaultdict
from pprint import pprint as p
from datetime import datetime
import numpy as np
import requests
import sec_downloader as sd
import sec_parser as sp
import weaviate

# Convert raw documents' format
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import ChatVectorDBChain

# Generation
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
from langchain_community.document_loaders import TextLoader

# Reranking
from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings.openai import OpenAIEmbeddings

# PGvector database
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_weaviate.vectorstores import WeaviateVectorStore

# SEC Parser
from sec_downloader import Downloader
from sec_parser import Edgar10QParser, TextElement, TitleElement, TopSectionTitle
from sklearn.metrics.pairwise import cosine_similarity
from weaviate.auth import AuthApiKey

from . import helper

from .db_work import SQLTool
from .pdf_to_gcp import HtmlToPdfGcpUploader

# jupyter_black.load()

load_dotenv()
gcp_instance = HtmlToPdfGcpUploader()

main_path = os.getcwd() + "/data/html/"

os.makedirs(main_path, exist_ok=True)

class qualitative:

    def __init__(self):
        # self.client = weaviate.Client(
        #     url=os.getenv("WEAVIATE_URL"),
        #     auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        #     additional_headers={
        #         "X-OpenAI-Api-Key": os.getenv(
        #             "OPENAI_API_KEY"
        #         )  # Replace with your inference API key
        #     },
        # )
        self.client = weaviate.Client(
            url="https://weaviate.traderverse.io/",
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv(
                    "OPENAI_API_KEY"
                )  # Replace with your inference API key
            },
        )
        # self.client = weaviate.Client(url="http://localhost:8080")
        self.db = SQLTool()
        self.dl = Downloader("Traderware", "x.tan@traderverse.io")

    def create_class(self, ticker):

        schema = {
            "classes": [
                {
                    "class": ticker,
                    "description": f"A class to represent the data of company using it's ticker {ticker}",
                    "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
                    "moduleConfig": {
                        "text2vec-openai": {},
                        "generative-openai": {},  # Ensure the `generative-openai` module is used for generative queries
                    },
                }
            ]
        }
        try:
            print("creatign schema")
            self.client.schema.create(schema)
        except Exception as e:
            print(e)

    def highlight_text_in_html(self, html_content, search_text):
        soup = BeautifulSoup(html_content, "html.parser")
        search_text = search_text.replace("\n", " ").replace("\u00a0", " ")
        i = 0
        # print(search_text.split("."))
        for tag in soup.find_all("p"):  # Loop through all tags\n
            # print(tag)
            # break
            if tag.string:
                i += 1
                new_string = re.sub(
                    r"\s+",
                    " ",
                    tag.string.replace(".", "").replace("\u00a0", " ").strip(),
                )
                for phrase in search_text.split(". "):
                    phrase = phrase.replace(".", "")
                    if phrase in new_string:

                        if len(phrase) > 3:
                            # print("this is the length", len(phrase))
                            new_string = new_string.replace(
                                phrase,
                                f"""<mark style="background-color:lightorange">{phrase}</mark>""",
                            )
                            print("present")
                tag.string.replace_with(BeautifulSoup(new_string, "html.parser"))
                # tag.clear()  # Clears the content of the tag
                # tag.append(BeautifulSoup(new_string, "html.parser"))
        return str(soup)

    def correct_spacing(self, text):
        # Replace any period not followed by a space with a period followed by a space
        corrected_text = re.sub(r"\.(?!\s)", ". ", text)
        # Remove any extra space before the period
        corrected_text = re.sub(r"\s+\.", ".", corrected_text)
        return corrected_text

    def highlight_text_in_html11(self, html_content, search_text):
        # If the tag has nested elements, recursively traverse them
        soup = BeautifulSoup(html_content, "html.parser")
        search_text = (
            search_text.replace("\n", " ").replace("\u00a0", " ").replace("..", ". ")
        )
        search_text_corrected = self.correct_spacing(search_text)

        def highlight_phrase_in_tag(tag, phrase):
            """
            Recursively search through the tag's contents and highlight matching phrases without losing structure.
            """
            # If the tag has nested elements, recursively traverse them
            if tag.contents:
                for child in tag.contents:
                    if isinstance(child, str):
                        # Clean the text for comparison

                        new_string = re.sub(
                            r"\s+",
                            " ",
                            child.replace(".", "")
                            .replace("\u00a0", " ")
                            .replace("&reg;", "")
                            .replace(",", "")
                            .strip(),
                        )
                        # print("###")
                        # print(new_string)
                        # print("$$$")
                        # print(phrase)
                        if len(phrase) > 3 and len(new_string) > 3:
                            # print("here")
                            if phrase in new_string:
                                if len(phrase) > 20:
                                    # print("present")
                                    # Replace the phrase with the highlighted version
                                    highlighted = new_string.replace(
                                        phrase,
                                        f'<mark style="background-color:lightorange">{phrase}</mark>',
                                    )
                                    child.replace_with(
                                        BeautifulSoup(highlighted, "html.parser")
                                    )
                            elif new_string in phrase:
                                if len(new_string) > 20:
                                    highlighted = new_string.replace(
                                        new_string,
                                        f'<mark style="background-color:lightorange">{new_string}</mark>',
                                    )
                                    child.replace_with(
                                        BeautifulSoup(highlighted, "html.parser")
                                    )
                    else:
                        # Recursively call this function for nested tags
                        highlight_phrase_in_tag(child, phrase)

        # Loop through all <p> tags
        for tag in soup.find_all("p"):
            # print("*****", search_text)
            # phrases = re.split(r"[.:]\s*", search_text)

            # # Iterate over the split phrases
            # for phrase in phrases:

            for phrase in search_text.split(". "):
                # print("*****", phrase)
                phrase = re.sub(
                    r"\s+",
                    " ",
                    phrase.replace(".", "").replace(",", "").strip(),
                )
                # phrase = phrase.replace(".", " ").replace(",", "")

                if len(phrase) > 3:

                    # Call the recursive function to highlight text within each <p> tag
                    highlight_phrase_in_tag(tag, phrase)
        for tag in soup.find_all("div"):
            # print("*****", search_text)
            # phrases = re.split(r"[.:]\s*", search_text)

            # # Iterate over the split phrases
            # for phrase in phrases:

            for phrase in search_text_corrected.split(". "):
                # print("*****", phrase)
                phrase = re.sub(
                    r"\s+",
                    " ",
                    phrase.replace(".", "").replace(",", "").strip(),
                )
                # phrase = phrase.replace(".", " ").replace(",", "")

                if len(phrase) > 3:

                    # Call the recursive function to highlight text within each <p> tag
                    highlight_phrase_in_tag(tag, phrase)
        return str(soup)

    def check_data_from_db(self, ticker, start_date, end_date, filling_type):

        file_record = self.db.fetch_sec_files(
            ticker, start_date, end_date, filling_type
        )
        # file_record=[
        #     {"file_name":"","link":"https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm"},

        # ]    #ticker_date_filling_type-reportingperiod

        return file_record

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

    def processing_html2txt(self, html):
        elements = sp.Edgar10QParser().parse(html)
        top_level_sections = [
            item for part in sp.TreeBuilder().build(elements) for item in part.children
        ]

        # Get the levels we need in the document
        levels = sorted(
            {
                k.semantic_element.level
                for k in top_level_sections
                if isinstance(k.semantic_element, (sp.TopSectionTitle, sp.TitleElement))
            }
        )
        level_to_markdown = {level: "#" * (i + 2) for i, level in enumerate(levels)}

        # Function to extract all the text (excluding tables), and cnvert to markdown format
        def convert_to_markdown(sections):
            markdown = ""
            for section in sections:
                if isinstance(
                    section.semantic_element, (TopSectionTitle, TitleElement)
                ):
                    markdown += f"{level_to_markdown.get(section.semantic_element.level, '#')} {section.semantic_element.text}\n"
                elif isinstance(section.semantic_element, TextElement):
                    markdown += f"{section.semantic_element.text}\n"
                for child in section.get_descendants():
                    if isinstance(
                        child.semantic_element, (TopSectionTitle, TitleElement)
                    ):
                        markdown += f"{level_to_markdown.get(child.semantic_element.level, '#')} {child.semantic_element.text}\n"
                    elif isinstance(child.semantic_element, TextElement):
                        markdown += f"{child.semantic_element.text}\n"
            return markdown

        # recall the function to extract all the text (excluding tables)
        raw_essay = convert_to_markdown(top_level_sections)
        return raw_essay

    def convert_html_to_pdf(self, input_html, output_pdf):
        try:
            try:
                command = ["wkhtmltopdf", input_html, output_pdf]
                result = subprocess.run(command, capture_output=True, text=True)

            except:
                # Construct the wkhtmltopdf command
                wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
                # Construct the command
                command = [wkhtmltopdf_path, input_html, output_pdf]

                result = subprocess.run(command, capture_output=True, text=True)
            # Check if the command was successful
            if result.returncode == 0:
                print(f"PDF successfully created: {output_pdf}")
            else:
                print(f"Error in conversion: {result.stderr}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def getting_data(self, file, status=True):
        try:
            print("trying")
            html_content = self.dl.download_filing(url=file["final_link"]).decode()
            # print(html_content)
        except:
            html_content = self.download_using_request(file["final_link"])

        # # converting html to pdf
        # output_pdf = (
        #     file["symbol"] + "_" + file["type"] + "_" + file["filling_date"] + ".pdf"
        # )
        # self.convert_html_to_pdf(file["final_link"], output_pdf)

        # async self.upload_to_gcp(output_pdf)

        # os.remove(output_pdf)
        if status:
            raw_essay = helper.processing_html2txt(html_content)
            single_sentences_list = re.split(r"(?<=[.#:])\s+", raw_essay)

            # Turn this list of sentence into a list of dictionaries for further embedding works:
            sentences = [
                {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
            ]
            sentences = helper.combine_sentences(sentences)

            return sentences, html_content
        else:
            sentences = ""
            return sentences, html_content

    def process_sentences(self, sentences):

        # Use OpenAIEmbeddings to index the conbined sentences
        oaiembeds = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Get the embeddings,do this in batch to make it quicker.
        embeddings = oaiembeds.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        # add this list of embedding to our list of dicts
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        breakpoint_percentile_threshold = 95  # set up the initial threshold value
        chunk_size_ceiling = 1000

        distances, sentences = helper.calculate_cosine_distances(sentences)
        threshold, chunks, chunk_sizes = helper.find_appropriate_threshold(
            sentences, distances, breakpoint_percentile_threshold, chunk_size_ceiling
        )
        # print("Final threshold used:", threshold)

        breakpoint_distance_threshold = np.percentile(distances, threshold)

        # Get the index of the distances that are above the threshold. This will tell us where we should split our text
        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]  # The indices of those breakpoints on list

        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = (
                breakpoint_index
                if i < len(indices_above_thresh) - 1
                else len(distances)
            )
        start_index = 0
        # Create a list to hold the grouped sentences
        chunks = []
        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)

        # print("Most awaited chunks", type(chunks), chunks)

        print("Chunking")
        return chunks

    def ingesting_into_weaviate_db(
        self, sentences, ticker, date, filling_type, file_name
    ):
        i = 0
        self.client.batch.configure(batch_size=300)  # Configure batch
        with self.client.batch as batch:  # Initialize a batch process
            print(i)
            i += 1000
            for sentence in sentences:  # Batch import data
                # print(i)
                i += 1

                # print(sentence)
                properties = {
                    "content": sentence,
                    "file_name": file_name,
                    "date": date,
                    "filling_type": filling_type,
                    "ticker": ticker,
                }
                batch.add_data_object(data_object=properties, class_name=ticker)

    def data_process(self, file):
        print("processing")
        import time
        try:
            a = time.time()
            sentences, html_content = self.getting_data(file)
            # print("sentences", sentences)
            data_list = self.process_sentences(sentences)
            b = time.time()

            safe_date = file["filling_date"].split("T")[0]

            # Create the file name and path
            file_name = f"{file['symbol']}_{file['type']}_{safe_date}"
            
            with open(main_path + file_name + ".html", "w", encoding="utf-8") as file_html:
                file_html.write(html_content)

            gcp_instance.SaveFiling(file_name + ".html", False)

            gcp_instance.delete_files(file_name + ".html", False)
            print("###", file)
            self.ingesting_into_weaviate_db(
                data_list, file["symbol"], file["filling_date"], file["type"], file_name
            )

            return file_name
        except Exception as e:
            print(e)

    def update_db_record(self, file_name, id):
        html_url = f"https://storage.googleapis.com/eod_stock_data/raw/filings/html/{file_name}"
        self.db.update_filling_name(file_name, html_url, id)

    def semantic_chuncking(self, ticker, file_names, queries):
        import json

        responses = {}
        # for file in file_names:
        per_query_response = []
        for query in queries:
            print(query)
            response = (
                self.client.query
                # .get(ticker, ["content", "file_name"])
                .get(
                    ticker,
                    ["content", "file_name", "date", "filling_type", "ticker"],
                )
                .with_near_text({"concepts": [query]})
                .with_where(
                    {
                        "path": ["file_name"],
                        # "operator": "Equal",
                        "operator": "ContainsAny",
                        "valueText": file_names,  # ,Problem with file and query
                    }
                )
                # .with_generate(single_prompt="Explain {answer} as you might to a five-year-old.")
                .with_limit(3)
                .do()
            )

            print(json.dumps(response, indent=4))
            per_query_response.append(response)

        print(per_query_response)

        grouped_data = defaultdict(list)

        for dataQ in per_query_response:

            file_data = dataQ["data"]["Get"][ticker]
            for item in file_data:
                grouped_data[item["file_name"]].append(item)

        # Convert defaultdict to a regular dictionary
        grouped_data = dict(grouped_data)
        # for query_response in per_query_response:
        return grouped_data

    def qualitative_searching(
        self, user_queries, ticker, start_date, end_date, filling_type
    ):

        if filling_type == "ALL":
            filling_type = ["10-K", "10-Q"]
        else:
            filling_type = filling_type.replace(" ", "")
            filling_type = filling_type.split(",")

        print("*" * 50, self.client.schema.exists(ticker), "*" * 50)
        if not self.client.schema.exists(ticker):
            self.create_class(ticker)
        file_names = []
        # file_names_donwload = []
        import time

        a = time.time()
        file_records = self.check_data_from_db(
            ticker, start_date, end_date, filling_type
        )
        # file_records=[{'id': 1, 'symbol': 'AAPL', 'type': '10-Q', 'filling_date': '2024-08-02', 'final_link': 'https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm', 'filling_name': None, 'created_at':""}]
        b = time.time()
        print("time for retrieval", b - a)
        print(file_records)
        for file in file_records:
            if file["filling_name"]:
                file_names.append(file["filling_name"])
                # call gcp function to have html of that file
                # sentences, html_content = self.getting_data(file, False)

            elif file["filling_name"] == None:

                file_name = self.data_process(file)  # saving html in gcp too
                self.update_db_record(file_name, file["id"])
                file_names.append(file_name)
                # file_names_donwload.append(file_name)

        responses = self.semantic_chuncking(ticker, file_names, user_queries)
        search_text = ""
        print(responses)
        # for response in responses:

        files_html_in_dict = gcp_instance.fetched_html(
            list(responses.keys())
        )  # Getting HTML against the file_names
        for key, value in responses.items():
            # all_data = value[0]["data"]["Get"][ticker]

            for data in value:
                search_text += data["content"] + ". "
                print("______")
            print("looking for", search_text)
            # with open("testing.html", "w", encoding="utf-8") as filee:
            #     filee.write(files_html_in_dict[key])
            highlighted_html = self.highlight_text_in_html11(
                files_html_in_dict[key], search_text
            )
            safe_key = key.split("T00:00:00")[0]
            with open(safe_key + ".html", "w", encoding="utf-8") as file:
                file.write(highlighted_html)
        # os.remove(main_path+file_name)
        return responses


# ticker = "AAPL"
# start_date = "2013-01-01"
# end_date = "2013-12-31"
# filling_type = "10-K,10-Q"
# user_queries = ["how many stores of apple"]


# isinstance1 = qualitative()

# a = time.time()

# result = isinstance1.qualitative_searching(
#     user_queries, ticker, start_date, end_date, filling_type
# )
# b = time.time()
# print("TIME", b - a)
# print("result: ", result)


# ticker = "AAPL"
# start_date = "2024-08-01"
# end_date = "2024-12-01"


# qual_instance = qualitative()
# a = time.time()
# result = qual_instance.qualitative_searching(
#     user_queries, ticker, start_date, end_date, filling_type
# )
# print(result)
# b = time.time()
# print("time for retrieval", b - a)
