import nest_asyncio
from dotenv import load_dotenv
import os
import argparse
from typing import List

from pypdf import PdfReader, PdfWriter
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI

# Apply necessary patch for asyncio in interactive environments
nest_asyncio.apply()

load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
DATA_DIR = os.getenv('DATA_DIR_PATH') # Base directory where the output PDFs will be saved
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class KnowledgeParser:
    def __init__(self, api_key, result_type="markdown", language="en", verbose=True):
        """
        Initialize the KnowledgeParser with the required API key and optional parameters.

        Parameters:
        - api_key (str): API key for LlamaParse.
        - result_type (str): The type of result to return, either "markdown" or "text".
        - language (str): The language of the documents to parse. Default is English.
        - verbose (bool): Whether to print verbose output during parsing.
        """
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,
            verbose=verbose,
            language=language
        )

        self.node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8) # TODO: why 8 workers and why this model
    
    def extract_pages(self, pdf_path: str, start_page: int, end_page: int = None) -> str:
        """
        Extracts a range of pages from a PDF file and saves them to a temporary file.

        Parameters:
        - pdf_path (str): Path to the original PDF file.
        - start_page (int): The first page to extract (0-indexed).
        - end_page (int): The last page to extract (inclusive, 0-indexed).

        Returns:
        The path to the temporary PDF file with the extracted pages.
        """

        # Base directory where the output PDFs will be saved
        data_dir = os.getenv('DATA_DIR_PATH')

        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # Determine the last page if end_page is None
        if end_page is None:
            end_page = len(reader.pages) - 1

        for page in range(start_page, end_page + 1):
            try:
                writer.add_page(reader.pages[page])
            except IndexError:
                print(f"Page {page} is out of range.")
                break  # Exceeds the number of pages in the document

        # Extract the base name of the original file without its extension
        base_name = os.path.basename(pdf_path)
        base_name_no_ext = os.path.splitext(base_name)[0]

        # Construct the output filename using the original file name and the page range
        output_filename = f"{base_name_no_ext}_pages_{start_page}_to_{end_page}.pdf"
        # Construct the absolute path for the output file
        output_path = os.path.join(data_dir, output_filename)

        with open(output_path, "wb") as output_pdf:
            writer.write(output_pdf)
        
        return output_path

    async def parse_pdf_async(self, pdf_path: str) -> List[Document]:
        """
        Asynchronously parse the content of a PDF file.

        Parameters:
        - pdf_path (str): Path to the PDF file to parse.

        Returns:
        A list of parsed documents.
        """
        return await self.parser.aload_data(pdf_path)

    def parse_pdf_sync(self, pdf_path: str) -> List[Document]:
        """
        Synchronously parse the content of a PDF file.

        Parameters:
        - pdf_path (str): Path to the PDF file to parse.

        Returns:
        A list of parsed documents.
        """
        return self.parser.load_data(pdf_path)
    

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process a PDF with LlamaParse")

    # Add arguments for the PDF file path and optionally start and end pages
    parser.add_argument("pdf_path", help="The path to the PDF file to parse")
    parser.add_argument("--start_page", type=int, default=0, help="The start page number to parse (0-indexed)")
    parser.add_argument("--end_page", type=int, help="The end page number to parse (inclusive, 0-indexed)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Initialize the KnowledgeParser
    knowledge_parser = KnowledgeParser(api_key=LLAMA_CLOUD_API_KEY)

    # Extract the relevant pages
    new_pdf_path = knowledge_parser.extract_pages(args.pdf_path, args.start_page, args.end_page)
    documents = knowledge_parser.parse_pdf_sync(new_pdf_path)
