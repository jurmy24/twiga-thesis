import nest_asyncio
from llama_parse import LlamaParse
from pypdf import PdfReader, PdfWriter

# Apply necessary patch for asyncio in interactive environments
nest_asyncio.apply()

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
    
    def extract_pages(self, pdf_path, start_page, end_page):
        """
        Extracts a range of pages from a PDF file and saves them to a temporary file.

        Parameters:
        - pdf_path (str): Path to the original PDF file.
        - start_page (int): The first page to extract (0-indexed).
        - end_page (int): The last page to extract (inclusive, 0-indexed).

        Returns:
        The path to the temporary PDF file with the extracted pages.
        """
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        for page in range(start_page, end_page + 1):
            try:
                writer.add_page(reader.pages[page])
            except IndexError:
                break  # Exceeds the number of pages in the document

        output_path = f"data/temp_{start_page}_{end_page}.pdf"
        with open(output_path, "wb") as output_pdf:
            writer.write(output_pdf)
        
        return output_path

    async def parse_pdf_async(self, pdf_path):
        """
        Asynchronously parse the content of a PDF file.

        Parameters:
        - pdf_path (str): Path to the PDF file to parse.

        Returns:
        A list of parsed documents.
        """
        return await self.parser.aload_data(pdf_path)

    def parse_pdf_sync(self, pdf_path):
        """
        Synchronously parse the content of a PDF file.

        Parameters:
        - pdf_path (str): Path to the PDF file to parse.

        Returns:
        A list of parsed documents.
        """
        return self.parser.load_data(pdf_path)

# Example usage
if __name__ == "__main__":
    api_key = "your_llama_parse_api_key_here"
    
    pdf_parser = KnowledgeParser(api_key=api_key)

    pdf_path = ""
    new_path = pdf_parser.extract_pages(pdf_path, start_page=5, end_page=10)

    # Adjust start_page and end_page as needed
    documents = pdf_parser.parse_pdf_sync(new_path)
    print(documents)
