import os
from striprtf.striprtf import rtf_to_text
import docx
from PyPDF2 import PdfReader
# import textract  # For older .doc files (needs antiword installed on Linux/Mac)

def read_file(file_path):
    """
    Read text content from various file formats (TXT, RTF, DOC, DOCX, PDF)
    
    Args:
        file_path (str): Path to the file to be read
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If file doesn't exist
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif ext == '.rtf':
            with open(file_path, 'r', encoding='utf-8') as f:
                rtf_content = f.read()
                return rtf_to_text(rtf_content)
                
        elif ext == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
            
        # elif ext == '.doc':
        #     # Using textract which requires antiword to be installed
        #     return textract.process(file_path).decode('utf-8')
            
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                text = []
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
                
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")

# Example usage
# if __name__ == "__main__":
#     file_path = input("Enter file path: ")
#     try:
#         content = read_file(file_path)
#         print("\nFile content:")
#         print(content[:1000] + "..." if len(content) > 1000 else content) 
#     except Exception as e:
#         print(f"Error: {e}")