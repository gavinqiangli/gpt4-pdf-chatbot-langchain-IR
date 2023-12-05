from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import requests
import json
import os
import html2text
from langchain.chat_models import ChatOpenAI
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from langchain.prompts import ChatPromptTemplate
from llama_index import VectorStoreIndex
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import TextLoader



load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


# 1. Scrape raw HTML

def scrape_website(url: str):

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url,
        "elements": [{
            "selector": "body"
        }]
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        f"https://chrome.browserless.io/scrape?token={brwoserless_api_key}",
        headers=headers,
        data=data_json
    )

    # Check the response status code
    if response.status_code == 200:
        # Decode & Load the string as a JSON object
        result = response.content
        data_str = result.decode('utf-8')
        data_dict = json.loads(data_str)

        # Extract the HTML content from the dictionary
        html_string = data_dict['data'][0]['results'][0]['html']

        return html_string
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# 2. Convert html to markdown

def convert_html_to_markdown(html):

    # Create an html2text converter
    converter = html2text.HTML2Text()

    # Configure the converter
    converter.ignore_links = False

    # Convert the HTML to Markdown
    markdown = converter.handle(html)

    return markdown


# Turn https://developers.webflow.com/docs/getting-started-with-apps to https://developers.webflow.com

def get_base_url(url):
    parsed_url = urlparse(url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


# Turn relative url to absolute url in html

def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')

    for img_tag in soup.find_all('img'):
        if img_tag.get('src'):
            src = img_tag.get('src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['src'] = absolute_url
        elif img_tag.get('data-src'):
            src = img_tag.get('data-src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['data-src'] = absolute_url

    for link_tag in soup.find_all('a'):
        href = link_tag.get('href')
        if href is not None and href.startswith(('http://', 'https://')):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag['href'] = absolute_url

    updated_html = str(soup)

    return updated_html


def get_markdown_from_url(url):
    base_url = get_base_url(url)
    html = scrape_website(url)
    updated_html = convert_to_absolute_url(html, base_url)
    markdown = convert_html_to_markdown(updated_html)

    return markdown


# 3. Create vector index from markdown

def create_index_from_text(docpath):
    
    loader = TextLoader(docpath, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    print('creating vector store...')

    # Create and store the embeddings in the vectorStore
    embeddings = OpenAIEmbeddings()   

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    # name_space = 'web-test'; # namespace is optional for your vectors

    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name, text_key="text")

    index = pinecone.Index(index_name); # change to your own index name

    print("Index created!")
    return index


# 4. Retrieval Augmented Generation (RAG)


def generate_answer(query, index):

    # Get relevant data with similarity search
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    texts = [node.node.text for node in nodes]

    print("Retrieved texts!", texts)

    # Generate answer with OpenAI
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    template = """
    CONTEXT: {docs}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {query}
    ANSWER (formatted in Markdown):
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"docs": texts, "query": query})

    return response.content

# # List of URLs to scrape (web.txt)
# urls = [
#     "https://transformainsights.com/blog/september-events",
#     "https://transformainsights.com/iot-platforms",
#     "https://transformainsights.com/iot-connectivity",
#     "https://transformainsights.com/mobile-private-networks",
#     "https://transformainsights.com/low-power-wide-area-networks",
#     "https://transformainsights.com/5g-iot",
#     "https://transformainsights.com/digital-transformation",
#     "https://transformainsights.com/ai-machine-learning",
#     "https://transformainsights.com/sustainability",
#     "https://transformainsights.com/edge-computing",
#     # Add more URLs here
# ]

# List of URLs to scrape (web2.txt)
urls = [
    "https://transformainsights.com/research/forecast/highlights",
    "https://transformainsights.com/blog/road-connected-by-design",
    "https://transformainsights.com/blog/why-iot-delivered-managed-service",
    "https://transformainsights.com/blog/euicc-reach-24-billion-2032",
    "https://transformainsights.com/blog/global-operator-iot-connections-24-billion",
    "https://transformainsights.com/blog/6-reasons-device-connectivity-bundling",
    "https://transformainsights.com/blog/evolving-iot-security-threat-landscape",
    "https://transformainsights.com/blog/expanding-functionality-iot-cmp",
    "https://transformainsights.com/blog/iot-cmps-period-transition",
    "https://transformainsights.com/blog/new-report-ranks-iot-mvnos",
    "https://transformainsights.com/blog/spin-off-iot-business-unit",
    "https://transformainsights.com/blog/savings-esim-isim-iot",
    "https://transformainsights.com/blog/iot-connectivity-feature-not-product",
    "https://transformainsights.com/blog/enterprise-iot-adopters-favour-platform-off-the-shelf",
    
    # Add more URLs here
]

# Iterate through the list of URLs
#fullmarkdowndoc = ""
docpath = 'docs/web2.txt'
for url in urls:
    markdown = get_markdown_from_url(url)
    if markdown:
        print(f"URL: {url}")   
        #fullmarkdowndoc += markdown 
        #fullmarkdowndoc += '\n\n'
        # Open a file in append mode ('a')
        with open(docpath, 'a', encoding='utf-8') as file:
            file.write(markdown + '\n\n')

    else:   
        print(f"Failed to fetch markdown from URL: {url}")

file.close()
index = create_index_from_text(docpath) 
print("scrapeweb ingestion complete")



