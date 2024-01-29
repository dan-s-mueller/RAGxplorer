"""
Ragxplorer.py, forked by dan
"""
import os
import uuid
import random
import json
from typing import (
    Optional,
    Any
    )

from pydantic import BaseModel, Field
import pandas as pd

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction,
    HuggingFaceEmbeddingFunction
    )

import plotly.graph_objs as go

from .rag import (
    build_vector_database,
    get_doc_embeddings,
    get_docs,
    query_chroma
    )

from .projections import (
    set_up_umap,
    get_projections,
    prepare_projections_df,
    plot_embeddings
    )

from .query_expansion import (
    generate_hypothetical_ans,
    generate_sub_qn
    )

from .constants import OPENAI_EMBEDDING_MODELS


class _Documents(BaseModel):
    text: Optional[Any] = None
    ids: Optional[Any] = None
    embeddings: Optional[Any] = None
    projections: Optional[Any] = None

class _Query(BaseModel):
    original_query: Optional[Any] = None
    original_query_projection: Optional[Any] = None
    actual_search_queries: Optional[Any] = None
    retrieved_docs: Optional[Any] = None

class _VizData(BaseModel):
    base_df: Optional[Any] = None
    query_df: Optional[Any] = None
    visualisation_df: Optional[Any] = None

class RAGxplorer(BaseModel):
    """
    RAGxplorer class for managing the RAG exploration process.
    """
    embedding_model: Optional[str] = Field(default="all-MiniLM-L6-v2")
    _chosen_embedding_model: Optional[Any] = None
    _vectordb: Optional[Any] = None
    _documents: _Documents = _Documents()
    _projector: Optional[Any] = None
    _query: _Query = _Query()
    _VizData: _VizData = _VizData()

    def __init__(self, **data):
        super().__init__(**data)
        self._set_embedding_model()

    def _set_embedding_model(self):
        """ Sets the embedding model """
        if self.embedding_model == 'all-MiniLM-L6-v2':
            print('~ Setting all-MiniLM-L6-v2 embedding model...')
            self._chosen_embedding_model = SentenceTransformerEmbeddingFunction()

        elif self.embedding_model in OPENAI_EMBEDDING_MODELS:
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            print('~ Setting openai embedding model: '+self.embedding_model+'...')
            self._chosen_embedding_model = OpenAIEmbeddingFunction(api_key = os.getenv("OPENAI_API_KEY"), 
                                                                   model_name = self.embedding_model)
        else:
            try:
                if "HF_API_KEY" not in os.environ:
                    raise OSError("HF_API_KEY is not set")
                print('~ Setting hf embedding model...')
                self._chosen_embedding_model = HuggingFaceEmbeddingFunction(api_key = os.getenv("HF_API_KEY"),
                                                                            model_name = self.embedding_model)
            except Exception as exc:
                raise ValueError("Invalid embedding model. Please use all-MiniLM-L6-v2, or a valid OpenAI or HuggingFace embedding model.") from exc

    def load_db(self, document_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 0, 
            path_to_db:str = None, index_name:str = None, 
            df_export_path:str = None,
            vector_qty: float = None,
            umap_params: dict = None,
            verbose: bool = False):
        """
        First checks for document_path to load data from a PDF file and prepare it for exploration. 
        Else, if path_to_db exists, it will connect to the database instead of building it. 
        
        Args:
            document: Path to the PDF document to load.
            chunk_size: Size of the chunks to split the document into.
            chunk_overlap: Number of tokens to overlap between chunks.
            path_to_db: Path to the database to connect to.
            index_name: Name of the index to connect to.
            vector_qty: Number of vectors to build the database with. If blank, all vectors will be used.
        """
        if path_to_db is None:
            if verbose:
                print(" ~ Building the vector database...")
            self._vectordb = build_vector_database(document_path, chunk_size, chunk_overlap, self._chosen_embedding_model)
            if verbose:
                print("Completed Building Vector Database ✓")
        else:
            if verbose:
                print(" ~ Connecting to the vector database...")
            client = chromadb.PersistentClient(path=path_to_db)            
            self._vectordb = client.get_collection(name=index_name,embedding_function=self._chosen_embedding_model)
            if verbose:
                print("Connected to Vector Database ✓")
    
        self._documents.embeddings = get_doc_embeddings(self._vectordb)
        self._documents.text = get_docs(self._vectordb)
        self._documents.ids = self._vectordb.get()['ids']

        if vector_qty is not None:
            # Reduce the number of vectors
            if verbose:
                print(' ~ Reducing the number of vectors from '+str(len(self._documents.embeddings))+' to '+str(vector_qty)+'...')
            indices = random.sample(range(len(self._documents.embeddings)), vector_qty)
            id = str(uuid.uuid4())[:8]
            temp_index_name=index_name+'-'+id
            
            # Create a temporary index with the reduced number of vectors
            client.create_collection(name=temp_index_name,embedding_function=self._chosen_embedding_model)
            temp_collection = client.get_collection(name=temp_index_name,embedding_function=self._chosen_embedding_model)
            temp_collection.add(
                ids=[self._documents.ids[i] for i in indices],
                embeddings=[self._documents.embeddings[i] for i in indices],
                documents=[self._documents.text[i] for i in indices]
            )

            # Replace the original index with the temporary one
            self._vectordb = temp_collection
            self._documents.embeddings = get_doc_embeddings(self._vectordb)
            self._documents.text = get_docs(self._vectordb)
            self._documents.ids = self._vectordb.get()['ids']
            if verbose:
                print('Reduced number of vectors to '+str(len(self._documents.embeddings))+' ✓')
                print('Copy of database saved as '+temp_index_name+' ✓')

        if verbose:
            print(" ~ Reducing the dimensionality of embeddings...")
        self._projector = set_up_umap(embeddings=self._documents.embeddings,
                                      umap_params=umap_params)
        if verbose:
            print('Set up UMAP transformer ✓')
        if verbose:
            print('~ Projecting data...')
        self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                    umap_transform=self._projector)
        self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                       document_projections=self._documents.projections,
                                                       document_text=self._documents.text)
        if df_export_path is not None:
            # Save the parameters to a JSON file
            # Get the parameters of the UMAP transformer and the DataFrame
            export_data = {
                'visualization_index_name' : temp_index_name if 'temp_index_name' in locals() else index_name,
                'umap_params': self._projector.get_params(),
                'viz_data': self._VizData.base_df.to_json(orient='split')
            }

            # Save the data to a JSON file
            with open(df_export_path, 'w') as f:
                json.dump(export_data, f, indent=4)

            if verbose:
                print("Exported flattened dataframe, and umap parameters for visualization ✓")
        if verbose:
            print("Completed reducing dimensionality of embeddings ✓")

    def visualize_query(self, query: str, retrieval_method: str="naive", top_k:int=5, query_shape_size:int=5,path_to_db:str = None, viz_data_df_path:pd.DataFrame=None, verbose:bool = False) -> go.Figure:
        """
        Visualize the query results in a 2D projection using Plotly.

        Args:
            query (str): The query string to visualize.
            retrieval_method (str): The method used for document retrieval. Defaults to 'naive'.
            top_k (int): The number of top documents to retrieve.
            query_shape_size (int): The size of the shape to represent the query in the plot.

        Returns:
            go.Figure: A Plotly figure object representing the visualization.

        Raises:
            RuntimeError: If the document has not been loaded before visualization.
        """
        # Use the provided dataframe for projections, re-establish the projector with the same parameters, connect to database
        if viz_data_df_path is not None: 
            # Read the data from the JSON file
            with open(viz_data_df_path, 'r') as f:
                data = json.load(f)

            # Connect to database
            if verbose:
                print(" ~ Connecting to the vector database...")
            client = chromadb.PersistentClient(path=path_to_db)            
            self._vectordb = client.get_collection(name=data['visualization_index_name'],embedding_function=self._chosen_embedding_model)
            if verbose:
                print("Reconnected to Vector Database ✓")
            self._documents.embeddings = get_doc_embeddings(self._vectordb)
            self._documents.text = get_docs(self._vectordb)
            self._documents.ids = self._vectordb.get()['ids']

            # Assign read in viz_data to the base_df
            self._VizData.base_df = pd.read_json(data['viz_data'], orient='split')
            if verbose:
                print('Read in existing visualization data ✓')

            # Get the parameters of the UMAP transformer and reassign to a new projector
            self._projector = set_up_umap(embeddings=self._documents.embeddings, 
                                          umap_params=data['umap_params'])
            if verbose:
                print('Set up UMAP transformer ✓')        
        else:
            if self._vectordb is None or self._VizData.base_df is None:
                raise RuntimeError("Please load a pdf first.")

        self._query.original_query = query

        if (self.embedding_model == "all-MiniLM-L6-v2") or (self.embedding_model in OPENAI_EMBEDDING_MODELS):
            if verbose:
                print("~ Embedding model all-MiniLM-L6-v2 or OpenAI: "+str(self._chosen_embedding_model))
            self._query.original_query_projection = get_projections(embedding=self._chosen_embedding_model([self._query.original_query]),
                                                                    umap_transform=self._projector)
            if verbose:
                print("Query projection completed ✓")
        else:
            if verbose:
                print("~ Embedding model not all-MiniLM-L6-v2 or OpenAI: "+str(self._chosen_embedding_model))
            self._query.original_query_projection = get_projections(embedding=[self._chosen_embedding_model(self._query.original_query)],
                                                                    umap_transform=self._projector)
            if verbose:
                print("Query projection completed ✓")

        self._VizData.query_df = pd.DataFrame({"x": [self._query.original_query_projection[0][0]],
                                      "y": [self._query.original_query_projection[1][0]],
                                      "document_cleaned": query,
                                      "category": "Original Query",
                                      "size": query_shape_size})

        if retrieval_method == "naive":
            self._query.actual_search_queries = self._query.original_query

        elif retrieval_method == "HyDE":
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._query.actual_search_queries = generate_hypothetical_ans(query=self._query.original_query)

        elif retrieval_method == "HyDE":
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._query.actual_search_queries = generate_sub_qn(query=self._query.original_query)

        if verbose:
            print("~ Querying database...")
        self._query.retrieved_docs = query_chroma(chroma_collection=self._vectordb,
                                                  query=self._query.actual_search_queries,
                                                  top_k=top_k)

        if verbose:
            print("~ Preparing query data for visualization...")
        self._VizData.base_df.loc[self._VizData.base_df['id'].isin(self._query.retrieved_docs), "category"] = "Retrieved"
        
        self._VizData.visualisation_df = pd.concat([self._VizData.base_df, self._VizData.query_df], axis = 0)
        if verbose:
            print("Plot generated ✓")
        return plot_embeddings(self._VizData.visualisation_df)