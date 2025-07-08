import pdfplumber
import fitz  # PyMuPDF
import os
import uuid
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic.dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
import nltk
import spacy
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import yake
import textstat
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model for better text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks with semantic information"""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_header: Optional[str]
    subsection_header: Optional[str]
    paragraph_number: int
    sentence_start: int
    sentence_end: int
    word_count: int
    chunk_text: str
    chunk_type: str  # 'header', 'paragraph', 'list_item', 'table', 'math', etc.
    confidence: float
    semantic_density: float  # Information density score
    readability_score: float  # Flesch reading ease
    key_terms: List[str]  # Extracted key terms
    chunk_index: int  # Position in document

@dataclass
class ReasoningStep:
    """Represents a step in the AI reasoning process"""
    step_id: str
    step_type: str  # 'retrieval', 'analysis', 'synthesis', 'evaluation'
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    timestamp: datetime

class EnhancedDocumentProcessor:
    def __init__(self, embedding_model: str = "intfloat/e5-small-v2"):
        """
        Initialize with enhanced embedding model and advanced search capabilities
        
        Args:
            embedding_model: Primary embedding model
                - "intfloat/e5-large-v2": Best for research/legal/technical docs
                - "BAAI/bge-base-en-v1.5": Compact local fallback
                - "intfloat/multilingual-e5-large-instruct": Multilingual support
        """
        self.embedding_model_name = embedding_model
        
        # Initialize models lazily (only when needed)
        self._embedding_model = None
        self._cross_encoder = None
        self._keybert = None
        self._yake_extractor = None
        self._embedding_cache = {}  # (doc_id, model_name): embeddings
        
        # Initialize ChromaDB with new configuration format
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection with enhanced metadata
        self.collection = self.chroma_client.get_or_create_collection(
            name="omnidoc_documents_enhanced",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store for hybrid search
        self.document_corpus = {}  # {doc_id: {'chunks': [], 'bm25': BM25Okapi, 'tfidf': TfidfVectorizer}}
        
        # Reasoning chain storage
        self.reasoning_chains = {}
        
        logger.info("✅ Document processor initialized (models will load on first use)")

    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            try:
                logger.info(f"📥 Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"✅ Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {self.embedding_model_name}, falling back to bge-small-en")
                self._embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self._embedding_model_name = "BAAI/bge-small-en-v1.5"
        return self._embedding_model

    @property
    def cross_encoder(self):
        """Lazy load cross-encoder"""
        if self._cross_encoder is None:
            try:
                logger.info("📥 Loading distilled cross-encoder for reranking")
                # Use distilled/quantized model for speed
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
                logger.info("✅ Loaded distilled cross-encoder for reranking")
                # To quantize: see ONNX/bitsandbytes instructions in README
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cross-encoder: {e}")
                self._cross_encoder = None
        return self._cross_encoder

    @property
    def keybert(self):
        """Lazy load KeyBERT"""
        if self._keybert is None:
            try:
                logger.info("📥 Loading KeyBERT for key term extraction")
                self._keybert = KeyBERT()
                logger.info("✅ Loaded KeyBERT for key term extraction")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load KeyBERT: {e}")
                self._keybert = None
        return self._keybert

    @property
    def yake_extractor(self):
        """Lazy load YAKE"""
        if self._yake_extractor is None:
            try:
                logger.info("📥 Loading YAKE for keyword extraction")
                self._yake_extractor = yake.KeywordExtractor(
                    lan="en", 
                    n=1, 
                    dedupLim=0.9, 
                    top=10
                )
                logger.info("✅ Loaded YAKE for keyword extraction")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load YAKE: {e}")
                self._yake_extractor = None
        return self._yake_extractor

    def extract_text_with_structure(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Extract text with enhanced structural information"""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_pdf_with_structure(file_path)
        elif file_extension == '.txt':
            return self._extract_txt_with_structure(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _extract_pdf_with_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF with page numbers, headers, and structure"""
        structured_content = {
            "pages": [],
            "headers": [],
            "full_text": ""
        }
        
        try:
            # Use PyMuPDF for better structure extraction
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with positioning
                blocks = page.get_text("dict")["blocks"]  # type: ignore
                page_content = {
                    "page_number": page_num + 1,
                    "blocks": [],
                    "headers": [],
                    "text": ""
                }
                
                for block in blocks:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        
                        block_text = block_text.strip()
                        if block_text:
                            # Detect if this is a header based on font size and position
                            font_size = block["lines"][0]["spans"][0]["size"] if block["lines"] else 12
                            is_header = font_size > 14 or (
                                len(block_text.split()) <= 10 and 
                                block_text.isupper() or 
                                re.match(r'^[0-9]+\.', block_text)
                            )
                            
                            block_info = {
                                "text": block_text,
                                "font_size": font_size,
                                "is_header": is_header,
                                "bbox": block["bbox"]
                            }
                            
                            page_content["blocks"].append(block_info)
                            page_content["text"] += block_text + "\n"
                            
                            if is_header:
                                page_content["headers"].append(block_text)
                                structured_content["headers"].append({
                                    "text": block_text,
                                    "page": page_num + 1
                                })
                
                structured_content["pages"].append(page_content)
                structured_content["full_text"] += page_content["text"] + "\n"
            
            doc.close()
            
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            # Fallback to pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        structured_content["pages"].append({
                            "page_number": page_num + 1,
                            "text": text,
                            "blocks": [],
                            "headers": []
                        })
                        structured_content["full_text"] += text + "\n"
        
        return structured_content

    def _extract_txt_with_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract TXT with basic structure detection"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        
        # Basic structure detection for TXT files
        lines = content.split('\n')
        structured_content = {
            "pages": [{"page_number": 1, "text": content, "blocks": [], "headers": []}],
            "headers": [],
            "full_text": content
        }
        
        # Detect headers (lines that are short and end with numbers or are all caps)
        for line in lines:
            line = line.strip()
            if line and (
                len(line.split()) <= 8 and 
                (line.isupper() or re.match(r'^[0-9]+\.', line))
            ):
                structured_content["headers"].append({
                    "text": line,
                    "page": 1
                })
        
        return structured_content

    def create_semantic_chunks(self, structured_content: Dict[str, Any], filename: str) -> List[ChunkMetadata]:
        """Create semantic-aware chunks with dynamic sizing"""
        chunks = []
        chunk_id_counter = 0
        
        for page in structured_content["pages"]:
            page_num = page["page_number"]
            page_text = page["text"]
            
            # Analyze text complexity and density
            doc = nlp(page_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            # Calculate semantic density for dynamic chunk sizing
            semantic_density = self._calculate_semantic_density(page_text)
            readability_score = textstat.flesch_reading_ease(page_text)  # type: ignore
            
            # Determine optimal chunk size based on content type
            if semantic_density > 0.8:  # Math-heavy or technical content
                target_chunk_size = 400  # Still smaller for technical
            elif semantic_density < 0.3:  # Prose-heavy content
                target_chunk_size = 1000  # Larger for prose
            else:
                target_chunk_size = 800  # Default larger chunk
            
            # Group sentences semantically
            current_chunk = []
            current_word_count = 0
            current_sentence_start = 0
            
            for i, sentence in enumerate(sentences):
                sentence_words = len(sentence.split())
                
                # Check if adding this sentence would exceed target size
                if current_word_count + sentence_words > target_chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_metadata = self._create_chunk_metadata(
                        chunk_text, filename, page_num, current_sentence_start, i-1,
                        chunk_id_counter, semantic_density, readability_score, structured_content
                    )
                    chunks.append(chunk_metadata)
                    chunk_id_counter += 1
                    
                    # Reset for next chunk
                    current_chunk = [sentence]
                    current_word_count = sentence_words
                    current_sentence_start = i
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            
            # Create final chunk for this page
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(
                    chunk_text, filename, page_num, current_sentence_start, len(sentences)-1,
                    chunk_id_counter, semantic_density, readability_score, structured_content
                )
                chunks.append(chunk_metadata)
                chunk_id_counter += 1
        
        return chunks

    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density (ratio of technical terms to total words)"""
        doc = nlp(text)
        
        # Count technical indicators
        technical_indicators = 0
        total_words = len(doc)
        
        for token in doc:
            # Check for technical patterns
            if (token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 8) or \
               re.search(r'\d+', token.text) or \
               re.search(r'[A-Z]{3,}', token.text) or \
               token.text.lower() in ['algorithm', 'method', 'analysis', 'function', 'variable']:
                technical_indicators += 1
        
        return technical_indicators / max(total_words, 1)

    def _create_chunk_metadata(self, chunk_text: str, filename: str, page_num: int, 
                              sentence_start: int, sentence_end: int, chunk_id: int,
                              semantic_density: float, readability_score: float,
                              structured_content: Dict[str, Any]) -> ChunkMetadata:
        """Create enhanced chunk metadata with key terms"""
        
        # Find relevant headers
        relevant_headers = self._find_relevant_headers(chunk_text, structured_content["headers"], page_num)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(chunk_text, relevant_headers)
        
        # Extract key terms
        key_terms = self._extract_key_terms(chunk_text)
        
        return ChunkMetadata(
            chunk_id=f"chunk_{chunk_id}",
            document_id="",  # Will be set later
            filename=filename,
            page_number=page_num,
            section_header=relevant_headers[0] if relevant_headers else None,
            subsection_header=relevant_headers[1] if len(relevant_headers) > 1 else None,
            paragraph_number=chunk_id + 1,
            sentence_start=sentence_start,
            sentence_end=sentence_end,
            word_count=len(chunk_text.split()),
            chunk_text=chunk_text,
            chunk_type=chunk_type,
            confidence=0.9,
            semantic_density=semantic_density,
            readability_score=readability_score,
            key_terms=key_terms,
            chunk_index=chunk_id
        )

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms using multiple methods"""
        key_terms = []
        
        # Method 1: KeyBERT
        if self.keybert:
            try:
                keywords = self.keybert.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
                key_terms.extend([kw[0] for kw in keywords[:5]])
            except Exception as e:
                logger.warning(f"KeyBERT extraction failed: {e}")
        
        # Method 2: YAKE
        if self.yake_extractor:
            try:
                keywords = self.yake_extractor.extract_keywords(text)
                key_terms.extend([kw[0] for kw in keywords[:3]])
            except Exception as e:
                logger.warning(f"YAKE extraction failed: {e}")
        
        # Method 3: Simple pattern matching
        doc = nlp(text)
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 4 and 
                token.text.lower() not in ['this', 'that', 'these', 'those']):
                key_terms.append(token.text.lower())
        
        # Remove duplicates and limit
        return list(set(key_terms))[:10]

    def _find_relevant_headers(self, chunk_text: str, headers: List[Dict], page_num: int) -> List[str]:
        """Find headers relevant to this chunk"""
        relevant_headers = []
        
        try:
            for header in headers:
                # Ensure header is a dictionary with required keys
                if isinstance(header, dict) and "page" in header and "text" in header:
                    if header["page"] <= page_num:
                        # Check if header is mentioned in chunk or chunk is under this header
                        if (
                            header["text"].lower() in chunk_text.lower() or
                            any(word in chunk_text.lower() for word in header["text"].lower().split())
                        ):
                            relevant_headers.append(header["text"])
                elif isinstance(header, str):
                    # Handle case where header is just a string
                    if header.lower() in chunk_text.lower():
                        relevant_headers.append(header)
        except Exception as e:
            logger.warning(f"Error processing headers: {e}")
        
        return relevant_headers[:2]  # Limit to 2 levels

    def _determine_chunk_type(self, chunk_text: str, headers: List[str]) -> str:
        """Determine the type of chunk based on content"""
        text_lower = chunk_text.lower()
        
        if headers:
            return "header_section"
        elif re.search(r'^\d+\.', chunk_text.strip()):
            return "list_item"
        elif len(chunk_text.split()) < 20:
            return "short_text"
        elif re.search(r'[A-Z]{3,}', chunk_text):
            return "acronym_heavy"
        else:
            return "paragraph"

    def process_document(self, file_path: str, filename: str, api_keys: Optional[Dict[str, str]] = None, provider: str = "openai") -> Dict[str, Any]:
        """Process a document with enhanced metadata and semantic analysis"""
        if api_keys is None:
            api_keys = {}
        try:
            # Extract structured content with headers
            structured_content = self.extract_text_with_structure(file_path, filename)
            content = structured_content["full_text"]
            
            # Generate enhanced summary with API keys if available
            summary = self.generate_summary(content, max_words=150, api_keys=api_keys, provider=provider)
            
            # Create enhanced chunks with metadata
            chunks = self.create_semantic_chunks(structured_content, filename)
            
            # Extract document headers
            headers = structured_content["headers"]
            
            # Store in vector database with enhanced metadata
            doc_id = str(uuid.uuid4())
            self.store_enhanced_chunks(doc_id, chunks)
            
            return {
                "id": doc_id,
                "name": filename,
                "type": os.path.splitext(filename)[1].lower(),
                "content": content,
                "summary": summary,
                "upload_time": datetime.now().isoformat(),
                "chunks": [chunk.chunk_text for chunk in chunks],
                "chunk_metadata": [self._chunk_to_dict(chunk) for chunk in chunks],
                "headers": headers
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise

    def _chunk_to_dict(self, chunk: ChunkMetadata) -> Dict[str, Any]:
        """Convert chunk metadata to dictionary"""
        return {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "section_header": chunk.section_header,
            "subsection_header": chunk.subsection_header,
            "paragraph_number": chunk.paragraph_number,
            "chunk_type": chunk.chunk_type,
            "word_count": chunk.word_count,
            "confidence": chunk.confidence
        }

    def store_enhanced_chunks(self, doc_id: str, chunks: List[ChunkMetadata]):
        """Store chunks with enhanced metadata and prepare for hybrid search"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare enhanced metadata
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk.document_id = doc_id  # Set document ID
            
            metadata = {
                "document_id": doc_id,
                "filename": chunk.filename,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section_header": chunk.section_header or "",
                "subsection_header": chunk.subsection_header or "",
                "paragraph_number": chunk.paragraph_number,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "confidence": chunk.confidence,
                "semantic_density": chunk.semantic_density,
                "readability_score": chunk.readability_score,
                "key_terms": ", ".join(chunk.key_terms) if chunk.key_terms else "",
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text
            }
            
            metadatas.append(metadata)
            ids.append(f"{doc_id}_{chunk.chunk_id}")
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Prepare for hybrid search
        self._prepare_hybrid_search_index(doc_id, chunks)

    def _prepare_hybrid_search_index(self, doc_id: str, chunks: List[ChunkMetadata]):
        """Prepare BM25 and TF-IDF indices for hybrid search"""
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        
        # Tokenize for BM25
        tokenized_chunks = []
        for text in chunk_texts:
            doc = nlp(text.lower())
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            tokenized_chunks.append(tokens)
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(chunk_texts)
        
        # Store indices
        self.document_corpus[doc_id] = {
            'chunks': chunk_texts,
            'bm25': bm25,
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix
        }

    def log_reasoning_chain(self, session_id: str, steps: List[ReasoningStep]):
        """Log reasoning chain for transparency"""
        self.reasoning_chains[session_id] = {
            'session_id': session_id,
            'timestamp': datetime.now(),
            'steps': steps
        }

    def get_reasoning_chain(self, session_id: str) -> Optional[Dict]:
        """Get reasoning chain for a session"""
        return self.reasoning_chains.get(session_id)

    def generate_summary(self, content: str, max_words: int = 150, api_keys: Optional[Dict[str, str]] = None, provider: str = "openai") -> str:
        if api_keys is None:
            api_keys = {}
        try:
            # Use the LLM service for structured summary generation
            from .llm_service import LLMService
            llm_service = LLMService(api_keys)
            
            # Get structured summary
            summary_data = llm_service.generate_summary(content, provider=provider)
            
            # Extract summary text
            summary = summary_data.get("summary", "")
            
            # Ensure summary is exactly max_words or less
            summary_words = summary.split()
            if len(summary_words) > max_words:
                # Truncate to exact word limit
                summary_words = summary_words[:max_words]
                summary = " ".join(summary_words)
                
                # Ensure proper ending
                if not summary.endswith(('.', '!', '?')):
                    summary += '.'
            
            # Validate summary quality
            if summary and len(summary.strip()) > 20:
                return summary
            else:
                # Fall back to semantic summary
                return self._generate_fallback_summary(content, max_words)
            
        except Exception as e:
            logger.warning(f"Structured summary generation failed for provider '{provider}': {e}")
            # Fallback to semantic summary
            return self._generate_fallback_summary(content, max_words)

    def _generate_fallback_summary(self, content: str, max_words: int = 150) -> str:
        """Generate semantic summary using sentence scoring when LLM fails"""
        try:
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                # If no meaningful sentences, create basic summary
                word_count = len(content.split())
                char_count = len(content)
                return f"Document contains {word_count} words and {char_count} characters. Content has been processed and indexed for AI-powered analysis."
            
            # Score sentences based on multiple factors
            scored_sentences = []
            for sentence in sentences:
                score = 0
                
                # Length score (prefer medium-length sentences)
                length = len(sentence.split())
                if 10 <= length <= 30:
                    score += 2
                elif 5 <= length <= 50:
                    score += 1
                
                # Keyword density score
                words = sentence.lower().split()
                if len(words) > 0:
                    # Check for important words (not stop words)
                    important_words = [w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'about', 'many', 'then', 'them', 'these', 'some', 'what', 'more', 'very', 'when', 'just', 'into', 'than', 'only', 'over', 'such', 'most', 'make', 'like', 'after', 'first', 'well', 'should', 'because', 'through', 'during', 'before', 'between', 'under', 'never', 'always', 'often', 'sometimes', 'usually', 'rarely', 'never', 'always', 'often', 'sometimes', 'usually', 'rarely']]
                    keyword_density = len(important_words) / len(words)
                    score += keyword_density * 3
                
                # Position score (prefer early sentences)
                position = sentences.index(sentence)
                if position < len(sentences) // 3:
                    score += 1
                
                # Capitalization score (prefer sentences that start with capital letters)
                if sentence and sentence[0].isupper():
                    score += 0.5
                
                scored_sentences.append((score, sentence))
            
            # Sort by semantic importance score
            scored_sentences.sort(reverse=True)
            
            # Build summary with strict word limit
            summary_parts = []
            word_count = 0
            
            for _, sentence in scored_sentences:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= max_words:
                    summary_parts.append(sentence)
                    word_count += sentence_words
                else:
                    # Try to fit partial sentence if it's very important
                    if word_count < max_words - 10:  # Leave some buffer
                        words = sentence.split()
                        remaining_words = max_words - word_count
                        partial_sentence = " ".join(words[:remaining_words])
                        if partial_sentence:
                            summary_parts.append(partial_sentence + "...")
                    break
            
            summary = " ".join(summary_parts)
            if summary and not summary.endswith(('.', '...')):
                summary += '.'
            
            return summary if summary else f"Document processed successfully. Content contains {len(content.split())} words and has been indexed for analysis."
            
        except Exception as e:
            logger.error(f"Fallback summary generation failed: {e}")
            return "Document processed successfully. Content has been analyzed and indexed for AI-powered queries."

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document with enhanced metadata"""
        results = self.collection.get(where={"document_id": document_id})  # type: ignore
        
        if not results['documents'] or results['documents'][0] is None:
            return []
        
        chunks = []
        for i, doc in enumerate(results['documents']):
            metadata = results['metadatas'][i] if results['metadatas'] is not None else {}
            chunks.append({
                "text": doc,
                "metadata": metadata
            })
        
        return chunks

    def delete_document(self, document_id: str):
        """Delete document and all its chunks"""
        self.collection.delete(where={"document_id": document_id})

    def hybrid_search_with_mmr(self, query: str, document_ids: Optional[List[str]] = None, 
                              top_k: int = 10, diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        if document_ids is None:
            document_ids = []
        t0 = time.time()
        # Step 1: Dense retrieval
        where_clause = None
        if document_ids:
            where_clause = {"document_id": {"$in": document_ids}}
        
        dense_results = self.collection.query(
            query_embeddings=self.embedding_model.encode([query]).tolist(),
            n_results=top_k * 2,  # Get more results for MMR
            where=where_clause if where_clause is not None else None  # type: ignore
        )
        
        if not dense_results['documents'] or dense_results['documents'][0] is None:
            return []
        
        logger.info(f"Dense retrieval took {time.time() - t0:.2f}s")
        t1 = time.time()
        # Step 2: BM25 search
        bm25_results = self._bm25_search(query, document_ids, top_k * 2)
        
        logger.info(f"BM25 search took {time.time() - t1:.2f}s")
        t2 = time.time()
        # Step 3: Combine and rerank
        combined_results = self._combine_search_results(dict(dense_results), bm25_results, query)
        
        logger.info(f"Combine and rerank took {time.time() - t2:.2f}s")
        t3 = time.time()
        # Step 4: Apply MMR diversification
        diversified_results = self._apply_mmr_diversification(combined_results, query, diversity_weight)
        
        logger.info(f"MMR diversification took {time.time() - t3:.2f}s")
        t4 = time.time()
        # Step 5: Cross-encoder reranking (limit to top 5 for speed)
        rerank_limit = 5
        if self.cross_encoder and len(diversified_results) > 1:
            reranked_results = self._rerank_with_cross_encoder(diversified_results[:rerank_limit], query)
        else:
            reranked_results = diversified_results[:rerank_limit]
        
        logger.info(f"Cross-encoder reranking took {time.time() - t4:.2f}s")
        # Step 6: Format results with enhanced metadata
        formatted_results = []
        for result in reranked_results:
            # Debug: Print the full result object
            print("[DEBUG] Formatting result:", result)
            # Robustly get relevance_score: use result['relevance_score'] if > 0, else calculate from distance
            raw_score = result.get("relevance_score", None)
            if raw_score is not None and raw_score > 0:
                relevance_score = raw_score
            else:
                # Fallback: calculate from distance if possible
                dist = result.get("distance", 1.0)
                relevance_score = max(0.0, min(1.0, 1 - dist))
            formatted_result = {
                "chunk": result["chunk"],
                "metadata": result["metadata"],
                "distance": result["distance"],
                "relevance_score": relevance_score,
                "citation": self._generate_citation(result["metadata"]),
                "key_terms": result["metadata"].get("key_terms", []),
                "semantic_density": result["metadata"].get("semantic_density", 0.5),
                "readability_score": result["metadata"].get("readability_score", 50)
            }
            formatted_results.append(formatted_result)
        # Debug: Print top 3 formatted_results before returning
        print("[DEBUG] Top 3 formatted_results returned:", [
            {
                'citation': r.get('citation'),
                'relevance_score': r.get('relevance_score'),
                'chunk_preview': r.get('chunk', '')[:100]
            } for r in formatted_results[:3]
        ])
        return formatted_results

    def _bm25_search(self, query: str, document_ids: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
        if document_ids is None:
            document_ids = []
        """BM25 keyword-based search"""
        bm25_results = []
        
        # Search in relevant documents
        search_docs = document_ids if document_ids else list(self.document_corpus.keys())
        
        for doc_id in search_docs:
            if doc_id in self.document_corpus:
                corpus = self.document_corpus[doc_id]
                if 'bm25' in corpus:
                    # Get BM25 scores
                    scores = corpus['bm25'].get_scores(query.split())
                    chunk_texts = corpus['chunks']
                    
                    # Combine with chunk metadata
                    for i, (score, chunk_text) in enumerate(zip(scores, chunk_texts)):
                        if score > 0:
                            bm25_results.append({
                                "chunk": chunk_text,
                                "score": score,
                                "document_id": doc_id,
                                "chunk_index": i
                            })
        
        # Sort by score and return top_k
        bm25_results.sort(key=lambda x: x["score"], reverse=True)
        return bm25_results[:top_k]

    def _combine_search_results(self, dense_results: Dict, bm25_results: List[Dict], query: str) -> List[Dict]:
        """Combine dense and sparse search results"""
        combined = []
        
        # Add dense results
        for i, (doc, metadata, distance) in enumerate(zip(
            dense_results['documents'][0],
            dense_results['metadatas'][0],
            dense_results['distances'][0]
        )):
            # Print raw distance for debugging
            print(f"[DEBUG] Raw dense distance: {distance}")
            # If distance is in [0, 2] (cosine), normalize to [0, 1]
            if distance > 1:
                norm_distance = (distance - 1) / 1  # Map [1,2] to [0,1]
                relevance_score = max(0.0, min(1.0, 1 - norm_distance))
                print(f"[DEBUG] Normalized (cosine) distance: {norm_distance}, relevance_score: {relevance_score}")
            else:
                relevance_score = max(0.0, min(1.0, 1 - distance))
                print(f"[DEBUG] Standard distance: {distance}, relevance_score: {relevance_score}")
            combined.append({
                "chunk": doc,
                "metadata": metadata,
                "distance": distance,
                "relevance_score": relevance_score,
                "search_type": "dense"
            })
        
        # Add BM25 results with boosting
        for bm25_result in bm25_results:
            # Check if this chunk is already in dense results
            existing = next((r for r in combined if r["chunk"] == bm25_result["chunk"]), None)
            if existing:
                # Boost existing result with proper bounds
                bm25_score = max(0.0, min(1.0, bm25_result["score"]))
                print(f"[DEBUG] BM25 boost: bm25_score={bm25_score}, before={existing['relevance_score']}")
                existing["relevance_score"] = max(0.0, min(1.0, (existing["relevance_score"] + bm25_score * 0.3) / 1.3))
                print(f"[DEBUG] BM25 boost: after={existing['relevance_score']}")
                existing["search_type"] = "hybrid"
            else:
                # Add new result with proper bounds
                bm25_score = max(0.0, min(1.0, bm25_result["score"]))
                print(f"[DEBUG] BM25 only: bm25_score={bm25_score}")
                combined.append({
                    "chunk": bm25_result["chunk"],
                    "metadata": {"document_id": bm25_result["document_id"]},
                    "distance": 1 - bm25_score,
                    "relevance_score": bm25_score,
                    "search_type": "sparse"
                })
        
        # Sort by relevance score
        print(f"[DEBUG] Final combined relevance scores: {[r['relevance_score'] for r in combined]}")
        if all(r['relevance_score'] == 0 for r in combined):
            print("[DEBUG] All relevance scores are zero! Check distance and BM25 normalization.")
        combined.sort(key=lambda x: x["relevance_score"], reverse=True)
        return combined

    def _apply_mmr_diversification(self, results: List[Dict], query: str, diversity_weight: float) -> List[Dict]:
        """Apply Maximal Marginal Relevance for result diversification"""
        if len(results) <= 1:
            return results
        
        # Encode query and results
        query_embedding = self.embedding_model.encode([query])[0]
        result_embeddings = self.embedding_model.encode([r["chunk"] for r in results])
        
        # MMR algorithm
        selected_indices = [0]  # Start with highest relevance result
        remaining_indices = list(range(1, len(results)))
        
        while len(selected_indices) < len(results) and remaining_indices:
            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = results[idx]["relevance_score"]
                
                # Diversity from selected results
                diversity = 0
                for selected_idx in selected_indices:
                    diversity += cosine_similarity(
                        [result_embeddings[idx]], 
                        [result_embeddings[selected_idx]]
                    )[0][0]
                diversity = diversity / len(selected_indices)
                
                # MMR score
                mmr_score = diversity_weight * relevance + (1 - diversity_weight) * (1 - diversity)
                mmr_scores.append((mmr_score, idx))
            
            # Select result with highest MMR score
            mmr_scores.sort(reverse=True)
            selected_indices.append(mmr_scores[0][1])
            remaining_indices.remove(mmr_scores[0][1])
        
        # Return results in MMR order
        return [results[i] for i in selected_indices]

    def _rerank_with_cross_encoder(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not self.cross_encoder:
            return results
        
        # Create pairs for cross-encoder
        pairs = [[query, result["chunk"]] for result in results]
        
        # Get reranking scores
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Combine with original scores
        for i, (result, rerank_score) in enumerate(zip(results, rerank_scores)):
            # Weighted combination: 60% original, 40% rerank
            result["relevance_score"] = result["relevance_score"] * 0.6 + rerank_score * 0.4
        
        # Resort by new scores
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _generate_citation(self, metadata: Dict[str, Any]) -> str:
        """Generate human-readable citation from metadata"""
        citation_parts = []
        
        # Add section header if available and meaningful
        if metadata.get("section_header") and metadata["section_header"].strip():
            citation_parts.append(f"Section: {metadata['section_header']}")
        
        # Add subsection header if available and meaningful
        if metadata.get("subsection_header") and metadata["subsection_header"].strip():
            citation_parts.append(f"Subsection: {metadata['subsection_header']}")
        
        # Add page number if valid
        page_num = metadata.get("page_number", 0)
        if page_num and page_num > 0:
            citation_parts.append(f"Page {page_num}")
        
        # Add paragraph number if valid
        para_num = metadata.get("paragraph_number", 0)
        if para_num and para_num > 0:
            citation_parts.append(f"Paragraph {para_num}")
        
        # If no meaningful citation parts, provide a basic one
        if not citation_parts:
            return "Document content"
        
        return ", ".join(citation_parts)

    def _get_or_cache_embeddings(self, texts, doc_id):
        # Embedding cache per (doc_id, model_name)
        cache_key = (doc_id, self.embedding_model_name)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        # Parallelize embedding
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.embedding_model.encode, texts))
        self._embedding_cache[cache_key] = embeddings
        return embeddings 