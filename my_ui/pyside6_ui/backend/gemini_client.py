import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
from typing import List, Dict, Any, Optional

# Import logger
from utils.logger import setup_logger

# Set up logger
logger = setup_logger('backend.gemini_client')

# Load environment variables from .env file
load_dotenv(os.path.join(os.getcwd(),"my_ui","pyside6_ui","backend",".env"))

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in .env file")
    raise ValueError("GOOGLE_API_KEY not found in .env file")
else:
    logger.info("Successfully loaded GOOGLE_API_KEY from environment")

# Configure the Gemini API with the key
# genai.configure(api_key=GOOGLE_API_KEY)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini client with the specified model.
        
        Args:
            model_name: The name of the Gemini model to use.
                        Default is "gemini-1.5-pro"
        """
        self.model_name = model_name
        self.embedding_model = "gemini-embedding-exp-03-07"
        self.generation_model = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info(f"Initialized GeminiClient with model: {model_name}, embedding model: {self.embedding_model}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A list of floats representing the embedding vector
        """
        logger.info("Generating embedding for text")
        try:
            response = self.generation_model.models.embed_content(
                model=self.embedding_model,
                contents=[text],
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )
            logger.info("Successfully generated embedding")
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            Generated text response
        """
        logger.info(f"Generating text with temperature: {temperature}")
        try:
            response = self.generation_model.models.generate_content(
                model=self.model_name,
                contents=[prompt],
            )
            logger.info("Successfully generated text response")
            return response.text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Have a conversation with the model.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
                     Roles can be 'user' or 'model'
            temperature: Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            The model's response text
        """
        logger.info(f"Starting chat with {len(messages)} messages, temperature: {temperature}")
        
        # Convert messages to the format expected by the Gemini API
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        try:
            # Create a chat session
            chat = self.generation_model.start_chat(history=gemini_messages)

            # Get the last user message to respond to
            last_user_msg = next((msg["content"] for msg in reversed(messages) 
                                if msg["role"] == "user"), "")

            # Generate a response
            response = chat.send_message(
                last_user_msg,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            logger.info("Successfully generated chat response")
            return response.text
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise

    def rag_response(self, query: str, context: List[str], temperature: float = 0.7) -> str:
        """
        Generate a response using Retrieval-Augmented Generation (RAG).
        
        Args:
            query: The user's query
            context: A list of relevant context passages retrieved from the vector database
            temperature: Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            The model's response based on the query and context
        """
        logger.info(f"Generating RAG response for query with {len(context)} context passages")
        
        # Combine context into a single string
        context_text = "\n\n".join(context)

        # Create a prompt that includes the context and query
        prompt = f"""
        Context information:
        {context_text}
        
        Based on the context information provided above, please answer the following question:
        {query}
        
        If the context doesn't contain relevant information to answer the question, 
        please state that you don't have enough information.
        """

        try:
            # Generate a response
            response = self.generate_text(prompt, temperature)
            logger.info("Successfully generated RAG response")
            return response
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}", exc_info=True)
            raise

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating batch embeddings for {len(texts)} texts")
        try:
            response = self.generation_model.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )
            logger.info(f"Successfully generated {len(response.embeddings)} embeddings")
            return [embedding.values for embedding in response.embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}", exc_info=True)
            raise
