import asyncio
import logging
from typing import List, Tuple, Optional

from ollama import AsyncClient
from embedding_system import EmailEmbedder

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AdvancedRAGAgent:
    def __init__(self):
        self.client = AsyncClient()
        self.embedder = EmailEmbedder()
        self.embedder.load_embeddings()

    async def needs_rag_processing(self, query: str) -> bool:
        """
        Determine if the query requires to search in an email database,
        returning True/False based on the model's response.
        """
        decision_prompt = f"""
        [SYSTEM]
        Determine if the query requires to search in an email database.
        Respond STRICTLY with 'YES' or 'NO':

        [QUERY]
        {query}
        """

        response = await self.client.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": decision_prompt}]
        )
        answer = response.message['content'].strip().upper()
        return (answer == "YES")

    async def extract_search_terms(self, user_query: str) -> str:
        """
        Reformulate the user's query into an optimized form for semantic search.
        Returns only the optimized query.
        """
        prompt = f"""
        [SYSTEM]
        Reformulate this query into an optimal form for semantic search.
        Keep the original meaning but use more effective retrieval terms.

        [QUERY]
        {user_query}

        Respond ONLY with the optimized query.
        """

        response = await self.client.chat(
            model="deepseek-r1:8b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message['content'].strip()

    def format_rag_results(self, results: List[Tuple[int, float]]) -> str:
        """
        Format the RAG results into a concise text for LLM context.
        """
        context = []
        for idx, score in results:
            try:
                email = self.embedder.get_email_details(idx)
                context.append(
                    f"📩 Email {idx} (Relevance: {score:.2f})\n"
                    f"Subject: {email['subject']}\n"
                    f"From: {email['sender']}\n"
                    f"Content: {email['body'][:300]}..."
                )
            except Exception as e:
                logger.error(f"Error formatting email {idx}: {str(e)}")
        return "\n\n".join(context)

    async def generate_final_response(self, query: str, context: str) -> str:
        """
        Generate a final user-facing response, using context if available.
        """
        response_prompt = f"""
        [SYSTEM]
        You are an email assistant. Use the provided context to answer the query.
        If no relevant emails are found, state that clearly.

        [CONTEXT]
        {context}

        [QUERY]
        {query}

        Provide a clear, concise answer in natural language.
        """

        response = await self.client.chat(
            model="deepseek-r1:8b",
            messages=[{"role": "user", "content": response_prompt}]
        )
        return response.message['content'].strip()

    async def process_query(self, query: str) -> str:
        """
        Full query processing pipeline:
          1) Determine if RAG is needed
          2) Reformulate query for semantic search
          3) Perform RAG search on emails
          4) Generate the final user response
        """
        try:
            # Step 1: Determine RAG necessity
            requires_rag = await self.needs_rag_processing(query)
            print(f"Requires RAG? {requires_rag}")
            if not requires_rag:
                # Generate a response without RAG context
                return await self.generate_final_response(query, "")

            # Step 2: Reformulate query for search
            search_terms = await self.extract_search_terms(query)

            # Step 3: Perform RAG search
            results = self.embedder.search(search_terms, top_k=3)
            for idx, score in results:
            # Step 4: Generate the final response
            context = self.format_rag_results(results)
            print(context)
            return await self.generate_final_response(query, context)

        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            return "An error occurred while processing your request."


async def main():
    agent = AdvancedRAGAgent()

    print("\n" + "="*40)
    print("  Advanced Email Assistant")
    print("="*40)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("👉 Your question: ")
            if query.lower() == "exit":
                break

            response = await agent.process_query(query)
            print("\n🤖 Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
