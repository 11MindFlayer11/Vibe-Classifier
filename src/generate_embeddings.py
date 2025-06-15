import pandas as pd
import os
from generate_embeddings import EmbeddingMaker
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def load_data(csv_path):
    """Load and validate the input data"""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["id", "image_url", "prod"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def main():
    # Setup
    setup_directories()

    # Paths
    input_csv = "data/catalog/images+id+prodtype.csv"
    output_index = "embeddings/product_embeddingsIP_new.index"
    output_ids = "embeddings/product_embedding_idsIP_new.csv"

    try:
        # Load data
        logger.info("Loading data...")
        images_df = load_data(input_csv)

        # Initialize embedding maker
        logger.info("Initializing EmbeddingMaker...")
        maker = EmbeddingMaker()

        # Generate embeddings
        logger.info("Generating embeddings...")
        final_embeddings = maker.generate_embeddings_from_df(images_df)

        # Save embeddings
        logger.info("Saving embeddings...")
        maker.save_to_faiss(
            final_embeddings, index_path=output_index, id_path=output_ids
        )

        logger.info(
            f"Successfully generated embeddings for {len(final_embeddings)} products"
        )
        logger.info(f"Embeddings saved to {output_index} and {output_ids}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
