from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

set_seed(42)

logger = get_logger(__name__)

logger.info("Project initialized successfully")

print(config.get("project", "name"))
print(config.get("dataset", "batch_size"))

# This adds your project root to the system path programmatically
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))