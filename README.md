# Lab Tool Usage Tracker

This project tracks the usage of laboratory equipment in scientific literature by analyzing bioRxiv/medRxiv preprints. It uses natural language processing to extract and resolve equipment mentions, assess novelty, and visualize trends over time.

## Features

- **Automated Harvesting**: Fetches recent preprints from bioRxiv/medRxiv in specified categories.
- **Novelty Classification**: Uses local zero-shot classifiers to identify novel wet-lab work.
- **Machine Extraction**: Rule-based approach to extract equipment mentions from methods sections.
- **Semantic Resolution**: Resolves raw machine names to canonical entities using a gazetteer and fuzzy matching.
- **PostgreSQL Storage**: Stores candidates, mentions, and metadata in a remote PostgreSQL database.
- **Interactive Dashboard**: Visualizes equipment adoption trends over time with filtering capabilities.

## Architecture

The pipeline consists of three main stages:

1. **Stage 1 & 2 (Harvesting)**: 
   - Fetches preprints from bioRxiv/medRxiv API
   - Filters by target categories and equipment-related keywords
   - Stores candidates in the `candidates` table

2. **Stage 3 (Processing)**:
   - **Novelty Classification**: Determines if preprint describes novel experimental work
   - **Machine Extraction**: Extracts equipment mentions from methods text
   - **Semantic Resolution**: Maps raw mentions to known equipment entities
   - Stores results in the `machine_mentions` table

3. **Dashboard**:
   - Provides interactive visualizations of equipment trends
   - Shows live data from PostgreSQL or falls back to demo data

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database (or set `DATABASE_URL` environment variable)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/g-torr/lab_tool_usage.git
   cd lab_tool_usage
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```env
   DATABASE_URL=your_postgresql_connection_string
   ```

   You can also use the provided `.env.example` as a template.

### Database Setup

The pipeline expects two tables to exist in your PostgreSQL database:
- `candidates`
- `machine_mentions`

You can create them by running the initialization script:
```bash
python -c "from src.db import init_db; init_db()"
```

Or if you prefer to let the pipeline handle it (not recommended for production as it will reset data):
```bash
# WARNING: This will drop existing tables!
python src/interactive_dashboard.py  # Without --append flag
```

For production use, we recommend initializing the database separately and then using the `--append` flag to preserve data.

## Usage

### Running the Full Pipeline

To run the complete harvesting and processing pipeline:

```bash
# For a fresh run (WARNING: clears existing data)
python src/interactive_dashboard.py

# To append new data without clearing existing records
python src/interactive_dashboard.py --append
```

### Running Just the Dashboard

To launch the interactive dashboard (uses existing data in the database):

```bash
python app.py
```

The dashboard will be available at http://localhost:7860

### Running Tests

```bash
python -m pytest
```

## Configuration

### Categories

Modify `TARGET_CATEGORIES` in `src/interactive_dashboard.py` to change which bioRxiv categories are processed.

### Equipment Keywords

Edit `BROAD_CATCHMENT` to adjust the keywords used for initial equipment detection.

### Exclusion Terms

Modify `EXCLUSION_TERMS` to filter out unwanted article types (reviews, meta-analyses, etc.).

### Dry Lab Triggers

Adjust `DRY_LAB_TRIGGERS` to better identify computational-only work.

## Deployment

### Render.com / Other Platforms

The dashboard is hosted at https://lab-tool-usage.onrender.com/

Use the following start command:
```bash
gunicorn app:server
```

## Project Structure

```
lab_tool_usage/
├── app.py                  # Dash dashboard application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── interactive_dashboard.py  # Main pipeline orchestrator
│   ├── app.py              # Dashboard (symlink or copy to root app.py)
│   ├── db.py               # Database connection helpers
│   ├── machine_extractor.py    # Equipment extraction logic
│   ├── novelty_classifier.py   # Novelty detection
│   ├── semantic_resolver.py    # Machine name resolution
│   ├── create_registry.py      # Equipment gazetteer creation
│   └── normalize_machine.py    # Machine name normalization
├── tests/                  # Unit tests
│   ├── test_machine_extractor.py
│   ├── test_novelty_classifier.py
│   ├── test_semantic_resolver.py
│   └── test_interactive_dashboard.py
├── data/                   # Local data (gitignored)
│   ├── machine_registry.csv
│   ├── machine_mentions.csv
│   ├── stage2_labels.json
│   └── unmatched.csv
├── db/                     # Database schemas (gitignored)
```

## How It Works

1. **Harvesting**: The pipeline queries bioRxiv/medRxiv for recent preprints in biomedical categories.
2. **Filtering**: Preprints are filtered for equipment-related keywords in title/abstract.
3. **Novelty Check**: Novel wet-lab work is identified using a local zero-shot classifier.
4. **Extraction**: Equipment mentions are extracted from methods sections using a rule-based approach.
5. **Resolution**: Raw mentions are matched to known equipment entities using fuzzy matching and a gazetteer.
6. **Storage**: Results are stored in PostgreSQL for analysis and visualization.
7. **Visualization**: The dashboard shows temporal trends and breakdowns of equipment usage.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- bioRxiv/medRxiv for providing open access to preprints
- The Hugging Face team for model hosting and inference API
- The open-source NLP community for spaCy, scispacy, and transformers