from sec_filing_pipeline import SECFilingPipeline
import os
# Initialize with API keys
pipeline = SECFilingPipeline(
    fmp_api_key=os.getenv("FMP_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Run the pipeline for a single symbol
successful, total = pipeline.run(
    symbol="AAPL",
    date_from="2021-01-01",
    date_to="2025-05-06",
    # form_types=["10-K"]
    # form_types=["10-Q"]
    # form_types=["8-K"]
    # form_types=["DEF 14A"]
    # form_types=["DEFA14A"]
    # form_types=["PX14A6G"]
    # form_types=["S-3"]
    form_types=["S-3ASR"]
)

print(f"Successfully processed {successful} out of {total} filings")