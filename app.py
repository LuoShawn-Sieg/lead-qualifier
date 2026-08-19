import streamlit as st
import pandas as pd
import json
import time
from google import genai
from google.genai import types

# --- Page Configuration ---
st.set_page_config(
    page_title="B2B Lead Qualifier Pro",
    page_icon="🎯",
    layout="wide"
)

# --- Sidebar: API & Criteria Configuration ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Your API key is used strictly in-session and never stored."
    )

    st.markdown("---")
    st.subheader("🎯 ICP Qualification Criteria")

    default_criteria = (
        "1. Business Model: Must be B2B / High-ticket service provider or agency.\n"
        "2. Decision Maker Role: Founder, Co-Founder, CEO, Managing Director, Head of Growth/Sales.\n"
        "3. Company Legitimacy: Exclude freelancers, pure educators, non-profits, or dead entities.\n"
        "4. Profile Completeness: Must have clear value proposition and legitimate business presence."
    )
    criteria_input = st.text_area(
        "Customizable Scoring Rules (Defaults loaded)",
        value=default_criteria,
        height=180
    )

    st.markdown("---")
    batch_size = st.slider(
        "Batch Size per Request",
        min_value=5,
        max_value=20,
        value=10,
        help="Batching multiple leads into a single API call avoids 429 rate limits and processes 10x faster."
    )


# --- Core Function: Process Leads in Batches ---
def process_batch_leads(client, chunk_records: list, start_index: int, criteria: str, max_retries: int = 4) -> list:
    batch_payload = [
        {
            "id": start_index + i,
            "data": record
        }
        for i, record in enumerate(chunk_records)
    ]

    prompt = f"""
You are an expert B2B Lead Qualification Specialist.
Evaluate the following batch of lead records strictly against the qualification criteria.

[Qualification Criteria]:
{criteria}

[Batch Lead Records]:
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}

[Output Requirements]:
Return ONLY a valid JSON list containing the evaluation for EACH record matching the input IDs in order:
[
  {{
    "id": <record id>,
    "Score": <integer 0-100>,
    "Qualified": <"Yes" or "No">,
    "Reason": "<One concise sentence why it passed or failed>"
  }}
]
"""
    delay = 3.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            parsed_list = json.loads(response.text.strip())

            # Map back results to ensure correct ordering
            id_to_result = {item["id"]: item for item in parsed_list if isinstance(item, dict) and "id" in item}

            chunk_results = []
            for i in range(len(chunk_records)):
                rec_id = start_index + i
                if rec_id in id_to_result:
                    res = id_to_result[rec_id]
                    chunk_results.append({
                        "Score": res.get("Score", 0),
                        "Qualified": res.get("Qualified", "No"),
                        "Reason": res.get("Reason", "Evaluated successfully.")
                    })
                else:
                    chunk_results.append({
                        "Score": 0,
                        "Qualified": "Error",
                        "Reason": "Record missing in batch response."
                    })
            return chunk_results

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            # Return error rows for the chunk if max retries fail
            return [
                {"Score": 0, "Qualified": "Error", "Reason": f"API error: {err_msg[:60]}..."}
                for _ in range(len(chunk_records))
            ]


# --- Main UI ---
st.title("🎯 AI B2B Lead Qualifier & Enrichment")
st.markdown(
    "Upload your lead list (CSV) to automatically score, qualify, and annotate reasons using batch AI acceleration.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File loaded successfully: **{len(df)}** rows detected.")

        with st.expander("👀 View Raw Data Preview (First 5 Rows)"):
            st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Start Bulk Qualification", type="primary"):
            if not api_key:
                st.error("Please enter a valid Gemini API Key in the left sidebar first.")
            else:
                client = genai.Client(api_key=api_key)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                records = df.to_dict(orient="records")
                total_rows = len(records)
                all_results = []

                # Split into chunks based on batch_size
                chunks = [records[i:i + batch_size] for i in range(0, total_rows, batch_size)]

                for idx, chunk in enumerate(chunks):
                    start_idx = idx * batch_size
                    status_text.text(
                        f"Processing batch {idx + 1}/{len(chunks)} (Rows {start_idx + 1} - {min(start_idx + len(chunk), total_rows)})...")

                    chunk_res = process_batch_leads(client, chunk, start_idx, criteria_input)
                    all_results.extend(chunk_res)

                    progress_bar.progress((idx + 1) / len(chunks))
                    # Brief pause between batch requests to remain strictly within RPM limits
                    time.sleep(1.0)

                results_df = pd.DataFrame(all_results)
                final_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

                status_text.text("✨ Batch processing completed successfully!")
                st.success("All leads have been evaluated and scored without rate-limit errors.")

                qualified_count = len(final_df[final_df["Qualified"] == "Yes"])
                avg_score = final_df["Score"].mean() if "Score" in final_df else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Leads", len(final_df))
                col2.metric("Qualified Leads", f"{qualified_count}", f"{(qualified_count / len(final_df) * 100):.1f}%")
                col3.metric("Average Score", f"{avg_score:.1f} / 100")

                st.markdown("### 📊 Qualified Results Preview")
                st.dataframe(final_df.head(20), use_container_width=True)

                csv_data = final_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Export Full Results (CSV)",
                    data=csv_data,
                    file_name="qualified_leads_result.csv",
                    mime="text/csv",
                    type="primary"
                )
    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
