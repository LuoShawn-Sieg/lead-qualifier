import streamlit as st
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # 免费版 API 推荐并发设为 2-3，配合重试机制最稳定
    max_workers = st.slider(
        "Concurrent Workers",
        min_value=1,
        max_value=5,
        value=2,
        help="Free API tier has rate limits. 2-3 workers is optimal."
    )


# --- Helper Function: Evaluate Lead via Gemini with Retry Logic ---
def evaluate_single_lead(client, lead_info: dict, criteria: str, max_retries: int = 4) -> dict:
    prompt = f"""
You are an expert B2B Lead Qualification Specialist.
Evaluate the following single lead record strictly against the provided qualification criteria.

[Qualification Criteria]:
{criteria}

[Lead Data]:
{json.dumps(lead_info, ensure_ascii=False, indent=2)}

[Output Requirements]:
Output ONLY a valid JSON object without any additional text or Markdown markers:
{{
  "Score": <integer between 0 and 100>,
  "Qualified": <"Yes" or "No">,
  "Reason": "<One concise sentence stating why this lead passed or failed the criteria>"
}}
"""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            parsed = json.loads(response.text.strip())
            return {
                "Score": parsed.get("Score", 0),
                "Qualified": parsed.get("Qualified", "No"),
                "Reason": parsed.get("Reason", "Evaluation completed")
            }
        except Exception as e:
            err_msg = str(e)
            # 遇到 429 频率限制或 503 临时不可用时自动等待重试
            if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # 指数退避增长等待时间
                    continue
            return {
                "Score": 0,
                "Qualified": "Error",
                "Reason": f"API error: {err_msg[:80]}..."
            }


# --- Main Dashboard Logic ---
st.title("🎯 AI B2B Lead Qualifier & Enrichment")
st.markdown("Upload your lead list (CSV) to automatically score, qualify, and annotate reasons.")

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

                results = [None] * len(df)
                records = df.to_dict(orient="records")

                status_text.text("Processing leads smoothly with rate-limit protection...")

                completed_count = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(evaluate_single_lead, client, record, criteria_input): i
                        for i, record in enumerate(records)
                    }

                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        results[index] = future.result()
                        completed_count += 1
                        progress = completed_count / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed: {completed_count}/{len(df)} leads...")

                results_df = pd.DataFrame(results)
                final_df = pd.concat([df, results_df], axis=1)

                status_text.text("✨ Batch processing completed!")
                st.success("All leads have been evaluated and scored.")

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
