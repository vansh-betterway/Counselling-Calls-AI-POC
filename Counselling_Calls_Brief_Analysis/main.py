import os
import json
import csv
from sarvamai import SarvamAI
from openai import OpenAI

AUDIO_PATH = r"" 


OUTPUT_DIR = "outputs"
CSV_FILE = os.path.join(OUTPUT_DIR, "counselling_calls.csv")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

openai_client = OpenAI()


def extract_segment_text(segment):
    for key in ["text", "utterance", "transcript", "content", "value"]:
        if key in segment and isinstance(segment[key], str):
            return segment[key]
    return ""


def safe_json_from_text(text):
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(text[start:end])


def save_summary_to_txt(summary_text):
    summary_dir = os.path.join(OUTPUT_DIR, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    audio_filename = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
    path = os.path.join(summary_dir, f"summary_{audio_filename}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text.strip())

    print(f" Summary saved to: {path}")


def enforce_counsellor_name_rules(data):
    valid = {"aman", "saurav"}
    name = data.get("counsellor_name")

    if not name:
        data["counsellor_name"] = None
        return data

    if name.lower() not in valid:
        data["counsellor_name"] = None

    if data.get("patient_name") and data["counsellor_name"]:
        if data["patient_name"].lower() == data["counsellor_name"].lower():
            data["counsellor_name"] = None

    return data


def enforce_location_rules(data): 
    location = data.get("patient_location")

    if not location or not location.strip():
        data["patient_location"] = "N/A"
        return data

    loc = location.lower()

    if "gurgaon" in loc or "gurugram" in loc:
        data["patient_location"] = "Gurgaon"
        return data

    if "delhi" in loc:
        data["patient_location"] = "Delhi"
        return data

    data["patient_location"] = "Others"
    return data


def transcribe_with_sarvam():
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    job = client.speech_to_text_translate_job.create_job(
        model="saaras:v2.5",
        with_diarization=True,
        num_speakers=2,
        prompt="Healthcare counselling call"
    )

    job.upload_files([AUDIO_PATH])
    job.start()
    job.wait_until_complete()

    results = job.get_file_results()
    if not results.get("successful"):
        raise Exception(f"Sarvam failed: {results.get('failed')}")

    base = os.path.dirname(os.path.abspath(__file__))
    sarvam_out = os.path.join(base, "sarvam_outputs")
    os.makedirs(sarvam_out, exist_ok=True)

    job.download_outputs(sarvam_out)

    json_file = next(f for f in os.listdir(sarvam_out) if f.endswith(".json"))
    with open(os.path.join(sarvam_out, json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    os.remove(os.path.join(sarvam_out, json_file))

    return data["diarized_transcript"]


def extract_with_openai(diarized):
    conversation = "\n".join(
        f"{speaker}: " + " ".join(extract_segment_text(seg) for seg in segments)
        for speaker, segments in diarized.items()
    )

    prompt = f"""
You are analyzing a healthcare counselling call.

VERY IMPORTANT RULES:
- Counsellor name can ONLY be Aman or Saurav.
- If Aman or Saurav is mentioned anywhere, assign it as counsellor_name.
- Any other name CANNOT be counsellor.
- If neither Aman nor Saurav is mentioned, counsellor_name MUST be null.
- Counsellor usually introduces themselves at the START of the call.

LOCATION RULES:
- Extract patient_location ONLY if explicitly spoken.
- Do NOT guess location.

Return ONLY valid JSON in this exact format:
{{
  "patient_name": string | null,
  "counsellor_name": "Aman" | "Saurav" | null,
  "patient_location": string | null,
  "phone_number": string | null,
  "call_summary": string
}}

Conversation:
\"\"\"
{conversation}
\"\"\"
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured data from transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_text = response.choices[0].message.content
    return safe_json_from_text(raw_text)


def save_to_csv(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "Phone Number",
                "Counsellor Name",
                "Location",
                "Patient Name"
            ])

        writer.writerow([
            data.get("phone_number"),
            data.get("counsellor_name"),
            data.get("patient_location"),
            data.get("patient_name")
        ])


def main():
    print("Transcribing with Sarvam...")
    diarized = transcribe_with_sarvam()

    print("Extracting details with OpenAI...")
    extracted = extract_with_openai(diarized)

    extracted = enforce_counsellor_name_rules(extracted)
    extracted = enforce_location_rules(extracted)

    print("Saving to CSV...")
    save_to_csv(extracted)

    print("Saving summary to text file...")
    save_summary_to_txt(extracted["call_summary"])

    print("\n Successful")


if __name__ == "__main__":
    main()
