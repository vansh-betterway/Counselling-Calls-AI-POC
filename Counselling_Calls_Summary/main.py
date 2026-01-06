import os
import json 
import csv
from pydub import AudioSegment

from sarvamai import SarvamAI
from openai import OpenAI

AUDIO_PATH = r""

OUTPUT_DIR = "result"
CSV_FILE = os.path.join(OUTPUT_DIR,"counselling_summary.csv")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
openai_client = OpenAI()


DELHI_LOCALITIES = {
    "karol bagh",
    "rohini",
    "dwarka",
    "janakpuri",
    "lajpat nagar",
    "rajouri garden",
    "pitampura",
    "saket",
    "vasant kunj",
    "connaught place",
    "cp",
    "south delhi",
    "north delhi",
    "east delhi",
    "west delhi"
}



def get_audio_duration(AUDIO_PATH):
    audio = AudioSegment.from_file(AUDIO_PATH)
    return len(audio) / 1000 



def extract_segment_text(segment):
    for key in ["text", "utterance", "transcript", "content", "value"]:
        if key in segment and isinstance(segment[key], str):
            return segment[key]
    return ""



def safe_json_from_text(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])



def save_summary(summary_txt):
    summary_dir = os.path.join(OUTPUT_DIR,"Summary")
    os.makedirs(summary_dir,exist_ok=True)
    audio_filename = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
    path = os.path.join(summary_dir, f"summary_{audio_filename}.txt")
    with open(path,"w",encoding="utf-8") as f:
        f.write(summary_txt.strip())
    print(f"Summary saved to:{path}")



def transcribe_with_sarvam():
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    job = client.speech_to_text_translate_job.create_job(
        model ="saaras:v2.5",
        with_diarization=True,
        num_speakers=2,
        prompt="Healthcare counselling call"
    )

    job.upload_files([AUDIO_PATH])
    job.start()
    job.wait_until_complete()

    results = job.get_file_results()
    if not results.get("successful"):
        raise Exception(f"Sarvam failed:{results.get('failed')}")
    base = os.path.dirname(os.path.abspath(__file__))
    sarvam_out = os.path.join(base,"sarvam_outputs")
    os.makedirs(sarvam_out,exist_ok=True)

    job.download_outputs(sarvam_out)

    json_file = next(f for f in os.listdir(sarvam_out) if f.endswith(".json"))
    with open(os.path.join(sarvam_out,json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    os.remove(os.path.join(sarvam_out,json_file))
    return data["diarized_transcript"]



def extract_with_openai(diarized):
    conversation = "\n".join(
        f"{speaker}: " + " ".join(extract_segment_text(seg) for seg in segments)
        for speaker, segments in diarized.items()
    )

    prompt = f"""
You are a healthcare counselling call analysis assistant.

STRICT RULES:
• Return ONLY valid JSON
• Use null if information is not clearly mentioned
• Do NOT guess or hallucinate

DEFINITIONS (VERY IMPORTANT):

1. location_asked:
   - true ONLY if the counsellor asks for the PATIENT’S PERSONAL or RESIDENTIAL location
   - The question must clearly be about where the patient LIVES
   - Examples of valid questions:
     • "Where do you live?"
     • "Which city are you from?"
     • "What is your residential location?"
   - false if the discussion is about:
     • clinic location
     • hospital address
     • center location
     • directions
     • “we are located in…”
     • “our clinic is in…”
   - If there is any doubt, return false

2. patient_location:
   - Extract ONLY if location_asked = true
   - If not mentioned → null

3. betterway_explained:
   - true ONLY if counsellor explains the PROCESS, such as:
     • how consultation works
     • doctor assignment
     • treatment flow
     • follow-up steps
   - Mere mention of the word "Betterway" is NOT enough

4. counsellor_name:
   - Can ONLY be "Aman" or "Saurav"

5. counsellor_summary:
   - Write a DETAILED internal summary from the counsellor’s perspective
   - Must include:
     1. Purpose of the call
     2. Patient’s concern or reason for inquiry
     3. Explanation provided by the counsellor
     4. Any actions taken or decisions discussed
     5. Next steps or outcome
   - Use 5–7 complete sentences
   - Professional tone
   - No bullet points
   - No generic filler text

6. patient_name:
   - Extract ONLY if a person EXPLICITLY states a name using phrases like:
     • "my name is ___"
     • "the patient's name is ___"
     • "her/his name is ___"
   - Do NOT extract names from:
     • questions
     • greetings
     • organization names
     • clinic names
     • service names
     • phrases like "calling from ___"
   - If there is any ambiguity, return null

7. doctor_allocated:
   - true ONLY if the counsellor CONFIRMS an appointment
   - This includes:
     • Explicit confirmation like:
       "Your appointment is booked with Dr. ___"
       "Your consultation with Dr. ___ is confirmed"
     • OR the call is clearly a confirmation call for an already booked appointment
   - false if:
     • doctor is mentioned only as part of process
     • assignment is future or conditional
     • counsellor explains services
     • patient asks about doctor availability

8. appointment_booked:
   - true ONLY if the counsellor CONFIRMS that an appointment or slot
     HAS BEEN booked during this call
   - Examples:
     • "Your appointment is booked"
     • "Your slot has been confirmed"
     • "This is a confirmation call for your appointment"
   - false if booking is future, conditional, or only discussed

9. disease:
   - Extract the health problem, condition, or disease described in the call
   - Write it briefly (1–5 words per issue)
   - Include multiple issues if mentioned
   - If no disease or health problem is discussed, return null

RETURN FORMAT (exact):
{{
  "patient_name": string | null,
  "counsellor_name": "Aman" | "Saurav" | null,
  "location_asked": boolean,
  "patient_location": string | null,
  "betterway_explained": boolean,
  "doctor_allocated": boolean,
  "counsellor_summary": string,
  "appointment_booked": boolean,
  "disease": string | null
}}

TRANSCRIPT:
\"\"\"
{conversation}
\"\"\"
"""


    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return safe_json_from_text(response.choices[0].message.content)



def finalize_fields(data):
    data["phone_number"] = os.path.splitext(os.path.basename(AUDIO_PATH))[0]

    duration = get_audio_duration(AUDIO_PATH)
    data["call_duration_gte_2_min"] = duration >= 120

    if data.get("counsellor_name") not in ["Aman", "Saurav"]:
        data["counsellor_name"] = None

    if not data.get("location_asked"):
        data["patient_location"] = "N/A"
    else:
        raw_loc = data.get("patient_location")
        if not raw_loc or not raw_loc.strip():
            data["patient_location"] = "N/A"
        else:
            loc = raw_loc.lower()

            if "gurgaon" in loc or "gurugram" in loc:
                data["patient_location"] = "Gurgaon"

            elif( loc.strip() == "delhi" 
                 or any(area in loc for area in DELHI_LOCALITIES)
            ):
                data["patient_location"] = "Delhi"

            else:
                data["patient_location"] = "Others"


    data["betterway_explained"] = bool(data.get("betterway_explained"))
    data["doctor_allocated"] = bool(data.get("doctor_allocated"))
    data["appointment_booked"] = bool(data.get("appointment_booked"))

    if data.get("appointment_booked"):
        data["doctor_allocated"] = True

    
    return data




def save_to_csv(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "Phone Number",
                "Patient Name",
                "Counsellor Name",
                "Call More Than 2 Min",
                "Location Asked ?",
                "Patient Location",
                "Betterway Explained ?",
                "Doctor Allocated ?",
                "Disease",
                "Appointment Booked ?"
            ])

        writer.writerow([
            data["phone_number"],
            data.get("patient_name"),
            data["counsellor_name"],
            data["call_duration_gte_2_min"],
            data["location_asked"],
            data["patient_location"],
            data["betterway_explained"],
            data["doctor_allocated"],
            data.get("disease"),
            data["appointment_booked"]
        ])




def main():
    print("Transcribing call")
    diarized = transcribe_with_sarvam()

    print("Extracting insights")
    extracted = extract_with_openai(diarized)

    print("Finalizing fields")
    final_data = finalize_fields(extracted)

    print("Saving CSV")
    save_to_csv(final_data)

    print("Saving summary to text file")
    save_summary(extracted["counsellor_summary"])

    print("Successful!!!")


if __name__ == "__main__":
    main()